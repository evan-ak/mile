
import math
import random

import torch
import torch.utils.data

import os
import sys
import fnmatch
import datetime

from config import *
from dataset import *
from mile import MILE
from test import infer_base, infer_tree, infer_mile, infer_mile_beam

def train(cfg) :
    print("MILE: initializing ...")
    print("Encode with:", cfg.encode_method)
    print("Decode with:", cfg.decode_method)
    _MILE_ = MILE(cfg).cuda()
    cfg.device = next(_MILE_.parameters()).device
    print("MILE: initialized")
    print(_MILE_)
    _MILE_.confirm_parameter_groups()
    print()

    print("Dataset: loading ...")    
    match cfg.test_metric :
        case "5flod" :
            random.seed(cfg.test_split_seed)
            mask_train = set(random.sample(range(23162), int(23162*0.8)))
            mask_test = set(range(23162)) - mask_train
        case "public" :
            mask_test = set(math23_public_test)
            mask_train = set(range(23162)) - mask_test
        case "ablspl" :
            abl_split_method = ("number", "formula")[0]
            abl_test_rate = 0.2
            mask_train, mask_test = make_ablation_split(cfg, method=abl_split_method, p_test=abl_test_rate)
    AUG = cfg.get("use_aug", False)
    dataset_train = DatasetMath(cfg, mode=("loadonly" if AUG else "train"), mask=mask_train)
    dataset_test = DatasetMath(cfg, mode="test", mask=mask_test)

    load_g2t = {"graph": cfg.encode_method == "graph",
                "tree": cfg.decode_method == "tree"}
    if any(load_g2t.values())  :
        RNT = cfg.get("g2t_rnt", False)
        dataset_train.load_g2t_data(cfg.path_g2t_raw, cfg.path_g2t_graph, to_load=load_g2t, revise_num_token=RNT)
        dataset_test.load_g2t_data(cfg.path_g2t_raw, cfg.path_g2t_graph, to_load=load_g2t, revise_num_token=RNT)
    if AUG :
        dataset_aug = DatasetMutated(dataset_train, weight=cfg.weight_aug)
        dataset_train, dataset_train_raw = dataset_aug, dataset_train
    else :
        dataset_sampled = torch.utils.data.Subset(dataset_train, dataset_train.data_valid)
        dataset_train, dataset_train_raw = dataset_sampled, dataset_train
    print("Dataset: loaded")
    print()

    '''load savepoint'''
    if cfg.mile_load_savepoint is not None :
        for name, group in cfg.mile_load_savepoint :
            for file in os.listdir(cfg.dir_mile_savepoint) :
                if fnmatch.fnmatch(file, name) :
                    load_from = cfg.dir_mile_savepoint + file
                    print("Save point:", load_from)
                    break
            else :
                load_from = None
                print("No matched save point:", cfg.dir_mile_savepoint + name)
            if load_from is not None :
                print("Parameters load:", group)
                _MILE_.load_savepoint(load_from, group)
                print("Save point: loaded")
            print()
    
    SAVE_EVERY = cfg.args.get("--saveevery", None)
    if SAVE_EVERY == -1 :
        SAVE_EVERY = None
        SAVE_LAST = True
    else :
        SAVE_LAST = False

    '''optimizers'''
    optimizerSGD = lambda p: torch.optim.SGD(p, lr=0.01)
    optimizerAdam = lambda p: torch.optim.Adam(p, lr=0.001, betas=(0.9, 0.995), eps=1e-9)
    optimizer_options = {"lstm": {"lr":0.001},
                         "graph": {"lr":0.001},
                         "bert": {"lr":0.0001},
                         "tree": {"lr":0.001},
                         "trans": {"lr":0.0001},
                         "mile": {"lr":0.0001},
                         "else": {"lr":0.001},
                         }
    optimizer = optimizerAdam(_MILE_.get_parameters(optimizer_options, lrscale=cfg.args.get("--lrscale", 1.0)))

    '''schedulers'''
    maxb = cfg.mile_train_batch
    minb = maxb//2
    ndecay = 2
    def lr_scheduler_base(b) :
        if b < minb :
            return 1.0
        else :
            return (0.2 ** (ndecay*(b-minb)//(maxb-minb)+1))
    batch_warmup = 200
    def lr_scheduler_warmup(b) :
        if b < batch_warmup :
            return b/batch_warmup
        elif b < minb :
            return 1.0
        else :
            return (0.2 ** (ndecay*(b-minb)//(maxb-minb)+1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler_base)

    '''data loader'''
    bs_default = 128
    num_wokers = 16
    extra_keys = list(itertools.chain(*(keys for flag,keys in 
                    (((cfg.encode_method.startswith("graph")), ["xl", "graph"]),
                        ((cfg.decode_method == "tree"), ["xnp", "xnm", "num_stacks", "yl"]),
                        ((cfg.decode_method == "mile"), ["xnp", "xnm"]),
                    ) if flag)))
    keys_train = ["x", "y", "w"] + extra_keys
    keys_test = ["x", "nums", "ans"] + extra_keys
    dataloaderFbs = lambda bs: (lambda x: torch.utils.data.DataLoader(x, batch_size=bs, pin_memory=True,
                                                                        collate_fn=DatasetMath.get_collate_fn(keys_train),
                                                                        shuffle=True, num_workers=num_wokers))
    dataloaderF = dataloaderFbs(cfg.args.get("--bs", bs_default))

    dataloader = dataloaderF(dataset_train)
    if cfg.debug_flag :
        dataloader_raw = torch.utils.data.DataLoader(dataset_train_raw, batch_size=256, collate_fn=DatasetMath.get_collate_fn(keys_test), 
                                                        shuffle=False, num_workers=0)
        dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=16, collate_fn=DatasetMath.get_collate_fn(keys_test), 
                                                        shuffle=False, num_workers=0)
        debug_datasets = [dataloader_raw, dataloader_test]
        test_every = 1000
        if cfg.decode_method == "tree" :
            random.seed("sub set seed 0")
            dataset_debug_trainsub = torch.utils.data.Subset(dataset_train_raw, random.sample(range(len(dataset_train_raw)), 1024))
            dataloader_rawsub = torch.utils.data.DataLoader(dataset_debug_trainsub, batch_size=1, collate_fn=DatasetMath.get_collate_fn(keys_test), 
                                                        shuffle=False, num_workers=num_wokers)
            dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, collate_fn=DatasetMath.get_collate_fn(keys_test), 
                                                        shuffle=False, num_workers=num_wokers)
            debug_datasets = [dataloader_rawsub, dataloader_test]
    else :
        test_every = None

    match cfg.decode_method :
        case "lstm" :
            _infer = lambda _dataloader, _MILE_ : infer_base(_dataloader, _MILE_)
        case "tree" :
            _infer = lambda _dataloader, _MILE_ : infer_tree(_dataloader, _MILE_)
        case "mile" :
            _infer = lambda _dataloader, _MILE_ : infer_mile(_dataloader, _MILE_, unstructured_pgm=False)

    '''training options'''
    adjust_train_on = {
                        0: ({"all":True}, []),
                        # 0: ({"bert":False, "lstm":True, "tree":True}, []), 1000: ({"bert":True}, []),
                        }
    if cfg.encode_method.startswith("bert") and cfg.decode_method == "lstm" :
        adjust_train_on = {0: ({"bert":False, "lstm":True, "tree":True}, []), 1000: ({"bert":True}, [])}
    if cfg.args.get("--reinit", False) :
        adjust_train_on[0][1].append(_MILE_.reinit_bert)
    def adjust_train(adj) :
        _MILE_.set_train(adj[0])
        for func in adj[1] :
            func()
    if 0 in adjust_train_on :
        adjust_train(adjust_train_on[0])

    _MILE_.train()
    print("MILE: training on", len(dataset_train), "samples")
    confirm_split(dataset_train, dataset_test)
    c_batch = 0
    accu = 1.0

    '''batch count is used instead of epoch because formula mutation changes the epoch size'''
    try :
        if cfg.args.get("--testonly", None) :
            finished = True
            raise StopIteration
        finished = False
        while True :
            for batch in dataloader :
                try :
                    _MILE_.set_data(batch)
                except:
                    print(batch)
                    raise
                optimizer.zero_grad()

                out = _MILE_()
                loss = _MILE_.backward()

                torch.nn.utils.clip_grad_norm_(_MILE_.parameters(), 0.1)
                optimizer.step()

                accu = _MILE_.get_accuracy()
                print(f"\r {c_batch:>6d}   ",
                        "accu:", "  ".join([f"{v:>6.4f}" for v in accu.values()]), "  ",
                        "loss:", "  ".join([f"{v:>8.6f}" for v in loss.values()]), end=" "*16)

                c_batch += 1
                scheduler.step()
                if c_batch % 100 == 0:
                    print()
                if test_every and c_batch % test_every == 0 :
                    _MILE_.eval()
                    with torch.no_grad() :
                        try :
                            for _dataloader in debug_datasets :
                                _infer(_dataloader, _MILE_)
                        except Exception as e :
                            print(e)
                            raise
                    _MILE_.train()
                if SAVE_EVERY is not None and c_batch % SAVE_EVERY == 0 :
                    timestr = datetime.datetime.now().strftime("%Y%m%d")+"_"+datetime.datetime.now().strftime("%H%M%S")
                    save_to = cfg.dir_mile_savepoint + "savepoint_" + "_".join([cfg.args.get("--name", "unnamed"), str(c_batch), timestr])
                    torch.save(_MILE_.state_dict(), save_to)
                    print("### saved to", save_to)
                if c_batch in adjust_train_on :
                    adjust_train(adjust_train_on[c_batch])
                    break
                if c_batch >= cfg.mile_train_batch :
                    finished = True
                    raise StopIteration
    except StopIteration :
        pass
    except KeyboardInterrupt :
        print("\nTraining interrupted.")
        pass
    else :
        finished = True
    finally :
        print()
        if finished and cfg.debug_flag :
            accu = 0.0
            _MILE_.eval()
            for b in [1,2,5,10,20] :
                print("beam size :", b)
                _accu,_,_ = infer_mile_beam(dataloader_test, _MILE_, beam_size=b, vote=True)            
                accu = max(accu, _accu[0]/_accu[2])
        if SAVE_LAST :
            timestr = datetime.datetime.now().strftime("%Y%m%d")+"_"+datetime.datetime.now().strftime("%H%M%S")
            save_to = ( cfg.dir_mile_savepoint + "savepoint_" + 
                        "_".join([cfg.args.get("--name", "unnamed"), str(c_batch), timestr, f"{accu:>6.4f}"]) )
            torch.save(_MILE_.state_dict(), save_to)
            print("### saved to", save_to)
        if not finished :
            raise
        return

def arg_parsing(args) :
    def _eval(s) :
        try :
            return eval(s)
        except :
            return s
    return {arg.split('=')[0]: (_eval(arg.split('=')[1]) if '=' in arg else True) for arg in args}

def print_config(cfg) :
    for k in dir(cfg) :
        if not k.startswith("__") and not callable(v:=getattr(cfg, k)) :
            print(f"{k:<32}", v)
    return

if __name__ == "__main__" :
    print('\n================================================================================================')
    cfg = TotalConfig()
    cfg.args_raw = sys.argv[1:]
    cfg.args = arg_parsing(cfg.args_raw)
    cfg.overwrite(cfg.args)
    print(datetime.datetime.now(), '\n')
    print("Args:", cfg.args)
    print_config(cfg)
    print()
    train(cfg)
    print('\n', datetime.datetime.now())
    print("Ends normaly:", ' '.join(sys.argv))
    print('================================================================================================\n')
