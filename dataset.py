
import os
import json
import copy
import pickle
import random
import itertools
import importlib

import numpy
import torch

class DatasetMath(torch.utils.data.Dataset) :
    def __init__(self, cfg, mode="loadonly", mask=None) :
        self.cfg = cfg
        self.load_data(cfg.path_data_reg)
        if mask :
            assert len(mask) > 0
            self.mask = sorted(mask)
            self.raw = [self.raw[i] for i in self.mask]
            self.masked = True
        else :
            self.masked = False
        self.N = len(self.raw)
        self.data = {}
        match mode :
            case "train" :
                pgm_max_len = cfg.program_max_len
                self.programs_base = [None if d["pgm_tree"] is None else tuple(d["pgm_tree"]) for d in self.raw]
                match cfg.decode_method :
                    case "lstm" :
                        '''preorder traversal raw'''
                        self.programs_ptr = [None if p is None else BasePgm.trans_t2i(p, pad=pgm_max_len) for p in self.programs_base]
                        self.data_valid = [i for i,p in enumerate(self.programs_ptr) if p is not None and len(p) <= pgm_max_len]
                        self.data["y"] = self.programs_ptr
                    case "tree" :
                        self.data_valid = list(range(self.N))
                        self.data["y"] = None
                    case "mile" :
                        self.programs_miler = [None if d["pgm_mile"] is None else d["pgm_mile"] for d in self.raw]
                        self.programs_mile = [None if p is None else MILEPgm.trans_t2i(p) for p in self.programs_miler]
                        self.data_valid = [i for i,p in enumerate(self.programs_mile) if p is not None and len(p) <= pgm_max_len]
                        self.data["y"] = self.programs_mile
                if "pgm_rate" in self.raw[0] :            
                    self.programs_rate = [d["pgm_rate"] for d in self.raw]
                    self.weight = [(len(d["idx"])**0.5 * d["pgm_rate"]) for d in self.raw]
                else :
                    self.programs_rate = [None] * len(self.raw)
                    self.weight = [1.0] * len(self.raw)
                self.data["w"] = self.weight
            case "test" | "loadonly" :
                pass

        if not cfg.encode_method in ("bert", "roberta") :
            self.data["x"] = [d["toks"] for d in self.raw]
        else :
            self.data["x"] = [d["toks_bert"] for d in self.raw]
        self.preapre_x(pad=-1)
        self.data["nums"] = [tuple(d["nums"]) for d in self.raw]
        self.data["ans"] = [tuple(d["ans"]) for d in self.raw]
    
    def load_data(self, path) :
        if not os.path.isfile(path) :
            print("data file does not exist, rebuilding...")
            try :
                path_reg = path.rsplit('/', 1)[0] + "/reg_mile.py"
                spec = importlib.util.spec_from_file_location("main", path_reg)
                reg = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(reg)
                print(f"running {path_reg} ...")
                reg(file_out=path)
            except :
                raise 
        with open(path, 'r') as f :
            self.raw = json.load(f)

    def load_g2t_data(self, path_raw, path_graph, to_load, raw_idx=True, revise_num_token=False) :
        with open(path_raw, 'rb') as handle :
            raw = pickle.load(handle)
        with open(path_graph, 'rb') as handle :
            graph_nonzero = numpy.load(handle)
        N = graph_nonzero[0][-1]+1
        L = len(raw[0][0])
        graph = numpy.zeros((N, 5, L, L), dtype=bool)
        graph[tuple((d for d in graph_nonzero))] = True
        if raw_idx :
            idx_trans = [r["idx"][0] for r in self.raw]
            _raw = raw
            raw = (_raw[i] for i in idx_trans)
            graph = graph[idx_trans]
        elif self.masked :
            _raw = raw
            raw = (_raw[i] for i in self.mask)
            graph = graph[self.mask]
        raws = list(zip(*raw))
        self.tokens_g2t = raws[0]
        self.g2t_xl = raws[1]
        self.programs_g2t = raws[2]
        self.g2t_yl = raws[3]
        self.num_stacks = raws[5]
        self.graph = graph
        if revise_num_token :
            n_MAX = MILEPgm.memory_width["number"]
            n_TOK_0 = MILEPgm.NUM_TOKEN_RANGE[0]
            for line in self.tokens_g2t :
                n_count = 0
                for i,tok in enumerate(line) :
                    if tok == n_TOK_0 :
                        line[i] = n_TOK_0 + n_count
                        n_count += 1
                    if tok == 0 or n_count >= n_MAX :
                        break
        if to_load["graph"] :
            self.data["x"] = self.tokens_g2t
            self.data["xl"] = self.g2t_xl
            self.preapre_x(pad=-1)
        if to_load["tree"] :
            self.data["y"] = self.programs_g2t
            self.data["yl"] = self.g2t_yl
        self.data["num_stacks"] = self.num_stacks
        self.data["graph"] = self.graph

    def preapre_x(self, pad=False) :
        if pad :
            if pad == -1 :
                pad = max([len(x) for x in self.data["x"]])
            self.data["x"] = [x+[0]*(pad-len(x)) for x in self.data["x"]]
        '''x number position'''
        self.data["xnp"] = []
        '''x number mask'''
        self.data["xnm"] = []
        for x in self.data["x"] :
            xnp = [i for i,xi in enumerate(x) if MILEPgm.is_number(xi)][:MILEPgm.memory_width["number"]]
            nn = len(xnp)
            xnp = xnp + [0]*(MILEPgm.memory_width["number"]-nn)
            self.data["xnp"].append(xnp)
            xnm = [True]*nn + [False]*(MILEPgm.memory_width["number"]-nn)
            self.data["xnm"].append(xnm)

    '''all the index are from raw index in dataset'''
    def __getitem__(self, index) :
        return {k:d[index] for k,d in self.data.items()}

    def __len__(self) :
        return self.N

    @staticmethod
    def get_collate_fn(keys=None) :
        def _collate(data, kept=False) :
            if kept :
                return data            
            _d0 = data
            while (datatype:=type(_d0:=_d0[0])) is list :
                continue
            if datatype is None :
                return None
            elif datatype is int :
                return torch.tensor(data, dtype=torch.long)#.pin_memory()
            elif datatype is float :
                return torch.tensor(data, dtype=torch.float)#.pin_memory()
            elif datatype is bool :
                return torch.tensor(data, dtype=torch.bool)#.pin_memory()
            elif datatype in (str, tuple, dict) :
                return data
            elif isinstance(data[0], numpy.ndarray) :
                return torch.tensor(numpy.stack(data, axis=0), dtype=torch.long)#.pin_memory()
            elif isinstance(data[0], torch.Tensor) :
                return torch.stack(data, dim=0)
            else :
                print("unknown data type in batch:", datatype)
                assert False
        _keys_ketp = ("nums", "ans", "num_stacks", "xl", "yl")
        def collate_fn(data) :
            _keys = data[0].keys() if keys is None else keys
            return {k:_collate([d[k] for d in data], k in _keys_ketp) for k in _keys}
        return collate_fn

class DatasetMutated(torch.utils.data.Dataset) :
    def __init__(self, base, mr=None, weight=False) :
        self.data = {k:[] for k in base.data.keys()}
        self.data["y"] = []
        self.data["w"] = []
        keys_copy = self.data.keys() - {"y", "w"}
        self.mr = mr if mr is not None else (3,15)
        match base.cfg.decode_method :
            case "lstm" :
                _trans = lambda ps: [BasePgm.from_mile(p, pad=base.cfg.program_max_len)[0] for p in ps]
            case "tree" :
                _trans = lambda ps: [G2TPgm.from_mile(p, pad=base.cfg.program_max_len)[0] for p in ps]
            case "mile" :
                _trans = lambda ps: [MILEPgm.trans_t2i(p) for p in ps]
        for idx in range(len(base)) :
            pgm_v21_raw = base.raw[idx]["pgm_mile"]
            if pgm_v21_raw is None :
                continue
            # datacopy = base[idx]
            pms = self.mutate(pgm_v21_raw)
            # pms = [MILEPgm.trans_t2i(pm) for pm in pms]
            pms = _trans(pms)
            pms = [pm for pm in pms if pm is not None]
            N = len(pms)
            if N == 0 :
                continue
            # for k,v in datacopy.items() :
            for k in keys_copy :
                self.data[k] += [base[idx][k]]*N
            self.data["w"] += ([1/N]*N if weight else [1.]*N)
            self.data["y"] += pms
        self.N = len(self.data["x"])
        # if base.cfg.decode_method == "lstm" :
        #     self["y"] = [BasePgm.from_mile(p, pad=base.cfg.program_max_len) for p in self.data["y"]]
        # elif base.cfg.decode_method == "tree" :
        #     self["y"] = [G2TPgm.from_mile(p, pad=base.cfg.program_max_len) for p in self.data["y"]]

    def mutate(self, pgm) :
        L = len(pgm)
        if L<self.mr[0] or L>self.mr[1] :
            return [pgm]
        t = []
        for i in range(0,L,3) :
            if pgm[i] in ('+','*') :
                t.append([pgm[i:i+3],[pgm[i],pgm[i+2],pgm[i+1]]])
            else :
                t.append([pgm[i:i+3]])
        return [list(itertools.chain(*p)) for p in itertools.product(*t)]

    def __getitem__(self, index) :
        return {k:d[index] for k,d in self.data.items()}

    def __len__(self) :
        return self.N

class MILEPgm() :
    OP_i2t = ["<None>", "<START>", "<END>", "<UNKNOWN>", "+", "-", "*", "/", "^", "=",]
    OP_t2i = {k:i for i,k in enumerate(OP_i2t)}
    Consts = [0, 1, 2, 3.14, 4, 5, 10, 100]
    SEG = 3

    @classmethod
    def init(cls, cfg) :
        cls.memory_width = cfg.memory_width
        match cfg.encode_method :
            case "lstm" | "graph" :
                cls.NUM_TOKEN_RANGE = (4, 4+cfg.memory_width["number"])
                cls.is_number = lambda x: (x>=cls.NUM_TOKEN_RANGE[0] and x<cls.NUM_TOKEN_RANGE[1])
            case "bert" | "roberta" :
                cls.NUM_TOKENS = set([121,122,123,124,125,126,127,128,129,130,8108,8111,8110,8124,8122,8115,8121,8126,8123,8131,8113][:cfg.memory_width["number"]])
                cls.is_number = lambda x: x in cls.NUM_TOKENS
            case _ :
                raise NotImplementedError

        cls.max_pad = 10
        cls.max_raw_len = cls.SEG*cls.max_pad
        cls.memory_const = [0, *cls.Consts, *(0 for _ in range(cls.memory_width["const"] - len(cls.Consts)))]

        cls.PA_i2t = [*(f"C_{c}" for c in cls.Consts),
                      *(None for _ in range(cfg.memory_width["const"] - len(cls.Consts))),
                      *(f"N_{i:>02d}" for i in range(cfg.memory_width["number"])),
                      *(f"M_{i:>02d}" for i in range(cfg.memory_width["free"])),
                      ]
        cls.PA_t2i = {v:i+1 for i,v in enumerate(cls.PA_i2t)} | {None:0}
        cls.tokens_t2i = cls.OP_t2i | cls.PA_t2i

    @classmethod
    def trans_t2i(cls, _pgm, pad=None) :
        max_len = cls.max_raw_len if pad is None else pad*cls.SEG
        pgm = _pgm[:max_len]
        if len(unk:=[t for t in pgm if not t in cls.tokens_t2i]) :
            print(f"unknown token {unk} in program {_pgm}")
        out = [cls.tokens_t2i.get(t, 0) for t in pgm]
        out = [1,0,0] + out + [2,0,0] + [0]*(max_len - len(out))
        return out

    @classmethod
    def trans_old(cls, pgm, pad=None) :
        try :
            max_len = cls.max_raw_len if pad is None else pad*cls.SEG
            pgm = pgm[:max_len]
            out = [cls.OP_t2c[t] if i%cls.SEG == 0 else cls.PA_t2c[t] for i,t in enumerate(pgm)]
            return [1,0,0] + out + [2,0,0] + [0]*(max_len - len(out))
        except Exception as e:
            print("program tokenizing failed :", e)
            print(pgm)
            return None

    '''this must be conducted on raw program'''
    @staticmethod
    def pgm_traverse(pgm, idx) :
        if pgm[idx] == '=' :
            return [pgm[idx+1]]
        out = [pgm[idx]]
        for tok in pgm[idx+1:idx+3] :
            if tok[0] != 'M' :
                out += [tok, None]
            else :
                out += MILEPgm.pgm_traverse(pgm, 3*int(tok[2:]))
        return out

class G2TPgm() :
    OPs = ["*", "-", "+", "/", "^"]
    Consts = ["1", "3.14"]
    NNumbers = 15

    @classmethod
    def init(cls) :
        tokens = [*cls.OPs, *(f"C_{c}" for c in cls.Consts), *("N_"+str(i).zfill(2) for i in range(cls.NNumbers)), "UNK"]
        cls.tokens_i2t = tokens
        cls.tokens_t2i = {v:i for i,v in enumerate(cls.tokens_i2t)}
        cls.num_start = len(cls.OPs)
        cls.unk = len(tokens)-1
        cls.raw_pgm_tokens = [*("N_"+str(i).zfill(2) for i in range(10)), *(f"C_{c}" for c in (1,2,3.14,100)), *('+', '-', '*', '/')]
    
    @classmethod
    def trans_t2i(cls, pgm) :
        return [cls.tokens_t2i.get(t, cls.unk) for t in pgm]
    
    @classmethod
    def from_mile(cls, pgm, pad=None) :
        out = MILEPgm.pgm_traverse(pgm, len(pgm)-3)
        out = [cls.tokens_t2i.get(t, cls.unk) for t in out if t is not None]
        yl = len(out)
        if pad is not None :
            out = out + [0]*(pad-yl)
        return out,yl

class BasePgm() :
    OPs = ["*", "-", "+", "/", "^"]
    Consts = ["1", "3.14"]
    NNumbers = 15

    @classmethod
    def init(cls, cfg) :
        cls.tokens_i2t = ["<None>", "<START>", "<END>", "<UNKNOWN>", None,
                          *cls.OPs,
                          *(f"N_{i:>02d}" for i in range(cfg.memory_width["number"])),
                          *(f"C_{c}" for c in cls.Consts),
                          ]
        cls.tokens_t2i = {v:i for i,v in enumerate(cls.tokens_i2t)}
        cls.unk = cls.tokens_t2i["<UNKNOWN>"]
    
    @classmethod
    def trans_t2i(cls, pgm, pad=None) :
        out = [1, *(cls.tokens_t2i.get(t, cls.unk) for t in pgm), 2]
        if pad is not None :
            out = out + [0]*(pad-len(out))
        return out

    @classmethod
    def from_mile(cls, pgm, pad=None) :
        out = MILEPgm.pgm_traverse(pgm, len(pgm)-3)
        out = [cls.tokens_t2i.get(t, cls.unk) for t in out if t is not None]
        yl = len(out)
        if pad is not None :
            out = out + [0]*(pad-yl)
        return out,yl


def make_ablation_split(cfg, method="number", n_test=None, p_test=None) :
    assert n_test or p_test
    dataset = DatasetMath(cfg, mode="loadonly")
    match method :
        case "number" :
            _m = lambda dr: len(dr["nums"][0])
        case "formula" :
            _m = lambda dr: len(dr["pgm"]) if dr["pgm"] is not None else float("inf")
        case "qlength" :
            _m = lambda dr: (dr["toks"]+[2]).index(2)
        case _ :
            raise NotImplementedError
    idxs = sorted(((i, _m(dr)) for i,dr in enumerate(dataset.raw)), key=lambda x:-x[1])
    N = len(dataset)
    if n_test is not None :
        assert n_test > 0 and n_test < N
    else :
        n_test = int(N * p_test)
    mask_test, mask_train = [d[0] for d in idxs[:n_test]], [d[0] for d in idxs[n_test:]]
    return mask_train, mask_test

def confirm_split(_train, _test) :
    def _clip(x) :
        return x[:x.index(0)] if 0 in x else x
    _train = [(tuple(_clip(data["x"])), tuple(data["nums"][0])) for data in (_train[i] for i in range(len(_train)))]
    _train_qa, _train_q = set(_train), set((x[0] for x in _train))
    _test = [(tuple(_clip(data["x"])), tuple(data["nums"][0])) for data in (_test[i] for i in range(len(_test)))]
    _test_qa, _test_q = set(_test), set((x[0] for x in _test))
    print('\n'.join([f"{t.rjust(17)} : {n}" for t,n in 
                    [("count of train", len(_train)),
                     ("different q&n", len(_train_qa)),
                     ("different q", len(_train_q)),
                     ("count of test", len(_test)),
                     ("different q&n", len(_test_qa)),
                     ("different q", len(_test_q)),
                     ("intersection q&n", len(set.intersection(_train_qa,_test_qa))),
                     ("intersection q", len(set.intersection(_train_q,_test_q))),
                    ]]))

math23_public_test = [29, 63, 74, 146, 154, 199, 210, 215, 222, 284, 307, 310, 335, 358, 361, 363, 415, 418, 455, 486, 493, 495, 502, 505, 586, 615, 639, 714, 720, 724, 763, 798, 843, 844, 902, 934, 946, 953, 963, 996, 1026, 1049, 1056, 1115, 1156, 1222, 1251, 1283, 1291, 1305, 1313, 1324, 1358, 1360, 1375, 1380, 1386, 1409, 1481, 1488, 1495, 1536, 1557, 1568, 1609, 1650, 1662, 1674, 1696, 1708, 1714, 1733, 1763, 1768, 1777, 1782, 1830, 1837, 1928, 1955, 1987, 1997, 2003, 2006, 2012, 2088, 2105, 2150, 2153, 2181, 2187, 2261, 2275, 2282, 2306, 2338, 2348, 2398, 2419, 2460, 2470, 2476, 2516, 2518, 2529, 2553, 2556, 2572, 2637, 2638, 2651, 2659, 2692, 2695, 2727, 2739, 2748, 2790, 2803, 2808, 2841, 2847, 2938, 2969, 2989, 2994, 3004, 3016, 3026, 3092, 3109, 3127, 3170, 3180, 3182, 3230, 3236, 3266, 3301, 3350, 3356, 3371, 3389, 3401, 3484, 3527, 3552, 3553, 3613, 3614, 3622, 3672, 3681, 3697, 3703, 3728, 3787, 3801, 3852, 3873, 3899, 3903, 3925, 3940, 3946, 3952, 3981, 3986, 3988, 4022, 4054, 4079, 4094, 4113, 4118, 4140, 4166, 4201, 4205, 4206, 4219, 4234, 4249, 4304, 4316, 4351, 4360, 4376, 4395, 4454, 4515, 4523, 4526, 4535, 4546, 4586, 4593, 4596, 4604, 4727, 4745, 4763, 4777, 4795, 4891, 4928, 4953, 5003, 5031, 5041, 5046, 5057, 5075, 5124, 5128, 5131, 5155, 5174, 5188, 5197, 5225, 5245, 5270, 5293, 5338, 5352, 5356, 5399, 5402, 5410, 5418, 5425, 5430, 5431, 5434, 5435, 5444, 5459, 5493, 5535, 5601, 5622, 5629, 5634, 5672, 5735, 5781, 5805, 5811, 5815, 5824, 5836, 5863, 5887, 5889, 5892, 5958, 6005, 6027, 6047, 6072, 6086, 6092, 6154, 6182, 6190, 6211, 6218, 6242, 6258, 6267, 6276, 6300, 6302, 6398, 6425, 6441, 6454, 6461, 6473, 6480, 6486, 6531, 6534, 6545, 6570, 6603, 6616, 6654, 6655, 6687, 6694, 6707, 6715, 6737, 6741, 6750, 6785, 6793, 6833, 6859, 6948, 7033, 7042, 7067, 7088, 7090, 7092, 7093, 7127, 7146, 7155, 7191, 7196, 7199, 7225, 7231, 7262, 7286, 7305, 7314, 7315, 7331, 7341, 7348, 7360, 7387, 7401, 7443, 7466, 7538, 7556, 7659, 7682, 7691, 7693, 7715, 7722, 7736, 7760, 7798, 7812, 7814, 7834, 7851, 7873, 7880, 7888, 7970, 7983, 8032, 8055, 8067, 8073, 8090, 8096, 8110, 8121, 8130, 8135, 8145, 8166, 8172, 8173, 8195, 8196, 8276, 8334, 8377, 8397, 8400, 8412, 8455, 8456, 8466, 8478, 8528, 8552, 8653, 8677, 8730, 8741, 8748, 8752, 8777, 8783, 8786, 8789, 8791, 8820, 8860, 9052, 9055, 9070, 9071, 9080, 9086, 9145, 9156, 9160, 9258, 9280, 9285, 9289, 9296, 9390, 9399, 9405, 9457, 9458, 9468, 9483, 9549, 9583, 9591, 9604, 9618, 9620, 9713, 9718, 9750, 9757, 9763, 9787, 9816, 9821, 9853, 9890, 9897, 9899, 9987, 10003, 10009, 10022, 10044, 10072, 10104, 10174, 10180, 10235, 10330, 10342, 10364, 10368, 10371, 10438, 10464, 10523, 10536, 10567, 10587, 10606, 10628, 10675, 10688, 10689, 10758, 10789, 10803, 10892, 10897, 11033, 11042, 11046, 11047, 11118, 11151, 11229, 11234, 11250, 11267, 11301, 11316, 11346, 11377, 11417, 11420, 11430, 11432, 11501, 11502, 11510, 11520, 11523, 11542, 11584, 11611, 11618, 11625, 11627, 11651, 11654, 11660, 11724, 11735, 11736, 11742, 11749, 11804, 11834, 11847, 11859, 11861, 11919, 11979, 12024, 12073, 12081, 12084, 12091, 12127, 12138, 12197, 12203, 12236, 12241, 12251, 12257, 12356, 12363, 12390, 12400, 12402, 12420, 12424, 12466, 12524, 12536, 12549, 12553, 12633, 12656, 12698, 12745, 12756, 12778, 12782, 12791, 12812, 12849, 12977, 13009, 13040, 13062, 13212, 13230, 13345, 13354, 13390, 13391, 13410, 13433, 13438, 13477, 13503, 13515, 13536, 13539, 13592, 13613, 13652, 13694, 13743, 13751, 13755, 13764, 13796, 13813, 13846, 13851, 13867, 13917, 13929, 13931, 13960, 13978, 14010, 14070, 14077, 14166, 14204, 14207, 14231, 14258, 14295, 14304, 14319, 14399, 14407, 14435, 14463, 14592, 14605, 14665, 14707, 14727, 14738, 14745, 14791, 14804, 14814, 14860, 14889, 14903, 14915, 14922, 14933, 14939, 14951, 14954, 14977, 14980, 14999, 15036, 15066, 15079, 15097, 15129, 15160, 15171, 15172, 15191, 15194, 15209, 15215, 15232, 15249, 15254, 15279, 15310, 15316, 15323, 15326, 15356, 15374, 15423, 15447, 15475, 15478, 15480, 15483, 15487, 15497, 15503, 15533, 15554, 15569, 15581, 15591, 15597, 15632, 15656, 15688, 15744, 15772, 15776, 15807, 15840, 15852, 15882, 15959, 15972, 15996, 16000, 16005, 16028, 16037, 16057, 16081, 16102, 16118, 16146, 16170, 16204, 16290, 16358, 16363, 16375, 16402, 16505, 16519, 16563, 16567, 16573, 16584, 16592, 16608, 16629, 16634, 16660, 16671, 16673, 16688, 16821, 16840, 16850, 16901, 16908, 16917, 16950, 16959, 16969, 16985, 17010, 17025, 17060, 17082, 17107, 17110, 17111, 17118, 17122, 17135, 17190, 17217, 17240, 17261, 17297, 17319, 17334, 17339, 17361, 17415, 17416, 17423, 17433, 17454, 17487, 17550, 17557, 17597, 17602, 17635, 17685, 17702, 17703, 17744, 17757, 17781, 17820, 17821, 17847, 17865, 17914, 17921, 17943, 17994, 18015, 18019, 18027, 18045, 18053, 18089, 18115, 18133, 18135, 18148, 18153, 18168, 18203, 18232, 18244, 18295, 18299, 18310, 18320, 18334, 18338, 18363, 18371, 18400, 18422, 18441, 18448, 18452, 18474, 18489, 18505, 18509, 18546, 18622, 18630, 18702, 18711, 18729, 18739, 18740, 18756, 18810, 18828, 18829, 18831, 18852, 18903, 18906, 18915, 18938, 18989, 19084, 19115, 19185, 19206, 19231, 19268, 19279, 19299, 19305, 19332, 19333, 19336, 19361, 19409, 19423, 19428, 19431, 19525, 19544, 19547, 19552, 19576, 19582, 19615, 19618, 19628, 19644, 19652, 19655, 19668, 19673, 19702, 19739, 19827, 19845, 19892, 19911, 20020, 20039, 20045, 20067, 20112, 20141, 20144, 20160, 20163, 20174, 20183, 20231, 20244, 20279, 20318, 20329, 20344, 20390, 20445, 20449, 20452, 20457, 20462, 20473, 20475, 20483, 20503, 20531, 20544, 20570, 20584, 20593, 20698, 20705, 20736, 20766, 20815, 20817, 20829, 20832, 20929, 20932, 21028, 21040, 21055, 21115, 21151, 21155, 21197, 21215, 21222, 21226, 21229, 21257, 21266, 21278, 21282, 21287, 21330, 21334, 21341, 21342, 21353, 21420, 21425, 21426, 21447, 21448, 21468, 21469, 21476, 21482, 21520, 21552, 21558, 21562, 21570, 21598, 21614, 21677, 21698, 21700, 21717, 21722, 21728, 21741, 21755, 21760, 21763, 21764, 21777, 21800, 21804, 21813, 21834, 21835, 21841, 21846, 21856, 21870, 21910, 21918, 21973, 21993, 21994, 22073, 22090, 22101, 22139, 22145, 22154, 22158, 22178, 22256, 22259, 22277, 22282, 22299, 22300, 22322, 22324, 22329, 22361, 22368, 22375, 22420, 22430, 22450, 22464, 22518, 22526, 22654, 22660, 22670, 22675, 22703, 22716, 22722, 22745, 22759, 22767, 22780, 22785, 22786, 22807, 22825, 22832, 22840, 22846, 22852, 22915, 22929, 22947, 22970, 22975, 23009, 23033, 23053, 23054, 23064, 23070, 23074, 23147, 23150, 23155]
