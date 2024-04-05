
import json
import torch
import random

from dataset import MILEPgm, G2TPgm, BasePgm

OP_func = { "+" : lambda x,y: x+y,
            "-" : lambda x,y: x-y,
            "*" : lambda x,y: x*y,
            "/" : lambda x,y: x/y,
            # "^" : lambda x,y: x**y,
            "^" : lambda x,y: x**int(y) if abs(y)<10 else None,
            "=" : lambda x,y: x}
def try_OP(OP, param) :
    try :
        res = OP_func[OP](*param)
    except :
        res = None
    return res

def check_res(res, ans, _e=0.01) :
    try :
        return 1 if abs(res-ans) < _e else 0
    except :
        return None

def eval_mile_pgm(pgm, nums) :
    memorys = [[*MILEPgm.memory_const, *num, 
                *(0 for _ in range(MILEPgm.memory_width["number"] - len(num)))] 
                for num in nums]

    pp = 0
    while pp < len(pgm) :
        try :
            OP = MILEPgm.OP_i2t[pgm[pp]]
        except :
            '''when receiving unstructured program'''
            return [None for _ in nums]
        if OP in ("<None>", "<END>", "<UNKNOWN>") :
            break
        elif OP in ("<START>", ) :
            pp += MILEPgm.SEG
            continue
        for memory in memorys :
            try :
                memory.append(try_OP(OP, [memory[PA] for PA in pgm[pp+1:pp+MILEPgm.SEG]]))
            except :
                '''PA index out of range'''
                memory.append(None)
        pp += MILEPgm.SEG

    return [memory[-1] for memory in memorys]

def _get_num(p, nums) :
    if p[0] == "C" :
        v = float(p[2:])
        return [v for _ in nums]
    elif p[0] == "N" :
        i = int(p[2:])
        return [n[i] for n in nums]
    else :
        raise RuntimeError

def eval_base_pgm(pgm, nums) :
    stack = [None]
    end = 0
    for p in pgm :
        if p in BasePgm.OPs :
            stack.append(BasePgm.tokens_i2t[p])
            end -= 1
        else :
            try :
                n2 = _get_num(G2TPgm.tokens_i2t[p], nums)
            except :
                return [None for _ in nums]
            while type(stack[-1]) is list :
                n1 = stack.pop(-1)
                op = stack.pop(-1)
                n2 = [try_OP(op, ns) for ns in zip(n1,n2)]
            end += 1
            if end == 1 :
                return n2
            stack.append(n2)
    return [None for _ in nums]

def eval_tree_pgm(pgm, nums) :
    stack = [None]
    end = 0
    for p in pgm :
        if p < G2TPgm.num_start :
            stack.append(G2TPgm.tokens_i2t[p])
            end -= 1
        elif p < G2TPgm.unk :
            n2 = _get_num(G2TPgm.tokens_i2t[p], nums)
            while type(stack[-1]) is list :
                n1 = stack.pop(-1)
                op = stack.pop(-1)
                n2 = [try_OP(op, ns) for ns in zip(n1,n2)]
            end += 1
            if end == 1 :
                return n2
            stack.append(n2)
        else :
            break
    return [None for _ in nums]

def evaluator(dataset) :
    opt_iter = iter([((data:=dataset.data[idx])["ans_opt"], data["opt_cor"]) for idx in dataset.data_valid])
    def _eval(raw) :
        res = []
        raw = raw[::-1]
        while len(raw) > 0 :
            for ans_opt, opt_cor in zip(*next(opt_iter)) :
                ans = raw.pop(-1)
                diff = [(i, abs(ans - opt)) for i,opt in enumerate(ans_opt) if ans is not None and type(opt) in (int, float,)]
                if len(diff) == 0 :
                    opt = int(random.random()*5)
                else :
                    opt = min(diff, key=lambda x:x[1])[0]
                res.append(opt == opt_cor)
        return res
    return _eval

def infer_base(dataloader, _MILE_) :
    out_raw = []
    accu = [0, 0]
    accu_raw = []
    for batch in dataloader :
        _MILE_.set_data(batch)
        with torch.no_grad() :
            pgm = _MILE_()
        num, ans = batch["nums"], batch["ans"]
        for pgm, num, ans in zip(pgm, num, ans) :
            res = eval_base_pgm(pgm, num)
            out_raw.append(res)
            accur = [check_res(*ra) for ra in zip(res,ans)]
            accu[0] += sum(a for a in accur if a)
            accu[1] += len(accur)
            accu_raw += accur
            
        print(f"\rTesting on {accu[1]:>5d}    {accu[0]/accu[1]:>6.4f}", end = "")
    print(f"\rTest accuracy : {accu[0]:>5d}/{accu[1]:>5d}    {accu[0]/accu[1]:>6.4f}")
    return accu, accu_raw, out_raw

def infer_tree(dataloader, _MILE_) :
    out_raw = []
    accu = [0, 0]
    accu_raw = []
    for batch in dataloader :
        _MILE_.set_data(batch)
        with torch.no_grad() :
            pgm = _MILE_()
        num, ans = batch["nums"], batch["ans"]
        for pgm, num, ans in zip(pgm, num, ans) :
            res = eval_tree_pgm(pgm, num)
            out_raw.append(res)
            accur = [check_res(*ra) for ra in zip(res,ans)]
            accu[0] += sum(a for a in accur if a)
            accu[1] += len(accur)
            accu_raw += accur
            
        print(f"\rTesting on {accu[1]:>5d}    {accu[0]/accu[1]:>6.4f}", end = "")
    print(f"\rTest accuracy : {accu[0]:>5d}/{accu[1]:>5d}    {accu[0]/accu[1]:>6.4f}")
    return accu, accu_raw, out_raw

def infer_mile(dataloader, _MILE_, unstructured_pgm=False) :
    out_raw = []
    accu = [0, 0]
    accu_raw = []
    for batch in dataloader :
        _MILE_.set_data(batch)
        with torch.no_grad() :
            pgm = _MILE_()
        if not type(pgm) in (list, tuple,) :
            pgm = pgm.tolist()
        num, ans = batch["nums"], batch["ans"]
        for pgm, num, ans in zip(pgm, num, ans) :
            if unstructured_pgm :
                pgm = [1,0,0] + [c if i%3 == 0 else c-10 for i,c in enumerate(pgm)]
                if len(pgm) % MILEPgm.SEG != 0 :
                    pgm = pgm[:(len(pgm)//MILEPgm.SEG)*MILEPgm.SEG]
            res = eval_mile_pgm(pgm, num)
            out_raw.append(res)
            accur = [check_res(*ra) for ra in zip(res,ans)]
            accu[0] += sum(a for a in accur if a)
            accu[1] += len(accur)
            accu_raw += accur

        print(f"\rTesting on {accu[1]:>5d}    {accu[0]/accu[1]:>6.4f}", end = "")
    print(f"\rTest accuracy : {accu[0]:>5d}/{accu[1]:>5d}    {accu[0]/accu[1]:>6.4f}")
    return accu, accu_raw, out_raw

'''simplified version'''
def infer_mile_beam(dataloader, _MILE_, beam_size, vote=False) :
    accu = [0, 0]
    accu_raw = []
    out_raw = []
    if not vote :
        for batch in dataloader :
            _MILE_.set_data(batch)
            with torch.no_grad() :
                _MILE_.encoder(_MILE_.batch)
                pgm = _MILE_.decoder.forward_beam(_MILE_.batch, beam_size)
            num, ans = batch["nums"], batch["ans"]
            for pgm, num, ans in zip(pgm, num, ans) :
                accu[1] += len(ans)
                if len(pgm) == 0 :
                    continue
                res = eval_mile_pgm(pgm[0][1], num)
                accur = [check_res(*ra) for ra in zip(res,ans)]
                out_raw.append(res)
                accu[0] += sum(a for a in accur if a)
                accu_raw += accur
            print(f"\rTesting on {accu[1]:>5d}    {accu[0]/accu[1]:>6.4f}", end = "")
        print(f"\rTest accuracy : {accu[0]:>5d}/{accu[1]:>5d}    {accu[0]/accu[1]:>6.4f}")
        return accu, accu_raw, out_raw
    else :
        accu = [0, 0, 0]
        for batch in dataloader :
            _MILE_.set_data(batch)
            with torch.no_grad() :
                _MILE_.encoder(_MILE_.batch)
                pgm = _MILE_.decoder.forward_beam(_MILE_.batch, beam_size)
            num, ans = batch["nums"], batch["ans"]
            for pgm, num, ans in zip(pgm, num, ans) :
                accu[2] += len(ans)
                if len(pgm) == 0 :
                    continue
                res = eval_mile_pgm(pgm[0][1], num)
                accur = [check_res(*ra) for ra in zip(res,ans)]
                accu[0] += sum(a for a in accur if a)
                votes = [{} for _ in num]
                for _s,_pgm in pgm :
                    res = eval_mile_pgm(_pgm, num)
                    for r,v in zip(res, votes) :
                        r = round(r, 2) if r is not None else r
                        if not r in v :
                            v[r] = 0.0
                        v[r] += _s[0]
                res = [sorted(v.items(), key=lambda x:-x[1]) for v in votes]
                accur = [check_res(r[0][0], a) for r,a in zip(res,ans)]
                accu[1] += sum(a for a in accur if a)
                accu_raw += accur
                out_raw.append(res)

            print(f"\rTesting on {accu[2]:>5d}    {accu[0]/accu[2]:>6.4f}    {accu[1]/accu[2]:>6.4f}", end = "")
        print(f"\rTest accuracy : {accu[0]:>5d}/{accu[2]:>5d}    {accu[0]/accu[2]:>6.4f}",
                               f" {accu[1]:>5d}/{accu[2]:>5d}    {accu[1]/accu[2]:>6.4f}")
        return accu, accu_raw, out_raw
 
