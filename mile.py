
import itertools
import copy
import torch
import g2t_model
from transformers import BertConfig, BertModel

from dataset import MILEPgm, G2TPgm, BasePgm

'''Memory-Interactive Learning Engine'''
class MILE(torch.nn.Module) :
    def __init__(self, cfg) :
        super().__init__()
        MILEPgm.init(cfg)
        self.cfg = cfg
        self.grouped_parameters = {k:{} for k in ("lstm","graph","bert","trans","tree","mile","encoder","decoder","all")}

        self.encode_method = cfg.encode_method
        if self.encode_method == "lstm" :
            self.encoder = EncoderLSTM(cfg)
            self.grouped_parameters["lstm"].update({n:p for n,p in self.named_parameters()})
        elif self.encode_method in ["bert", "roberta"] :
            self.encoder = EncoderBert(cfg)
            self.grouped_parameters["bert"].update({n:p for n,p in self.named_parameters() if n.startswith("encoder.bert")})
            self.grouped_parameters["lstm"].update({n:p for n,p in self.named_parameters() if n.startswith("encoder.lstm")})
        elif self.encode_method == "graph" :
            self.encoder = EncoderGraph(cfg)
            self.grouped_parameters["graph"].update({n:p for n,p in self.named_parameters() if n.startswith("encoder.graph")})
            self.grouped_parameters["lstm"].update({n:p for n,p in self.named_parameters() if n.startswith("encoder.lstm")})
        self.grouped_parameters["encoder"].update({n:p for n,p in self.named_parameters()})

        self.decode_method = cfg.decode_method
        if self.decode_method == "lstm" :
            self.decoder = DecoderLSTM(cfg)
            self.grouped_parameters["lstm"].update({n:p for n,p in self.named_parameters() if n.startswith("decoder")})
        elif self.decode_method == "tree" :
            self.decoder = DecoderTree(cfg)
            self.grouped_parameters["tree"].update({n:p for n,p in self.named_parameters() if n.startswith("decoder")})
        elif self.decode_method == "mile" :
            self.decoder = DecoderMILE(cfg)
            self.grouped_parameters["trans"].update({n:p for n,p in self.named_parameters() if any((n.startswith(pfx) for pfx in 
                                                                                            ("decoder.number_embedding", "decoder.decoder_init")))})
            self.grouped_parameters["mile"].update({n:p for n,p in self.named_parameters() if n.startswith("decoder") and 
                                                                                            not n in self.grouped_parameters["trans"]})
        self.grouped_parameters["decoder"].update({n:p for n,p in self.named_parameters() if n.startswith("decoder")})
        self.grouped_parameters["all"].update({n:p for n,p in self.named_parameters()})

        self.batch = {}
        
    def load_savepoint(self, savepoint, load_group) :
        param_load = set().union(*(self.grouped_parameters[g].keys() for g in load_group))
        state_load = param_load.union((name for name,_ in self.named_buffers()))

        state_dict_load = torch.load(savepoint)
        state_dict_self = self.state_dict()
        for name,param in state_dict_load.items() :
            if name in param_load :
                state_dict_self[name].copy_(param)
                # print("param loaded :", name)
            else :
                # print("param unloaded :", name)
                pass
        print(param_load)

    def reinit_bert(self) :
        print("reinit ...")
        for n,p in self.named_parameters() :
            if "encoder.bert" in n :
                inited_by = None
                if "bert.embeddings" in n :
                    if "word_embeddings" in n :
                        torch.nn.init.normal_(p, mean=0, std=p.shape[-1]**(-0.25))
                        inited_by = "normal"
                else :
                    if "weight" in n :
                        if "LayerNorm" in n :
                            torch.nn.init.ones_(p)
                            inited_by = "ones"
                        else :
                            torch.nn.init.xavier_normal_(p)
                            inited_by = "xavier"
                    if "bias" in n :
                        torch.nn.init.zeros_(p)
                        inited_by = "zero"
                if inited_by :
                    print("reinit:", " / ".join([n, str(list(p.shape)), inited_by]))
            else :
                pass
        print()

    def confirm_parameter_groups(self) :
        for k,g in self.grouped_parameters.items() :
            print(k)
            if len (g) > 0 :
                count = {}
                for n in g.keys() :
                    if n.count('.') < 2 :
                        h = n
                    else :
                        h = '.'.join(n.split('.',2)[:2]) + "..."
                    if not h in count :
                        count[h] = 0
                    else :
                        count[h] += 1
                for h,c in count.items() :
                    if h.count('.') < 2 :
                        print(f"  {h}")
                    else :
                        print(f"  {h} {c}")
            else :
                print("  -none-")

    def get_parameters(self, groups, lrscale=1.0) :
        params_out = []
        params_left = set((n for n,p in self.named_parameters()))
        _set_else = False
        for group,opts in groups.items() :
            if group == "else" :
                _set_else = True
                continue
            params_out.append({"params": self.grouped_parameters[group].values(), **opts})
            params_left -= set(self.grouped_parameters[group].keys())
        if _set_else and params_left:
            params_out.append({"params": (p for n,p in self.named_parameters() if n in params_left), **groups["else"]})
        for v in params_out :
            v["lr"] *= lrscale
        return params_out

    def set_train(self, groups) :
        for group, flag in groups.items() :
            for param in self.grouped_parameters[group].values() :
                param.requires_grad = flag

    def set_data(self, batch) :
        _device = next(self.parameters()).device
        for key,data in batch.items() :
            if isinstance(data, torch.Tensor) :
                batch[key] = data.to(_device)
        if not "y" in batch :
            batch["y"] = None
        self.batch = batch

    def forward(self, batch=None) :
        if batch is not None :
            self.set_data(batch)
        else :
            batch = self.batch

        self.encoder(batch)
        out = self.decoder(batch)
        if isinstance(out, torch.Tensor) :
            return out.tolist()
        else :
            return out

    def backward(self) :
        return self.decoder.backward(self.batch)

    def get_accuracy(self) :
        return self.decoder.get_accuracy(self.batch)

class EncoderLSTM(torch.nn.Module) :
    def __init__(self, cfg) :
        super().__init__()
        self.use_pretrained_embedding = cfg.lstm_load_word_embedding
        if self.use_pretrained_embedding :
            with open(cfg.path_word_embedding_mask, 'r') as f :
                _input_embed_mask = eval(f.read())
            self.input_embedding_mask = torch.nn.Parameter(torch.tensor(_input_embed_mask, dtype=torch.long),requires_grad=False)
            self.input_embedding_loaded = torch.nn.Embedding(len(_input_embed_mask), cfg.input_embed_dim)
            self.input_embedding_loaded.load_state_dict(torch.load(cfg.path_word_embedding_weight))
            self.input_embedding_loaded.need_grad = False
            self.input_embedding_totrain = torch.nn.Embedding(len(_input_embed_mask), cfg.input_embed_dim)
        else :
            self.input_embedding = torch.nn.Embedding(cfg.input_word_size, cfg.input_embed_dim)
        cfg.encoder_embed_dim = 2*cfg.encoder_hidden_dim
        cfg.encoder_summary_dim = 8*cfg.encoder_hidden_dim
        self.lstm = torch.nn.LSTM( input_size=cfg.input_embed_dim,
                                   hidden_size=cfg.encoder_hidden_dim,
                                   num_layers=2,
                                   batch_first=True,
                                   # dropout=0,
                                   dropout=0.25,
                                   bidirectional=True )
    
    def forward(self, batch) :
        x = batch["x"]
        if self.use_pretrained_embedding :
            embed_loaded = self.input_embedding_loaded(x)
            embed_totrain = self.input_embedding_totrain(x)
            mask = self.input_embedding_mask[x].unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,self.encoder_embed_dim)
            x_embed = torch.gather(torch.stack((embed_totrain, embed_loaded), 0), 0, mask).squeeze(0)
        else :
            x_embed = self.input_embedding(x)
        xl = torch.nonzero(x==2, as_tuple=True)[-1]+1
        xpack = torch.nn.utils.rnn.pack_padded_sequence(x_embed, xl.tolist(), batch_first=True, enforce_sorted=False)
        lstm_output, lstm_hidden = self.lstm(xpack)
        encoder_embed, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        # [2, 4, N, Dh] -> [4, N, 2Dh] -> [N, 8Dh]
        encoder_summary = torch.cat(lstm_hidden, -1).permute(1,0,2).reshape(x.shape[0], -1)
        batch["encoder_embed"] = encoder_embed
        batch["encoder_summary"] = encoder_summary
        return encoder_embed, encoder_summary

class EncoderBert(torch.nn.Module) :
    def __init__(self, cfg) :
        super().__init__()
        loadpath = cfg.path_bert_load
        model_load = {"bert": "bert-base-chinese",
                      "roberta": "hfl/chinese-roberta-wwm-ext",}.get(cfg.encode_method, None)
        bertcfg = BertConfig.from_pretrained(model_load, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(model_load, config=bertcfg)
        self.nlayer = cfg.args.get("--bl", 12)
        if not cfg.encoder_ext_lstm :
            cfg.encoder_embed_dim = 768
            cfg.encoder_summary_dim = 768
            self.lstm = None
        else :
            cfg.encoder_embed_dim = 2*cfg.encoder_hidden_dim
            cfg.encoder_summary_dim = 768 + 8*cfg.encoder_hidden_dim
            self.lstm = torch.nn.LSTM( input_size=768,
                                       hidden_size=cfg.encoder_hidden_dim,
                                       num_layers=2,
                                       batch_first=True,
                                       dropout=0.25,
                                       bidirectional=True )
            
    def forward(self, batch) :
        x = batch["x"]
        att_mask = x>0
        last_layer, pooler_output, all_layers = self.bert(x, attention_mask=att_mask)[:]
        bert_embed = all_layers[self.nlayer]
        bert_summary = bert_embed[:,0,:]

        if self.lstm is None :
            encoder_embed = bert_embed
            encoder_summary = bert_summary
        else :
            xl = torch.nonzero(x==102, as_tuple=True)[-1]+1
            xpack = torch.nn.utils.rnn.pack_padded_sequence(bert_embed, xl.tolist(), batch_first=True, enforce_sorted=False)
            lstm_output, lstm_hidden = self.lstm(xpack)
            # [N, L, 2Dh]
            encoder_embed, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
            # [2, 4, N, Dh] -> [4, N, 2Dh] -> [N, 8Dh]
            lstm_hidden = torch.cat(lstm_hidden, -1).permute(1,0,2).reshape(x.shape[0], -1)
            encoder_summary = torch.cat((bert_summary, lstm_hidden), -1)
        batch["encoder_embed"] = encoder_embed
        batch["encoder_summary"] = encoder_summary
        return encoder_embed, encoder_summary

class EncoderGraph(torch.nn.Module) :
    def __init__(self, cfg):
        super().__init__()
        self.graph = g2t_model.EncoderSeq( input_size=cfg.input_word_size,
                                           embedding_size=128,
                                           hidden_size=512,
                                           n_layers=2,
                                           dropout=0.5 )
        if not cfg.encoder_ext_lstm :
            cfg.encoder_embed_dim = 512
            cfg.encoder_summary_dim = 512
            self.lstm = None
        else :
            cfg.encoder_embed_dim = 2*cfg.encoder_hidden_dim
            cfg.encoder_summary_dim = 512 + 8*cfg.encoder_hidden_dim
            self.lstm = torch.nn.LSTM( input_size=512,
                                        hidden_size=cfg.encoder_hidden_dim,
                                        num_layers=2,
                                        batch_first=True,
                                        # dropout=0,
                                        dropout=0.25,
                                        bidirectional=True )


    def forward(self, batch) :
        # [N, L] -> [L, N]
        x = batch["x"].t()
        xl = batch["xl"]
        graph = batch["graph"]
        # [N, L, Dh], [N, Dh]
        graph_embed, graph_summary = self.graph(x, xl, graph)

        if self.lstm is None :
            encoder_embed = graph_embed
            encoder_summary = graph_summary
        else :
            xpack = torch.nn.utils.rnn.pack_padded_sequence(graph_embed, xl, batch_first=True, enforce_sorted=False)
            lstm_output, lstm_hidden = self.lstm(xpack)
            '''sequence length is changed with pack'''
            # [N, L, 2Dh]
            encoder_embed, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
            # [2, 4, N, Dh] -> [4, N, 2Dh] -> [N, 8Dh]
            lstm_hidden = torch.cat(lstm_hidden, -1).permute(1,0,2).reshape(x.shape[1], -1)
            encoder_summary = torch.cat((graph_summary, lstm_hidden), -1)
        batch["encoder_embed"] = encoder_embed
        batch["encoder_summary_raw"] = graph_summary
        batch["encoder_summary"] = encoder_summary
        return encoder_embed, encoder_summary

class DecoderLSTM(torch.nn.Module) :
    def __init__(self, cfg):
        super().__init__()
        BasePgm.init(cfg)
        self.decoder_word_size = cfg.decoder_word_size
        self.decoder_max_len = cfg.program_max_len
        self.decoder_hidden_dim = cfg.decoder_hidden_dim
        self.decoder_embeding = torch.nn.Embedding(cfg.decoder_word_size, cfg.decoder_embed_dim)
        self.decoder_init = torch.nn.Sequential(torch.nn.Linear(cfg.encoder_summary_dim, 4*cfg.decoder_hidden_dim),
                                                torch.nn.Tanh())
        self.decoder_rnn = torch.nn.LSTM(input_size = cfg.decoder_embed_dim,
                                         hidden_size = cfg.decoder_hidden_dim,
                                         num_layers = 2,
                                         batch_first = True,
                                         # dropout = 0,
                                         dropout = 0.5)        
        self.decoder_linear = torch.nn.Linear(self.decoder_hidden_dim, cfg.decoder_word_size)
        self.decoder_with_attention = True

        # self.attention_actf = torch.nn.ReLU()
        self.attention_actf = torch.nn.Tanh()
        self.attention_query = torch.nn.Sequential(torch.nn.Linear(2*cfg.decoder_hidden_dim, cfg.encoder_embed_dim),
                                                        torch.nn.Tanh())
        self.attention_comb_out = torch.nn.Sequential(torch.nn.Linear(cfg.encoder_embed_dim+cfg.decoder_hidden_dim, cfg.decoder_hidden_dim),
                                                        torch.nn.Tanh())
        self.attention_comb_in = torch.nn.Sequential(torch.nn.Linear(cfg.encoder_embed_dim+cfg.decoder_embed_dim, cfg.decoder_embed_dim),
                                                        torch.nn.Tanh())
        self.encoder_embed_dim = cfg.encoder_embed_dim

    def forward(self, batch) :
        x = batch["x"]
        y = batch["y"]
        encoder_embed = batch["encoder_embed"]
        encoder_summary = batch["encoder_summary"]
        bs = x.shape[0]

        # [N, 8Dh] -> [N, 4Dh] -> [N, 2, 2, Dh] -> [2, 2, N, Dh]
        decoder_hidden = self.decoder_init(encoder_summary).view(-1, 2, 2, self.decoder_hidden_dim).permute(1,2,0,3).contiguous()
        decoder_token = torch.ones(bs, 1, dtype=torch.long, device=x.device)
        cross_attention = torch.zeros(bs, 1, self.encoder_embed_dim, dtype=x.dtype, device=x.device)

        decoder_linears = []
        decoder_tokens = []
        finished = [False]*bs

        for t in range(self.decoder_max_len) :
            if y is not None :
                decoder_token = y[:,t:t+1]
            finished = [f or t==2 for f,t in zip(finished, decoder_token.squeeze(-1).tolist())]
            if all(finished) :
                break

            decoder_embed = self.decoder_embeding(decoder_token)
            decoder_embed = self.attention_comb_in(torch.cat((decoder_embed, cross_attention), -1))
            decoder_output, decoder_hidden = self.decoder_rnn(decoder_embed, (*decoder_hidden,))            
            
            # [2, 2, N, 512] -> [2, N, 1024] -> [N, 1, 1024]
            hidden_last = torch.cat(decoder_hidden, -1)[-1].unsqueeze(1)
            attention_query = self.attention_query(hidden_last)
            attention_weight = torch.softmax(torch.bmm(attention_query, encoder_embed.permute(0,2,1)), -1)
            cross_attention = torch.bmm(attention_weight, encoder_embed)

            linear_in = self.attention_comb_out(torch.cat((decoder_output, cross_attention), -1))
            linear_out = self.decoder_linear(linear_in)
            decoder_token = linear_out[:,:,:self.current_decode_range].argmax(-1)

            decoder_linears.append(linear_out)
            decoder_tokens.append(decoder_token)
        
        decoder_linears = torch.cat(decoder_linears, 1)
        batch["decoder_linears"] = decoder_linears
        decoder_tokens = torch.cat(decoder_tokens, 1)
        batch["decoder_tokens"] = decoder_tokens
        return decoder_tokens

    def backward(self, batch, normalize_weight=False) :
        y = batch["y"]
        w = batch["w"]
        if w is not None and normalize_weight :
            w = w * self.batch_size / w.sum()

        decoder_linears = batch["decoder_linears"]
        CELoss = torch.nn.CrossEntropyLoss(reduction="none")
        
        L = min(y.shape[1], decoder_linears.shape[1]) -1
        y = y[:,1:1+L].contiguous().flatten()
        y_valid = (y > 0)
        decoder_linears = decoder_linears[:,:L,:].contiguous().view(-1, self.decoder_word_size)
        loss_raw = CELoss(decoder_linears, y)
        if w is not None :
            loss_raw = loss_raw * w.unsqueeze(1).repeat(1,L).flatten()
        loss = loss_raw[y_valid]

        loss = torch.sum(loss) / torch.sum(y_valid)
        loss.backward()
        return {"loss": loss.item()}

    def get_accuracy(self, batch) :
        y = batch["y"]
        z = batch["decoder_tokens"]

        y = y[:,1:]
        L = min(y.shape[1], z.shape[1])
        y_valid = (y[:,:L] > 0)
        accuT = torch.sum((y[:,:L] == z[:,:L]) & y_valid).float() / torch.sum(y_valid)
        accuP = torch.sum(~(((y[:,:L] != z[:,:L]) & y_valid).any(-1))).float() / y.shape[0]

        return {k:v.item() for k,v in 
                {"token": accuT,
                 "program": accuP,
                 }.items()}

class DecoderTree(torch.nn.Module) :
    def __init__(self, cfg) :
        super().__init__()
        '''word2index'''
        ''' 0-4 *-+/^
            5-6 1 3.14
            7-  N_
            22? UNK '''
        G2TPgm.init()
        self.beamsize = 5
        self.decoder_max_len = cfg.program_max_len
        if cfg.encoder_summary_dim == 512 :
            self.summary_init = None
        else :
            self.summary_init = torch.nn.Sequential(torch.nn.Linear(cfg.encoder_summary_dim, 512),
                                                    torch.nn.Tanh())
        if cfg.encoder_embed_dim == 512 :
            self.embed_init = None
        else :
            self.embed_init = torch.nn.Sequential(torch.nn.Linear(cfg.encoder_embed_dim, 512),
                                                    torch.nn.Tanh())
        self.predict = g2t_model.Prediction(hidden_size=512,
                                            # encoder_embed_size=cfg.encoder_embed_dim,
                                            op_nums=G2TPgm.num_start,
                                            input_size=len(G2TPgm.Consts))
        self.generate = g2t_model.GenerateNode(hidden_size=512,
                                               op_nums=G2TPgm.num_start,
                                               embedding_size=128)
        self.merge = g2t_model.Merge(hidden_size=512,
                                     embedding_size=128)

    def generate_tree_input(self, target, decoder_output, nums_stack_batch, num_start, unk):
        target_input = copy.deepcopy(target)
        for i in range(len(target)):
            if target[i] == unk:
                num_stack = nums_stack_batch[i].pop()
                # max_score = float("-inf")
                max_score = -1e12
                for num in num_stack:
                    if decoder_output[i, num_start + num] > max_score:
                        target[i] = num + num_start
                        max_score = decoder_output[i, num_start + num]
            if target_input[i] >= num_start:
                target_input[i] = 0
        return (torch.tensor(t, dtype=torch.long, device=decoder_output.device) for t in (target, target_input))

    def forward(self, batch) :
        x = batch["x"]
        y = batch["y"]
        yl = batch["yl"]
        encoder_embed = batch["encoder_embed"]
        encoder_summary = batch["encoder_summary"]
        nums_stack_batch = batch["num_stacks"]
        bs = x.shape[0]
        num_start, generate_nums, unk = G2TPgm.num_start, G2TPgm.Consts, G2TPgm.unk

        if y is not None :
            decode_length = max(yl)
            batch["y"] = batch["y"][:,:decode_length]
            target = batch["y"].t()
        else :
            decode_length = self.decoder_max_len
        seq_mask = (x==0)[:,:encoder_embed.shape[1]]
        num_mask = torch.nn.functional.pad(~batch["xnm"], (len(generate_nums), 0, 0, 0), "constant", 0).to(torch.bool)
        padding_hidden = torch.zeros(self.predict.hidden_size, dtype=torch.float, device=x.device).unsqueeze(0)

        if self.embed_init :
            encoder_embed = self.embed_init(encoder_embed)
        number_idx = (batch["xnp"] + (torch.arange(bs, device=x.device)*encoder_embed.shape[1]).unsqueeze(-1)).flatten()
        encoder_embed_number = encoder_embed.reshape(-1, encoder_embed.shape[-1])[number_idx]
        encoder_embed_number[~batch["xnm"].flatten()] = 0.0
        all_nums_encoder_outputs = encoder_embed_number.view(*(batch["xnp"].shape), -1)

        if self.summary_init :
            encoder_summary = self.summary_init(encoder_summary)
        node_stacks = [[g2t_model.TreeNode(_)] for _ in encoder_summary.split(1, dim=0)]
        embeddings_stacks = [[] for _ in range(bs)]
        left_childs = [None for _ in range(bs)]
        encoder_outputs = encoder_embed.transpose(0, 1)
        if y is not None :
            all_node_outputs = []
            for t in range(decode_length) :
                num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(
                node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                outputs = torch.cat((op, num_score), 1)
                all_node_outputs.append(outputs)
                target_t, generate_input = self.generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
                target[t] = target_t
                left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)
                left_childs = []
                for idx, l, r, node_stack, i, o in zip(range(bs), left_child.split(1), right_child.split(1), node_stacks, target[t].tolist(), embeddings_stacks):
                    if len(node_stack) != 0:
                        node = node_stack.pop()
                    else:
                        left_childs.append(None)
                        continue
                    if i < num_start:
                        node_stack.append(g2t_model.TreeNode(r))
                        node_stack.append(g2t_model.TreeNode(l, left_flag=True))
                        o.append(g2t_model.TreeEmbedding(node_label[idx].unsqueeze(0), False))
                    else:
                        current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                        while len(o) > 0 and o[-1].terminal:
                            sub_stree = o.pop()
                            op = o.pop()
                            current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                        o.append(g2t_model.TreeEmbedding(current_num, True))
                    if len(o) > 0 and o[-1].terminal:
                        left_childs.append(o[-1].embedding)
                    else:
                        left_childs.append(None)
            decoder_linears = torch.stack(all_node_outputs, dim=1)
            batch["decoder_linears"] = decoder_linears
            # decoder_tokens = target.transpose(0, 1).contiguous()
            decoder_tokens = decoder_linears.argmax(-1)            
            batch["decoder_tokens"] = decoder_tokens
            return decoder_tokens
        else :
            beams = [g2t_model.TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]
            for t in range(decode_length) :
                current_beams = []
                while len(beams) > 0:
                    b = beams.pop()
                    if len(b.node_stack[0]) == 0:
                        current_beams.append(b)
                        continue
                    left_childs = b.left_childs

                    num_score, op, current_embeddings, current_context, current_nums_embeddings = self.predict(
                        b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)
                    out_score = torch.nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
                    topv, topi = out_score.topk(self.beamsize)
                    for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                        current_node_stack = copy.deepcopy(b.node_stack)
                        current_left_childs = []
                        current_embeddings_stacks = copy.deepcopy(b.embedding_stack)
                        current_out = copy.deepcopy(b.out)

                        out_token = int(ti)
                        current_out.append(out_token)

                        node = current_node_stack[0].pop()

                        if out_token < num_start:
                            generate_input = torch.tensor([out_token], dtype=torch.long, device=x.device)
                            left_child, right_child, node_label = self.generate(current_embeddings, generate_input, current_context)
                            current_node_stack[0].append(g2t_model.TreeNode(right_child))
                            current_node_stack[0].append(g2t_model.TreeNode(left_child, left_flag=True))
                            current_embeddings_stacks[0].append(g2t_model.TreeEmbedding(node_label[0].unsqueeze(0), False))
                        else:
                            current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)
                            while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                                sub_stree = current_embeddings_stacks[0].pop()
                                op = current_embeddings_stacks[0].pop()
                                current_num = self.merge(op.embedding, sub_stree.embedding, current_num)
                            current_embeddings_stacks[0].append(g2t_model.TreeEmbedding(current_num, True))
                        if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                            current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                        else:
                            current_left_childs.append(None)
                        current_beams.append(g2t_model.TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                                                current_left_childs, current_out))
                beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
                beams = beams[:self.beamsize]
                if all((len(b.node_stack[0])==0 for b in beams)) :
                    break
            decoder_tokens = [beams[0].out]
            batch["decoder_tokens"] = decoder_tokens
            return decoder_tokens

    def backward(self, batch) :
        y = batch["y"]
        yl = batch["yl"]
        decoder_linears = batch["decoder_linears"]

        # loss = g2t_model.masked_cross_entropy(decoder_linears, y, torch.tensor(yl, dtype=torch.long, device=y.device))
        CELoss = torch.nn.CrossEntropyLoss(reduction="none")
        ylm = max(yl)
        y_valid = list(itertools.chain(*(range(i*ylm, i*ylm+yli) for i,yli in enumerate(yl))))
        loss = CELoss(decoder_linears.view(-1, decoder_linears.shape[-1]), y.flatten())[y_valid]
        loss = loss.sum() / sum(yl)
        loss.backward()
        return {"loss": loss.item()}

    def get_accuracy(self, batch) :
        y = batch["y"]
        z = batch["decoder_tokens"]

        yl = batch["yl"]
        ypad = y.shape[1]
        y_valid = torch.tensor([[1]*l + [0]*(ypad-l) for l in yl], dtype=torch.bool, device=y.device)
        accuT = torch.sum((y == z) & y_valid).float() / torch.sum(y_valid)
        accuP = torch.sum(~(((y != z) & y_valid).any(-1))).float() / y.shape[0]

        return {k:v.item() for k,v in 
                {"token": accuT,
                 "program": accuP,
                 }.items()}

class DecoderMILE(torch.nn.Module) :
    def __init__(self, cfg):
        super().__init__()
        ''' ==== memory operation ==== '''
        self.memory_width = cfg.memory_width
        self.memory_embed_dim = cfg.memory_embed_dim
        self.memory_max_width = 1+cfg.memory_width["const"]+cfg.memory_width["number"]+cfg.memory_width["free"]
        self.memory_valid_mask_raw = torch.zeros(self.memory_max_width, dtype=torch.bool)
        self.memory_valid_mask_raw[1:1+len(MILEPgm.Consts)] = 1
        self.memory_embed = None        

        self.empty_embed = torch.nn.Parameter(torch.randn(1, self.memory_embed_dim), requires_grad=True)
        self.const_embed = torch.nn.Parameter(torch.randn(cfg.memory_width["const"], self.memory_embed_dim), requires_grad=True)
        self.number_embedding = torch.nn.Sequential(torch.nn.Linear(cfg.encoder_embed_dim, self.memory_embed_dim),
                                                    torch.nn.ELU(),
                                                    torch.nn.Dropout(0.5),
                                                    torch.nn.Linear(cfg.memory_embed_dim, cfg.memory_embed_dim),
                                                    torch.nn.Tanh())

        ''' ==== decoder ==== '''
        self.decoder_max_len = cfg.program_max_len // MILEPgm.SEG
        self.decoder_op_size = cfg.decoder_word_size
        self.decoder_hidden_dim = cfg.decoder_hidden_dim
        # self.decoder_init = torch.nn.Sequential(torch.nn.Linear(cfg.encoder_summary_dim, 2*cfg.decoder_hidden_dim),
        #                                         torch.nn.Tanh())
        self.decoder_init = torch.nn.Sequential(torch.nn.Linear(cfg.encoder_summary_dim, 2*cfg.decoder_hidden_dim),
                                                # torch.nn.ELU(),
                                                torch.nn.Tanh(),
                                                # torch.nn.Dropout(0.5),
                                                torch.nn.Linear(2*cfg.decoder_hidden_dim, 2*cfg.decoder_hidden_dim),
                                                torch.nn.Tanh())

        '''              0   1   2   3   +   -   *   /   ^   =  '''
        self.mi_match = [0,0,0,0,0,0,0,0,0,0,1,2,3,3,4,5,6,7,8,0]
        self.pa_needs = [-1,-1,-1,-1,2,2,2,2,2,1]
        NH = max(self.mi_match) + 1
        self.memory_query_net = torch.nn.ModuleList([torch.nn.Sequential(
                                                        torch.nn.Linear(cfg.decoder_hidden_dim, cfg.memory_embed_dim),
                                                        torch.nn.Tanh()
                                                            ) for _ in range(NH)])
        self.memory_update_net = torch.nn.ModuleList([torch.nn.Sequential(
                                                        torch.nn.Linear(cfg.decoder_hidden_dim+cfg.memory_embed_dim, cfg.memory_embed_dim),
                                                        torch.nn.ELU(),
                                                        torch.nn.Dropout(0.5),
                                                        torch.nn.Linear(cfg.memory_embed_dim, cfg.memory_embed_dim),
                                                        torch.nn.Tanh()
                                                            ) for _ in range(NH)])
        self.memory_gate_net = torch.nn.ModuleList([torch.nn.Sequential(
                                                        torch.nn.Linear(cfg.decoder_hidden_dim+cfg.memory_embed_dim, cfg.memory_embed_dim),
                                                        torch.nn.Sigmoid()
                                                            ) for _ in range(NH)])
        self.memory_embed_net = torch.nn.ModuleList([torch.nn.Sequential(
                                                        torch.nn.Linear(cfg.decoder_hidden_dim+2*cfg.memory_embed_dim, cfg.memory_embed_dim),
                                                        torch.nn.ELU(),
                                                        torch.nn.Dropout(0.5),
                                                        torch.nn.Linear(cfg.memory_embed_dim, cfg.memory_embed_dim),
                                                        torch.nn.Tanh()
                                                            ) for _ in range(cfg.decoder_word_size)])
        self.decoder_op = torch.nn.Sequential(torch.nn.Linear(cfg.decoder_hidden_dim+cfg.memory_embed_dim+cfg.encoder_embed_dim, cfg.decoder_hidden_dim),
                                              torch.nn.ELU(),
                                              torch.nn.Dropout(0.5),
                                              torch.nn.Linear(cfg.decoder_hidden_dim, cfg.decoder_word_size))
        self.decoder_pa = torch.nn.ModuleList([torch.nn.RNN(input_size=cfg.memory_embed_dim,
                                                            hidden_size=cfg.decoder_hidden_dim,
                                                            num_layers=2,
                                                            batch_first=True,
                                                            dropout=0.5,
                                                            ) for _ in range(NH)])

        '''==== attention ===='''
        self.memory_sum_att_query_net = torch.nn.Sequential(torch.nn.Linear(cfg.decoder_hidden_dim, cfg.memory_embed_dim),
                                                            # torch.nn.Tanh(),
                                                            )
        self.encoder_att_query_net = torch.nn.Sequential(torch.nn.Linear(cfg.decoder_hidden_dim, cfg.encoder_embed_dim),
                                                            # torch.nn.Tanh(),
                                                            )
        self.update_seg = cfg.update_seg
        
    def forward(self, batch) :
        x = batch["x"]
        y = batch["y"]
        encoder_embed = batch["encoder_embed"]
        encoder_summary = batch["encoder_summary"]
        bs = x.shape[0]

        if y is not None :
            y = y.view(bs, -1, MILEPgm.SEG)
            y_op = y[:,:,0]
            y_pa = y[:,:,1:]
            decode_length = y.shape[1] - 1
        else :
            decode_length = self.decoder_max_len
            finished = [False]*bs
        z_op = []
        z_pa = []

        '''memory initialize'''
        memory_valid_mask = self.memory_valid_mask_raw.unsqueeze(0).repeat(bs, 1)
        memory_valid_mask[:,1+self.memory_width["const"]:1+self.memory_width["const"]+self.memory_width["number"]] = batch["xnm"]
        empty_embed = self.empty_embed.unsqueeze(0).expand(bs, -1, -1)
        const_embed = self.const_embed.unsqueeze(0).expand(bs, -1, -1)
        memory_static = 1+self.memory_width["const"]

        '''number embed'''
        number_idx = (batch["xnp"] + (torch.arange(bs, device=x.device)*encoder_embed.shape[1]).unsqueeze(-1)).flatten()
        encoder_embed_number = encoder_embed.reshape(-1, encoder_embed.shape[-1])[number_idx]
        '''check the mask'''
        encoder_embed_number = encoder_embed_number.view(*(batch["xnp"].shape), -1)
        number_embed = self.number_embedding(encoder_embed_number)

        memory_embed = torch.cat((empty_embed, const_embed, number_embed), 1)

        '''decoder initialize'''
        # [N, 8Dh] -> [N, 2Dh] -> [N, 2, Dh] -> [2, N, Dh]
        decoder_hidden = self.decoder_init(encoder_summary).view(-1, 2, self.decoder_hidden_dim).permute(1,0,2)
        decoder_output = decoder_hidden[-1,:,:]

        op_decode_linears = []
        pa_decode_linears = []

        for t in range(decode_length) :
            memory_used = memory_embed.shape[1]

            memory_embed_ref = memory_embed[:,memory_static:,:].clone()
            memory_sum_att_query = self.memory_sum_att_query_net(decoder_output).unsqueeze(-1)
            # [N, Wm-, Dm] * [N, Dm, 1] -> [N, Wm-] -> ... -> [N, 1, Wm-]
            memory_sum_weight = torch.bmm(memory_embed_ref, memory_sum_att_query).squeeze(-1)
            memory_sum_weight[~memory_valid_mask[:,memory_static:memory_used]] = -10e9
            memory_sum_weight = torch.softmax(memory_sum_weight, 1).unsqueeze(1)
            # [N, 1, Wm-] * [N, Wm-, Dm] -> [N, 1, Dm] -> [N, Dm]
            memory_summary = torch.bmm(memory_sum_weight, memory_embed_ref).squeeze(1)

            encoder_att_query = self.encoder_att_query_net(decoder_output).unsqueeze(-1)
            # [N, Le, 2De] * [N, 2De, 1] -> [N, Le, 1] -> [N, 1, Le]
            encoder_att_weight = torch.softmax(torch.bmm(encoder_embed, encoder_att_query), 1).view(bs, 1, -1)
            # [N, 1, Le] * [N, Le, 2De] -> [N, 1, 2De] -> [N, 2De]
            encoder_attention = torch.bmm(encoder_att_weight, encoder_embed).squeeze(1)

            # [N, *]
            op_decode_input = torch.cat((decoder_output, memory_summary, encoder_attention), -1)
            op_decode_linear = self.decoder_op(op_decode_input)
            op_decode_linears.append(op_decode_linear)
            ops = op_decode_linear.argmax(-1)
            z_op.append(ops)

            if y is not None :
                ops = y_op[:,t+1]
                _ops = ops.tolist()
                finished = [op == 0 for op in _ops]
            else :
                ops = ops.flatten()
                _ops = ops.tolist()
                finished = [f or op == 0 for f,op in zip(finished, _ops)]
            if all(finished) :
                z_pa.append(torch.zeros(bs, 2, dtype=torch.long, device=x.device))
                break

            op_group = [[] for _ in range(self.decoder_op_size)]
            for i,op in enumerate(_ops) :
                op_group[op].append(i)
            z_pa.append(torch.zeros(bs, 2, dtype=torch.long, device=x.device))
            pa_decode_linear = torch.zeros(bs, 2, self.memory_max_width, dtype=torch.float, device=x.device)
            memory_embed_app = torch.zeros(bs, self.memory_embed_dim, dtype=torch.float, device=x.device)
            decoder_hidden_new = torch.zeros(2, bs, self.decoder_hidden_dim, dtype=torch.float, device=x.device)
            for op,seg in enumerate(op_group) :
                if len(seg) == 0 :
                    continue
                if self.pa_needs[op] == -1 :
                    z_pa[-1][seg,:] = 0
                    pa_decode_linear[seg,:,:] = 0.0
                    memory_embed_app[seg,:] = 0.0
                    continue

                # [2, Nseg, Dh]
                decoder_hidden_seg = decoder_hidden[:,seg,:]
                decoder_output_seg = decoder_hidden_seg[-1,:,:]
                memory_orig_ref = []
                for pi in range(self.pa_needs[op]) :
                    mi = self.mi_match[2*op+pi]

                    if self.update_seg :
                        decoder_output_seg = decoder_hidden_seg[-1,:,:]

                    # [Nseg, Dm, 1]
                    memory_query = self.memory_query_net[mi](decoder_output_seg).unsqueeze(-1)
                    # [Nseg, Wm, Dm] * [Nseg, Dm, 1] -> [Nseq, Wm, 1]
                    memory_weight = torch.bmm(memory_embed[seg,:,:].clone(), memory_query).squeeze(-1)
                    memory_weight[~memory_valid_mask[seg,:memory_used]] = float('-inf')
                    pa_decode_linear[seg,pi,:memory_used] = memory_weight
                    pa_decode_linear[seg,pi,memory_used:] = float('-inf')
                    pas = memory_weight.argmax(-1)
                    z_pa[-1][seg,pi] = pas

                    if y is not None :
                        pas = y_pa[seg,t+1,pi]
                    else :
                        pass
                    _pas = pas.tolist()

                    # # [Nseg, Dm]
                    memory_orig = memory_embed[seg,_pas,:].clone()
                    memory_update_input = torch.cat((decoder_output_seg, memory_orig), -1)
                    memory_update = self.memory_update_net[mi](memory_update_input)
                    memory_gate = self.memory_gate_net[mi](memory_update_input)

                    memory_update = memory_update*memory_gate + memory_orig*(1-memory_gate)
                    memory_orig_ref.append(memory_orig)

                    for i,ipa in enumerate(_pas) :
                        if ipa >= memory_static :
                            memory_embed[seg[i],ipa,:] = memory_update[i]

                    # [Nseg, 1, Dm], [2, Nseg, Dh]
                    _, decoder_hidden_seg = self.decoder_pa[mi](memory_orig.unsqueeze(1), decoder_hidden_seg)

                memory_embed_net = self.memory_embed_net[op]
                if len(memory_orig_ref) < 2 :
                    memory_orig_ref += [torch.zeros(decoder_output_seg.shape[0], self.memory_embed_dim*(2-len(memory_orig_ref)), dtype=torch.float, device=x.device)]
                
                if self.update_seg :
                    decoder_output_seg = decoder_hidden_seg[-1,:,:]
                
                memory_embed_ref = torch.cat((decoder_output_seg, *memory_orig_ref), -1)
                memory_embed_new = memory_embed_net(memory_embed_ref)
                memory_embed_app[seg,:] = memory_embed_new
                decoder_hidden_new[:,seg,:] = decoder_hidden_seg
            pa_decode_linears.append(pa_decode_linear)
            memory_embed = torch.cat((memory_embed.clone(), memory_embed_app.unsqueeze(1)), 1)
            memory_valid_mask[:,memory_used] = torch.logical_and(ops>=4, ops<9)
            decoder_hidden = decoder_hidden_new
            decoder_output = decoder_hidden[-1,:,:]

        z_op = torch.stack(z_op, 1).unsqueeze(-1)
        z_pa = torch.stack(z_pa, 1)
        z = torch.cat((z_op, z_pa), -1).view(bs, -1)
        batch["z"] = z
        if y is not None :
            # [N, L, Nd]
            op_decode_linears = torch.stack(op_decode_linears, 1)
            batch["op_decode_linears"] = op_decode_linears
            # [N, L, 2, Wm]
            pa_decode_linears = torch.stack(pa_decode_linears, 1)
            batch["pa_decode_linears"] = pa_decode_linears
        return z

    def forward_beam(self, batch, beam_size) :
        x = batch["x"]
        encoder_embed = batch["encoder_embed"]
        encoder_summary = batch["encoder_summary"]
        bs = x.shape[0]

        decode_length = self.decoder_max_len
        op_max_cand = 4
        op_max_cand = min(op_max_cand, beam_size)
        pa_max_cand = 5
        pa_max_cand = min(pa_max_cand, beam_size)
        op_score_weight, *pa_score_weight = (1.0, 1.0, 1.0)
        length_norm = [None, "sum", "prod", "weak_exp"][1]
        match length_norm :
            case None :
                def update_score(scores) :
                    scores[:,0] = torch.prod(scores[:,1::4], 1)
            case "sum" :
                def update_score(scores) :
                    scores[:,0] = torch.sum(scores[:,1::4], 1) / (scores.shape[1]//4)
            case "prod" :
                def update_score(scores) :
                    scores[:,0] = torch.prod(scores[:,1::4], 1) ** (1 / (scores.shape[1]//4))
            case "weak_exp" :
                def update_score(scores) :
                    w = torch.exp(-torch.arange(scores.shape[1]//4-1, -1, -1, device=x.device) / 5)
                    scores[:,0] = torch.prod(scores[:,1::4]*w, 1)
            case _ :
                raise NotImplementedError
        out = []

        '''memory initialize'''
        memory_valid_mask = self.memory_valid_mask_raw.unsqueeze(0).repeat(bs, 1)
        memory_valid_mask[:,1+self.memory_width["const"]:1+self.memory_width["const"]+self.memory_width["number"]] = batch["xnm"]
        empty_embed = self.empty_embed.unsqueeze(0).expand(bs, -1, -1)
        const_embed = self.const_embed.unsqueeze(0).expand(bs, -1, -1)
        memory_static = 1+self.memory_width["const"]

        '''number embed'''
        number_idx = (batch["xnp"] + (torch.arange(bs, device=x.device)*encoder_embed.shape[1]).unsqueeze(-1)).flatten()
        encoder_embed_number = encoder_embed.reshape(-1, encoder_embed.shape[-1])[number_idx]
        '''check the mask'''
        encoder_embed_number = encoder_embed_number.view(*(batch["xnp"].shape), -1)
        number_embed = self.number_embedding(encoder_embed_number)

        memory_embed = torch.cat((empty_embed, const_embed, number_embed), 1)

        '''decoder initialize'''
        # [N, 8Dh] -> [N, 2Dh] -> [N, 2, Dh] -> [2, N, Dh]
        decoder_hidden = self.decoder_init(encoder_summary).view(-1, 2, self.decoder_hidden_dim)

        memory_valid_mask = memory_valid_mask.to(x.device)

        def iter_batch(*data) :
            for i in range(bs) :
                yield tuple(d[i:i+1] for d in data)
        for data in iter_batch(encoder_embed, memory_embed, memory_valid_mask, decoder_hidden) :
            encoder_embed, memory_embed, memory_valid_mask, decoder_hidden = data
            decoder_hidden = decoder_hidden.permute(1,0,2)
            z = torch.zeros(1, 0, dtype=torch.long, device=x.device)
            scores = torch.ones(1, 1, dtype=torch.float, device=x.device)
            finished = []

            for t in range(decode_length) :
                beam = []
                bs = z.shape[0]
                scores = torch.cat((scores, 
                                    torch.ones(scores.shape[0], 4, dtype=torch.float, device=x.device)
                                    ), 1)
                memory_used = memory_embed.shape[1]
                decoder_output = decoder_hidden[-1,:,:]

                memory_embed_ref = memory_embed[:,memory_static:,:].clone()
                memory_sum_att_query = self.memory_sum_att_query_net(decoder_output).unsqueeze(-1)
                # [N, Wm-, Dm] * [N, Dm, 1] -> [N, Wm-] -> ... -> [N, 1, Wm-]
                memory_sum_weight = torch.bmm(memory_embed_ref, memory_sum_att_query).squeeze(-1)
                # memory_sum_weight[~memory_valid_mask[:,memory_static:memory_used]] = float('-inf')  
                memory_sum_weight[~memory_valid_mask[:,memory_static:memory_used]] = -10e9
                memory_sum_weight = torch.softmax(memory_sum_weight, 1).unsqueeze(1)
                # [N, 1, Wm-] * [N, Wm-, Dm] -> [N, 1, Dm] -> [N, Dm]
                memory_summary = torch.bmm(memory_sum_weight, memory_embed_ref).squeeze(1)

                encoder_att_query = self.encoder_att_query_net(decoder_output).unsqueeze(-1)
                # [N, Le, 2De] * [N, 2De, 1] -> [N, Le, 1] -> [N, 1, Le]
                encoder_att_weight = torch.softmax(torch.bmm(encoder_embed, encoder_att_query), 1).view(bs, 1, -1)
                # [N, 1, Le] * [N, Le, 2De] -> [N, 1, 2De] -> [N, 2De]
                encoder_attention = torch.bmm(encoder_att_weight, encoder_embed).squeeze(1)

                # [N, *]
                op_decode_input = torch.cat((decoder_output, memory_summary, encoder_attention), -1)
                op_decode_linear = self.decoder_op(op_decode_input)
                op_decode_score = torch.nn.functional.softmax(op_decode_linear, dim=-1)
                op_score, ops = op_decode_score.topk(op_max_cand, -1)
                op_score, ops = op_score.flatten(), ops.flatten()
                bs_exp = ops.shape[0]
                z = torch.cat((z.repeat_interleave(op_max_cand, 0), 
                               ops.unsqueeze(-1),
                               torch.zeros(bs_exp, 2, dtype=torch.long, device=x.device)
                               ), -1)

                scores = scores.repeat_interleave(op_max_cand, 0)
                scores[:,-4] *= op_score**op_score_weight
                scores[:,-3] = op_score
                encoder_embed = encoder_embed.repeat_interleave(op_max_cand, 0)
                memory_embed = torch.cat((memory_embed.repeat_interleave(op_max_cand, 0),
                                          torch.zeros(bs_exp, 1, self.memory_embed_dim, dtype=torch.float, device=x.device)
                                          ), 1)                
                memory_valid_mask = memory_valid_mask.repeat_interleave(op_max_cand, 0)
                decoder_hidden = decoder_hidden.repeat_interleave(op_max_cand, 1)

                op_group = [[] for _ in range(self.decoder_op_size)]
                for i,op in enumerate(ops.tolist()) :
                    op_group[op].append(i)
                
                for op,seg in enumerate(op_group) :
                    if len(seg) == 0 :
                        continue

                    z_seg = z[seg,:]
                    score_seg = scores[seg]
                    encoder_embed_seg = encoder_embed[seg,:,:]
                    memory_embed_seg = memory_embed[seg,:,:]
                    memory_valid_mask_seg = memory_valid_mask[seg,:]
                    decoder_hidden_seg = decoder_hidden[:,seg,:]  

                    decoder_output_seg_ref_static = decoder_hidden_seg[-1,:,:]

                    if self.pa_needs[op] == -1 :
                        beam += [(score_seg, z_seg, encoder_embed_seg, 
                                  memory_embed_seg, memory_valid_mask_seg, decoder_hidden_seg)]
                        continue

                    memory_orig_ref = []
                    NPA = self.pa_needs[op]
                    for pi in range(NPA) :
                        mi = self.mi_match[2*op+pi]

                        if self.update_seg : 
                            decoder_output_seg = decoder_hidden_seg[-1,:,:]
                        else :
                            decoder_output_seg = decoder_output_seg_ref_static

                        # [Nseg, Dm, 1]
                        memory_query = self.memory_query_net[mi](decoder_output_seg).unsqueeze(-1)
                        # [Nseg, Wm, Dm] * [Nseg, Dm, 1] -> [Nseq, Wm, 1]
                        memory_weight = torch.bmm(memory_embed_seg[:,:-1,:], memory_query).squeeze(-1)
                        memory_weight[~memory_valid_mask_seg[:,:memory_used]] = float('-inf')
                        pa_decode_score = torch.nn.functional.softmax(memory_weight, dim=-1)
                        pa_score, pas = pa_decode_score.topk(pa_max_cand, -1)
                        pa_score, pas = pa_score.flatten(), pas.flatten()
                        bs_exp = pas.shape[0]
                        z_seg = z_seg.repeat_interleave(pa_max_cand, 0)
                        z_seg[:,-2+pi] = pas
                        
                        score_seg = score_seg.repeat_interleave(pa_max_cand, 0)
                        score_seg[:,-4] *= pa_score**pa_score_weight[pi]
                        score_seg[:,-2+pi] = pa_score
                        encoder_embed_seg = encoder_embed_seg.repeat_interleave(pa_max_cand, 0)
                        memory_embed_seg = memory_embed_seg.repeat_interleave(pa_max_cand, 0)
                        memory_valid_mask_seg = memory_valid_mask_seg.repeat_interleave(pa_max_cand, 0)
                        decoder_hidden_seg = decoder_hidden_seg.repeat_interleave(pa_max_cand, 1)

                        if self.update_seg : 
                            decoder_output_seg = decoder_hidden_seg[-1,:,:]
                        else :
                            decoder_output_seg_ref_static = decoder_output_seg_ref_static.repeat_interleave(pa_max_cand, 0)
                            decoder_output_seg = decoder_output_seg_ref_static
                        
                        memory_orig = memory_embed_seg[range(bs_exp),pas,:]
                        memory_update_input = torch.cat((decoder_output_seg, memory_orig), -1)
                        memory_update = self.memory_update_net[mi](memory_update_input)
                        memory_gate = self.memory_gate_net[mi](memory_update_input)
                        
                        memory_update = memory_update*memory_gate + memory_orig*(1-memory_gate)
                        memory_orig_ref.append(memory_orig.repeat_interleave(pa_max_cand**(NPA-1-pi), 0))
                        
                        for i,ipa in enumerate(pas.tolist()) :
                            if ipa >= memory_static :
                                memory_embed_seg[i,ipa,:] = memory_update[i]

                        # [Nseg, 1, Dm], [2, Nseg, Dh]
                        _, decoder_hidden_seg = self.decoder_pa[mi](memory_orig.unsqueeze(1), decoder_hidden_seg)
                
                    bs_exp = z_seg.shape[0]
                    memory_embed_net = self.memory_embed_net[op]
                    if len(memory_orig_ref) < 2 :
                        memory_orig_ref.append(torch.zeros(bs_exp, self.memory_embed_dim*(2-len(memory_orig_ref)), 
                                                           dtype=torch.float, device=x.device))

                    if self.update_seg : 
                        decoder_output_seg = decoder_hidden_seg[-1,:,:]
                    else :
                        decoder_output_seg = decoder_output_seg_ref_static
                        
                    memory_embed_ref = torch.cat((decoder_output_seg, *memory_orig_ref), -1)
                    memory_embed_new = memory_embed_net(memory_embed_ref)
                    memory_embed_seg[:,-1,:] = memory_embed_new
                    memory_valid_mask_seg[:,memory_used] = True

                    beam += [(score_seg, z_seg, encoder_embed_seg, 
                              memory_embed_seg, memory_valid_mask_seg, decoder_hidden_seg)]

                scores, z, encoder_embed, memory_embed, memory_valid_mask, decoder_hidden = zip(*beam)
                scores = torch.cat(scores, 0)
                z = torch.cat(z, 0)
                encoder_embed = torch.cat(encoder_embed, 0)
                memory_embed = torch.cat(memory_embed, 0)
                memory_valid_mask = torch.cat(memory_valid_mask, 0)
                decoder_hidden = torch.cat(decoder_hidden, 1)
                update_score(scores)

                sort_ref = ((s, True, z[i].tolist()) if z[i][3*t]==2 else (s, False, i) for i,s in enumerate(scores.tolist()))
                _sorted = sorted((*finished, *sort_ref), key=lambda x:-x[0][0])[:2*beam_size]

                if all((b[1] for b in _sorted[:beam_size])) :
                    out.append([(b[0], b[2]) for b in _sorted[:beam_size]])
                    break
                if length_norm is None :
                    idxs = [b[2] for b in _sorted[:beam_size] if not b[1]]
                else :
                    idxs = [b[2] for b in _sorted if not b[1]][:beam_size]
                finished = [b for b in _sorted if b[1]]
                
                z = z[idxs,:]
                scores = scores[idxs,:]
                encoder_embed = encoder_embed[idxs,:,:]
                memory_embed = memory_embed[idxs,:,:]
                memory_valid_mask = memory_valid_mask[idxs,:]
                decoder_hidden = decoder_hidden[:,idxs,:]
            else :
                out.append([(b[0], b[2]) for b in _sorted if b[1]][:beam_size])
        return out            
    
    def backward(self, batch, normalize_weight=False) :
        y = batch["y"]
        w = batch["w"]
        if w is not None and normalize_weight :
            w = w * self.batch_size / w.sum()

        # [N, L, 3]
        y = y.view(y.shape[0], -1, MILEPgm.SEG)[:,1:,:]
        y_op = y[:,:,0]
        y_pa = y[:,:,1:]
        op_decode_linears = batch["op_decode_linears"]
        pa_decode_linears = batch["pa_decode_linears"]
        CELoss = torch.nn.CrossEntropyLoss(reduction="none")

        L = min(y.shape[1], op_decode_linears.shape[1])
        y_op = y_op[:,:L].flatten()
        y_op_valid = (y_op > 0)
        loss_op_raw = CELoss(op_decode_linears.view(-1, op_decode_linears.shape[-1]), y_op)
        if w is not None :
            loss_op_raw = loss_op_raw * w.unsqueeze(1).repeat(1,L).flatten()
        loss_op = loss_op_raw[y_op_valid]

        L = min(y.shape[1], pa_decode_linears.shape[1])
        y_pa = y_pa[:,:L,:].flatten()
        y_pa_valid = (y_pa > 0)
        loss_pa_raw = CELoss(pa_decode_linears.view(-1, pa_decode_linears.shape[-1]), y_pa)
        if w is not None :
            loss_pa_raw = loss_pa_raw * w.unsqueeze(1).repeat(1,L*2).flatten()
        loss_pa = loss_pa_raw[y_pa_valid]

        loss_op = torch.sum(loss_op) / torch.sum(y_op_valid)
        loss_pa = torch.sum(loss_pa) / torch.sum(y_pa_valid)
        loss = loss_op + loss_pa
        loss.backward()

        return {k:v.item() for k,v in 
                {"loss": loss,
                 "loss_op": loss_op,
                 "loss_pa": loss_pa,
                 }.items()}

    def get_accuracy(self, batch) :
        y = batch["y"]
        z = batch["z"]

        y = y[:,MILEPgm.SEG:]
        L = min(y.shape[1], z.shape[1])
        y_valid = (y[:,:L] > 0)
        accuT = torch.sum((y[:,:L] == z[:,:L]) & y_valid).float() / torch.sum(y_valid)
        accuP = torch.sum(~(((y[:,:L] != z[:,:L]) & y_valid).any(-1))).float() / y.shape[0]

        y_mask_op = torch.zeros(MILEPgm.SEG, dtype=torch.bool, device=y.device)
        y_mask_op[0] = 1
        y_mask_op = y_mask_op.repeat(L//MILEPgm.SEG).unsqueeze(0).expand_as(y_valid)
        y_mask_pa = ~y_mask_op
        accuTop = torch.sum((y[:,:L] == z[:,:L]) & y_valid & y_mask_op).float() / torch.sum(y_valid & y_mask_op)
        accuTpa = torch.sum((y[:,:L] == z[:,:L]) & y_valid & y_mask_pa).float() / torch.sum(y_valid & y_mask_pa)

        return {k:v.item() for k,v in 
                {"token": accuT,
                 "program": accuP,
                 "token_op": accuTop,
                 "token_pa": accuTpa,
                 }.items()}
