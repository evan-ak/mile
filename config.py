
import os

class TotalConfig() :
    def __init__(self) :

        self.input_word_size = 12000
        self.input_embed_dim = 300
        self.encoder_hidden_dim = 256
        
        self.decoder_embed_dim = 300
        self.decoder_hidden_dim = 256

        self.memory_width = {"const":10, "number":10, "free":20}
        self.memory_embed_dim = 128

        self.local_dir = os.path.abspath(".")
        self.dir_data = self.local_dir + "/data"
        self.dir_saves = self.local_dir + "/saves"

        self.path_data_reg = self.dir_data + "/math23k_reg.json"
        self.encode_method = ["lstm", "graph", "bert", "roberta"][3]
        self.decode_method = ["lstm", "tree", "mile"][2]
        self.encoder_ext_lstm = False
        self.test_metric = ("5flod", "public")[0]
        self.test_split_seed = "standard seed 0"
        '''extra data'''
        self.lstm_load_word_embedding = False
        '''formula augmentation'''
        self.use_aug = True
        '''bert model'''
        self.path_bert_load = self.dir_data + "/bert-base-chinese"
        self.path_bert_token = self.dir_data + "/math23k_bert_token.pickle"
        '''graph2tree data'''
        self.path_g2t_raw = self.dir_data + "/g2t_raw.pickle"
        self.path_g2t_graph = self.dir_data + "/g2t_graph_nonzero.npy"
        self.memory_width = {"const":10, "number":15, "free":20}

        if self.decode_method == "lstm" :
            self.decoder_word_size = 100
            self.program_max_len = 50
            self.use_aug = False
        if self.decode_method == "tree" :
            self.decoder_word_size = 20
            self.program_max_len = 50
            self.use_aug = False
        if self.decode_method == "mile" :
            self.decoder_word_size = 10
            self.program_max_len = 60
            self.update_seg = False
        self.weight_aug = True
        self.normalize_weight = False
        self.mile_train_batch = 64000
        self.normalized_batch_size = 256

        self.dir_mile_savepoint = self.local_dir + "/saves"
        self.mile_load_savepoint = None
        # self.mile_load_savepoint = [("savepoint_label_*", {"all",}),]

        self.debug_flag = True

    def get(self, key, default=None) :
        return getattr(self, key, default)
    
    def overwrite(self, args) :
        for k,v in args.items() :
            if k[0] == "@" :
                setattr(self, k[1:], v)

class CustomData() :
    def __init__(self, dic = {}):
        for k, v in dic.items() :
            setattr(self, k, v)
