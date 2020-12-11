import os


class Config:
    def __init__(self):
        self.network = 'lstm'
        self.load_weights = False
        self.diag_lstm = False
        self.lstm_nontrain = False
        self.long_shot_sep = False
        self.distance_cut = True

        self.long_shot_thresh = 1000
        # self.chr_len = 288095
        self.chr_len = 288091
        self.distance_cut_off_mb = 500

        self.num_chr = 23
        self.resolution = 10000
        self.chr_id_embed_size = 2
        self.pos_embed_size = 16
        self.input_size_lstm = 2 * self.pos_embed_size
        self.hidden_size_lstm = 8
        self.output_size_lstm = 1

        self.learning_rate = 1e-2
        self.max_norm = 10

        self.hic_path = "/data2/hic_lstm/data/"
        # self.downstream_dir = "/Users/kevindsouza/Documents/UBC/PhD/Research/nucleosome/hic_lstm/data"
        self.downstream_dir = "/data2/hic_lstm/downstream"
        self.model_dir = '/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/exp_run'
        # self.hic_path = "/home/kevinbd/projects/rrg-maxwl/kevinbd/hic_lstm/hic_data/hic_data_all_chr/"
        # self.model_dir = "/home/kevinbd/projects/rrg-maxwl/kevinbd/hic_lstm/hic_lstm_code/src/saved_model/model_lstm"
        self.sizes_file = "chr_cum_sizes.npy"
        self.start_end_file = "starts.npy"
        self.sniper_annotation_path = "/data2/hic_lstm/data/sniper/"
        self.graph_annotation_path = "/data2/hic_lstm/data/graph/"
        self.pca_annotation_path = "/data2/hic_lstm/data/pca/"

        self.config_base = 'config.yaml'
        self.tensorboard_log_base = 't_log'
        self.config_file = os.path.join(self.model_dir, self.config_base)
        self.tensorboard_log_path = os.path.join(self.model_dir, self.tensorboard_log_base)

        if not os.path.exists(self.tensorboard_log_path):
            os.makedirs(self.tensorboard_log_path)

        self.num_epochs = 4
