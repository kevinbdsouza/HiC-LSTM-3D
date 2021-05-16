import os


class Config:
    def __init__(self):
        self.network = 'lstm'
        self.load_weights = False
        self.lstm_nontrain = False
        self.genome_len = 288091
        self.chunk_length = 500

        self.num_chr = 23
        self.resolution = 10000
        self.pos_embed_size = 3
        self.input_size_lstm = 2 * self.pos_embed_size
        self.hidden_size_lstm = 8
        self.output_size_lstm = 1

        self.learning_rate = 1e-2
        self.max_norm = 10
        self.num_epochs = 10
        self.batch_size = 200

        self.hic_path = "/data2/hic_lstm/data/"
        self.downstream_dir = "/data2/hic_lstm/downstream"
        self.model_dir = '/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/exp_run/'
        self.sizes_file = "chr_cum_sizes.npy"
        self.start_end_file = "starts.npy"
        self.output_directory = "../outputs/"
        self.processed_data_dir = self.output_directory + 'processed_data/'

        self.config_base = 'config.yaml'
        self.tensorboard_log_base = 't_log'
        self.config_file = os.path.join(self.model_dir, self.config_base)
        self.tensorboard_log_path = os.path.join(self.model_dir, self.tensorboard_log_base)

        if not os.path.exists(self.tensorboard_log_path):
            os.makedirs(self.tensorboard_log_path)

