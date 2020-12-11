from __future__ import division
from torch import optim
import torch
from torch import nn
from train_fns import encoder
from keras.optimizers import Adam
from keras.layers import Input
import logging

logger = logging.getLogger(__name__)


class Model:

    def __init__(self, cfg, gpu_id):
        self.cfg = cfg
        self.seq_lstm = encoder.SeqLSTM(cfg, cfg.input_size_lstm, cfg.hidden_size_lstm, cfg.output_size_lstm,
                                        gpu_id).cuda(gpu_id).train()

        if cfg.lstm_nontrain:
            for child in self.seq_lstm.children():
                for param in child.parameters():
                    param.requires_grad = False

            self.seq_lstm.chr_id_embed.requires_grad = True
            self.seq_lstm.pos_embed.requires_grad = True

    def load_weights(self):
        try:
            print('loading weights from {}'.format(self.cfg.model_dir))
            self.seq_lstm.load_state_dict(torch.load(self.cfg.model_dir + '/seq_lstm.pth'))

            # self.ca_embedding.load_state_dict(torch.load(self.cfg.model_dir + '/ca_embedding.pth'))
        except Exception as e:
            print("load weights exception: {}".format(e))

    def compile_optimizer(self, cfg):

        if cfg.lstm_nontrain:
            seq_lstm_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.seq_lstm.parameters()),
                                                  lr=self.cfg.learning_rate)
        else:
            seq_lstm_optimizer = torch.optim.Adam(self.seq_lstm.parameters(), lr=self.cfg.learning_rate)

        criterion = nn.MSELoss()
        return seq_lstm_optimizer, criterion

    def set_callback(self, callback):
        callback.set_model(self.seq_lstm)
        pass

    def save_weights(self):
        torch.save(self.seq_lstm.state_dict(), self.cfg.model_dir + '/seq_lstm.pth')
        # torch.save(self.pos_embed.state_dict(), self.cfg.model_dir + '/seq_lstm.pth')
        pass
