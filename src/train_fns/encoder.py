import logging
import torch
import train_fns.lstm as lstm
import numpy as np
from torch import nn
from torch.autograd import Variable

logger = logging.getLogger(__name__)


class SeqLSTM(nn.Module):
    def __init__(self, cfg, input_size, hidden_size, output_size, gpu_id):
        self.cfg = cfg
        super(SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = lstm.LSTM(input_size, hidden_size).cuda(gpu_id)
        self.out = nn.Linear(hidden_size, output_size).cuda(gpu_id)
        self.gpu_id = gpu_id
        self.chr_id_embed = nn.Embedding(cfg.num_chr, cfg.chr_id_embed_size).cuda(gpu_id)
        self.pos_embed = nn.Embedding(cfg.chr_len, cfg.pos_embed_size).cuda(gpu_id)
        self.init_chr_weights = np.random.multivariate_normal(np.zeros(cfg.chr_id_embed_size),
                                                              np.identity(cfg.chr_id_embed_size), cfg.num_chr)
        self.init_pos_weights = np.random.multivariate_normal(np.zeros(cfg.pos_embed_size),
                                                              np.identity(cfg.pos_embed_size), cfg.chr_len)
        self._initialize_embeddings()

    def forward(self, hidden, state, input):
        output, (hidden, state) = self.lstm(input, (hidden, state))
        output = self.out(output)
        return output, hidden, state

    def _initialize_embeddings(self):
        self.chr_id_embed.weight.data.copy_(torch.from_numpy(self.init_chr_weights))
        self.pos_embed.weight.data.copy_(torch.from_numpy(self.init_pos_weights))

    def initHidden(self):
        h = Variable(torch.randn(1, 1, self.hidden_size).float()).cuda(self.gpu_id)
        c = Variable(torch.randn(1, 1, self.hidden_size).float()).cuda(self.gpu_id)

        return h.cuda(self.gpu_id), c.cuda(self.gpu_id)
