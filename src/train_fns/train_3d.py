import numpy as np
import torch
import os
import sys
import time
from torch.utils.tensorboard import SummaryWriter

from train_fns import config
from train_fns.model import HicLSTM3D
from train_fns.data_utils import get_data_loader, get_bedfile

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train_model(cfg, model_name, cell, writer):
    # initalize model
    model = HicLSTM3D(cfg, device, model_name).to(device)
    model.load_weights()

    optimizer, criterion = model.compile_optimizer(cfg)

    # get data
    start = time.time()
    data_loader, samples = get_data_loader(model, cfg, cell)
    print("%s batches loaded" % str(len(data_loader)))
    end = time.time()
    print("Time to obtain data: %s" % str(end - start))

    # train model
    model.train_model(data_loader, criterion, optimizer, writer)

    # save model
    torch.save(model.state_dict(), cfg.model_dir + model_name + '.pth')

    # get model embeddings with bed file of genomic coordinates
    bed_file = get_bedfile(np.unique(samples[:, :2], axis=0), cfg)
    embeddings = model.get_embeddings(np.unique(samples[:, 2], axis=0))

    bed_file.to_csv(cfg.output_directory + "embedding_coordinates.bed", sep='\t', header=False, index=False)
    np.save(cfg.output_directory + "embeddings.npy", embeddings)


if __name__ == '__main__':
    cfg = config.Config()
    cell = "GM12878"

    model_name = "contact_prob"

    # set up tensorboard logging
    timestr = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter('./tensorboard_logs/' + "hiclstm3d" + timestr)

    train_model(cfg, model_name, cell, writer)

    print("done")
