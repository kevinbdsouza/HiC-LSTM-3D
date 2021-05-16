import os
import sys
import time
import numpy as np
import torch
from train_fns import config
from train_fns.model import HicLSTM3D
from train_fns.data_utils import get_hic_loader, get_bed

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def model_train(cfg, model_name, cell, sum_writer):
    model = HicLSTM3D(cfg, device, model_name).to(device)
    model.load_weights()
    optimizer, criterion = model.compile_optimizer()

    # load hic data
    hic_loader, unsplit_ids = get_hic_loader(cell, cfg)

    # train HicLSTM3D
    model.train_Hic3D(hic_loader, optimizer, criterion, sum_writer)
    torch.save(model.state_dict(), cfg.model_dir + model_name + '.pth')

    # get representations with genomic positions
    bed_file = get_bed(np.unique(unsplit_ids[:, :2], axis=0), cfg)
    representations = model.get_representations(np.unique(unsplit_ids[:, 2], axis=0))

    bed_file.to_csv(cfg.output_directory + "rep_coord.bed", sep='\t', header=False, index=False)
    np.save(cfg.output_directory + "representations.npy", representations)


if __name__ == '__main__':
    cfg = config.Config()
    model_name = "contact_prob"
    cell = "GM12878"

    # set up summary writer
    sum_writer = None

    model_train(cfg, model_name, cell, sum_writer)

    print("done")
