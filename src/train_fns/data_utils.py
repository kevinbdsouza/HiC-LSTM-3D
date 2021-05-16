import math
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import scipy.sparse
import matplotlib
import matplotlib.pyplot as plt


def contactProbabilities(values, delta=1e-10):
    coeff = np.nan_to_num(1 / (values + delta))
    CP = np.power(1 / np.exp(8), coeff)

    return CP


def bin_index(cfg, chr, pos_id):
    sizes = np.load(cfg.hic_path + cfg.sizes_file, allow_pickle=True).item()
    chr_key = ['chr' + str(chrom - 1) for chrom in chr]
    chr_begin = [int(sizes[key]/cfg.resolution) for key in chr_key]

    return pos_id + chr_begin


def get_hic(chr, cell, cfg):
    hic_data = pd.read_csv("%s%s/%s/hic_chr%s.txt" % (cfg.hic_path, cell, chr, chr), sep="\t", names=['i', 'j', 'v'])

    hic_data[['i', 'j']] = hic_data[['i', 'j']] / cfg.resolution
    hic_data[['i', 'j']] = hic_data[['i', 'j']].astype('int64')
    return hic_data


def get_gen_pos(cfg, chr, index_bin):
    sizes = np.load(cfg.hic_path + cfg.sizes_file, allow_pickle=True).item()
    chr_key = ['chr' + str(chrom - 1) for chrom in chr]
    chr_begin = [sizes[key] for key in chr_key]

    return (index_bin - chr_begin) * cfg.resolution


def load_hic_samples(chr, cfg, hic_data):
    hic_data = hic_data.apply(pd.to_numeric)
    hic_data['i_bin'] = bin_index(cfg, np.full(hic_data.shape[0], chr), hic_data['i'])
    hic_data['j_bin'] = bin_index(cfg, np.full(hic_data.shape[0], chr), hic_data['j'])
    hic_data['v'] = hic_data['v'].fillna(0)
    num_rows = max(hic_data['i'].max(), hic_data['j'].max()) + 1

    values_agg = []
    input_pos_agg = []
    list_nvals = []
    idx_agg = []
    for r in range(num_rows):
        # Hi-C values
        values = hic_data[hic_data['i'] == r]['v'].values
        nvalues = values.shape[0]
        if nvalues == 0:
            continue
        else:
            values = contactProbabilities(values)

        if (nvalues > 10):
            list_nvals.append(nvalues)
            values = torch.from_numpy(values)

            values_split = values.split(cfg.chunk_length, dim=0)
            values_agg = values_agg + list(values_split)

            # input indices
            row_id = torch.Tensor(hic_data[hic_data['i'] == r]['i_bin'].values)
            column_id = torch.Tensor(hic_data[hic_data['i'] == r]['j_bin'].values)
            input_ind = torch.cat((row_id.unsqueeze(-1), column_id.unsqueeze(-1)), 1)
            chunk_ind = torch.split(input_ind, cfg.chunk_length, dim=0)
            input_pos_agg = input_pos_agg + list(chunk_ind)

            idx_agg.append(input_ind)

    hic_values = pad_sequence(values_agg, batch_first=True)
    input_pos = pad_sequence(input_pos_agg, batch_first=True)
    unsplit_ids = np.vstack(idx_agg)
    unsplit_ids = np.concatenate((np.full((unsplit_ids.shape[0], 1), chr), unsplit_ids), 1).astype('int')

    return unsplit_ids, hic_values, input_pos


def load_hic_data(chr, cell, cfg):
    hic_data = get_hic(chr, cell, cfg)
    unsplit_ids, input_pos, hic_values = load_hic_samples(chr, cfg, hic_data)

    return unsplit_ids, input_pos, hic_values


def get_hic_loader_chr(chr, cell, cfg):
    unsplit_ids, hic_values, input_pos = load_hic_data(chr, cell, cfg)

    # build dataloader
    datacomb = TensorDataset(input_pos.float(), hic_values.float())
    hic_loader = DataLoader(dataset=datacomb, batch_size=cfg.batch_size, shuffle=False)

    return hic_loader, unsplit_ids


def get_hic_loader(cell, cfg):
    # build dataset
    input_pos_agg = torch.empty(0, cfg.chunk_length, 2)
    values_agg = torch.empty(0, cfg.chunk_length)
    unsplit_ids_agg = []

    # for chr in list(range(1, 11)) + list(range(12,23)):
    for chr in list(range(22, 23)):
        unsplit_ids, hic_values, input_pos = load_hic_data(chr, cell, cfg)

        input_pos_agg = torch.cat((input_pos_agg, input_pos), 0)
        values_agg = torch.cat((values_agg, hic_values.float()), 0)
        unsplit_ids_agg.append(unsplit_ids)

    unsplit_ids_agg = np.vstack(unsplit_ids_agg)

    # save data
    torch.save(input_pos_agg, cfg.processed_data_dir + 'input_pos.pth')
    torch.save(values_agg, cfg.processed_data_dir + 'hic_values.pth')
    torch.save(unsplit_ids_agg, cfg.processed_data_dir + 'unsplit_ids.pth')

    # build dataloader
    datacomb = TensorDataset(input_pos_agg, values_agg)
    hic_loader = DataLoader(dataset=datacomb, batch_size=cfg.batch_size, shuffle=False)

    return hic_loader, unsplit_ids_agg


def get_bed(cfg, unsplit_ids_agg):
    chrom = unsplit_ids_agg[:, 0]
    index_bin = unsplit_ids_agg[:, 1]
    start_pos = get_gen_pos(cfg, chrom, index_bin)
    stop_pos = start_pos + cfg.resolution

    bed_file = pd.DataFrame({'chr': chrom, 'start': start_pos, 'stop': stop_pos})
    return bed_file
