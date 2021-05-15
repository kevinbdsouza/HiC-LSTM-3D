import numpy as np
import pandas as pd
import scipy.sparse
import math
import os
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt


def get_bin_idx(chr, pos, cfg):
    sizes = np.load(cfg.hic_path + cfg.sizes_file, allow_pickle=True).item()
    chr = ['chr' + str(x - 1) for x in chr]
    chr_start = [sizes[key] for key in chr]

    return pos + chr_start


def get_genomic_coord(chr, bin_idx, cfg):
    sizes = np.load(cfg.hic_path + cfg.sizes_file, allow_pickle=True).item()
    chr = ['chr' + str(x - 1) for x in chr]
    chr_start = [sizes[key] for key in chr]

    return (bin_idx - chr_start) * cfg.resolution


def load_hic(cfg, cell, chr):
    data = pd.read_csv("%s%s/%s/hic_chr%s.txt" % (cfg.hic_path, cell, chr, chr), sep="\t", names=['i', 'j', 'v'])

    data[['i', 'j']] = data[['i', 'j']] / cfg.resolution
    data[['i', 'j']] = data[['i', 'j']].astype('int64')
    return data


def get_samples_dense(data, seq_lstm, chr, cfg):
    dist = cfg.distance_cut_off_mb
    mat = scipy.sparse.coo_matrix((data.v, (data.i, data.j))).tocsr()

    bin1 = get_bin_idx(chr, 0, cfg)
    nrows = mat.shape[0]

    values = torch.zeros(nrows, 2 * dist)
    input_idx = torch.zeros(nrows, 2 * dist, 2)

    for row in range(nrows):
        # get distance around diagonal
        start = max(row - dist, 0)
        stop = min(row + dist, mat.shape[0])

        # get Hi-C values
        vals = mat[row, start:stop].todense()
        vals = torch.squeeze(torch.from_numpy(vals))

        # get indices for inserting data
        idx1 = max(0, dist - row)
        idx2 = idx1 + vals.shape[0]

        # insert values
        vals_tmp = torch.zeros(1, 2 * dist)
        values[row, idx1:idx2] = vals

        # get indices
        j = torch.tensor(np.arange(start, stop))
        i = torch.full(j.shape, fill_value=row)
        ind = torch.cat((i, j), 1)
        input_idx[row, idx1:idx2, ] = ind

    # only add datapoint if one of the values is non-zero:
    nonzero_idx = torch.sum(values, dim=1).nonzero().squeeze()
    values = values[nonzero_idx,]
    input_idx = input_idx[nonzero_idx,]

    return input_idx, values


def get_samples_sparse(data, seq_lstm, chr, cfg):
    data = data.apply(pd.to_numeric)
    nrows = max(data['i'].max(), data['j'].max()) + 1
    data['v'] = data['v'].fillna(0)
    data['i_binidx'] = get_bin_idx(np.full(data.shape[0], chr), data['i'], cfg)
    data['j_binidx'] = get_bin_idx(np.full(data.shape[0], chr), data['j'], cfg)

    values = []
    input_idx = []
    nvals_list = []
    sample_index = []
    for row in range(nrows):
        # get Hi-C values
        vals = data[data['i'] == row]['v'].values
        nvals = vals.shape[0]
        if nvals == 0:
            continue
        else:
            vals = contactProbabilities(vals)

        if (nvals > 10):
            nvals_list.append(nvals)
            vals = torch.from_numpy(vals)

            split_vals = vals.split(cfg.sequence_length, dim=0)
            values = values + list(split_vals)

            # get indices
            j = torch.Tensor(data[data['i'] == row]['j_binidx'].values)
            i = torch.Tensor(data[data['i'] == row]['i_binidx'].values)

            # concatenate indices
            ind = torch.cat((i.unsqueeze(-1), j.unsqueeze(-1)), 1)
            split_ind = torch.split(ind, cfg.sequence_length, dim=0)
            input_idx = input_idx + list(split_ind)

            sample_index.append(ind)

    values = pad_sequence(values, batch_first=True)
    input_idx = pad_sequence(input_idx, batch_first=True)

    sample_index = np.vstack(sample_index)
    sample_index = np.concatenate((np.full((sample_index.shape[0], 1), chr), sample_index), 1).astype('int')

    plt.hist(nvals_list, bins=50)
    plt.savefig(cfg.plot_dir + "hist_rowlength_chr%s.pdf" % str(chr))
    plt.close()

    return input_idx, values, sample_index


def contactProbabilities(values, delta=1e-10):
    coeff = np.nan_to_num(1 / (values + delta))
    CP = np.power(1 / np.exp(8), coeff)

    return CP


def get_samples(data, seq_lstm, chr, cfg, dense):
    if dense:
        return get_samples_dense(data, seq_lstm, chr, cfg)
    else:
        return get_samples_sparse(data, seq_lstm, chr, cfg)


def get_data(seq_lstm, cfg, cell, chr):
    data = load_hic(cfg, cell, chr)
    input_idx, values, sample_index = get_samples(data, seq_lstm, chr, cfg, dense=False)

    return input_idx, values, sample_index


def get_data_loader_chr(seq_lstm, cfg, cell, chr, dense=False):
    input_idx, values, sample_index = get_data(seq_lstm, cfg, cell, chr)

    # create dataset, dataloader
    dataset = torch.utils.data.TensorDataset(input_idx.float(), values.float())
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=False)

    return data_loader, sample_index


def get_data_loader(seq_lstm, cfg, cell, dense=False):
    # create input dataset
    values = torch.empty(0, cfg.sequence_length)
    input_idx = torch.empty(0, cfg.sequence_length, 2)
    sample_index = []

    # for chr in list(range(1, 11)) + list(range(12,23)):
    for chr in list(range(22, 23)):
        idx, val, sample_idx = get_data(seq_lstm, cfg, cell, chr)

        values = torch.cat((values, val.float()), 0)
        input_idx = torch.cat((input_idx, idx), 0)
        sample_index.append(sample_idx)

    sample_index = np.vstack(sample_index)

    # save input data
    torch.save(input_idx, cfg.processed_data_dir + 'input_idx.pth')
    torch.save(values, cfg.processed_data_dir + 'values.pth')
    torch.save(sample_index, cfg.processed_data_dir + 'input_index.pth')

    # create dataset, dataloader
    dataset = torch.utils.data.TensorDataset(input_idx, values)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=False)

    return data_loader, sample_index


def get_bedfile(sample_index, cfg):
    chr = sample_index[:, 0]
    bin_idx = sample_index[:, 1]
    start_coord = get_genomic_coord(chr, bin_idx, cfg)
    stop_coord = start_coord + cfg.resolution

    bedfile = pd.DataFrame({'chr': chr, 'start': start_coord, 'stop': stop_coord})
    return bedfile
