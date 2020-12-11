import pandas as pd
import logging
import os
from common import utils
import seaborn as sns
import numpy as np
from common.log import setup_logging
from train_fns.test_hic import get_config
from train_fns.data_prep_hic import DataPrepHic
import traceback
from keras.callbacks import TensorBoard
from train_fns.monitor_testing import MonitorTesting
from train_fns.model import Model
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

gpu_id = 0
logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None


class InSilico():

    def __init__(self, cfg, cell, chr, mode):
        self.cfg = cfg
        self.mode = mode
        self.hic_path = cfg.hic_path
        self.chr = chr
        self.embed_rows = np.load(
            "/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/exp_run/embed_rows_test.npy")
        self.lstm_hidden_states = np.load(
            "/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/exp_run/lstm_hidden_states_test.npy")
        self.predicted_hic_data = np.load(
            cfg.hic_path + cell + "/" + str(chr) + "/" + "predicted_exp_hic_test_" + str(chr) + ".npy")
        self.cell = cell
        self.res = cfg.resolution
        self.sizes_path = self.cfg.hic_path + self.cfg.sizes_file
        self.sizes = np.load(self.sizes_path).item()
        self.hic_data_ob = DataPrepHic(self.cfg, self.cell, mode=mode, chr=str(self.chr))
        self.og_hic_data = None

    def get_og_hicmat(self):
        self.og_hic_data = self.hic_data_ob.get_data()
        og_hic_df = self.converto_hicmat(self.og_hic_data)
        return og_hic_df

    def get_windowmat_hic(self, hic_mat):

        window_hic = None

        window_hic = hic_mat.loc[4100:4400, 4100:4400]
        #4265
        #window_hic = window_hic - 0.1
        self.plot_hic_mats(window_hic)

        return window_hic

    def converto_hicmat(self, hic_data):

        cum_pos = self.hic_data_ob.get_cumpos()
        pos_columns = ["i", "j"]
        hic_data[pos_columns] -= cum_pos

        nrow = hic_data.iloc[len(hic_data) - 1]["i"] + 1
        ncol = nrow
        hic_mat = np.zeros((nrow, ncol))

        rows = hic_data["i"].to_numpy(dtype=int)
        cols = hic_data["j"].to_numpy(dtype=int)
        data = hic_data["v"].to_numpy(dtype=float)

        hic_mat[rows, cols] = data
        hic_mat[cols, rows] = data

        hic_df = pd.DataFrame(hic_mat)

        return hic_df

    def alter_data(self, data):
        column_list = ["chr", "start", "end", "dot", "score", "dot_2", "enrich", "pval", "qval", "peak"]
        data.columns = column_list
        data['target'] = 1
        data = data.filter(['start', 'end', "target"], axis=1)

        data["start"] = (data["start"]).astype(int) // self.res
        data["end"] = (data["end"]).astype(int) // self.res
        data = data.filter(['start', 'end', 'target'], axis=1)

        data = data.sort_values('start')
        return data

    def plot_hic_mats(self, hic_df1):

        ax = sns.heatmap(hic_df1, xticklabels=False, yticklabels=False, cbar=True, vmin=0, vmax=1)
        plt.show()

        pass

    def load_predicted_hic(self, path):

        hic_mat = np.load(path)
        hic_df = pd.DataFrame(hic_mat, columns=["diff", "i", "j", "v"])

        return hic_df

    def predict_hic(self, cfg, chr, lstm_hidden_states, embed_rows):
        monitor = MonitorTesting(cfg)
        callback = TensorBoard(cfg.tensorboard_log_path)

        model = Model(cfg, gpu_id)
        model.load_weights()
        model.set_callback(callback)
        seq_lstm = model.seq_lstm

        predicted_hic_data = pd.DataFrame(columns=["i", "j", "v", "diff"])
        start = self.hic_data_ob.start_ends["chr" + str(chr)]["start"] + self.hic_data_ob.get_cumpos()
        stop = self.hic_data_ob.start_ends["chr" + str(chr)]["stop"] + self.hic_data_ob.get_cumpos()

        seq_lstm_init = True
        iter_num = 0

        try:
            for i in range(start, stop):

                try:
                    subset_hic_data = self.og_hic_data.loc[self.og_hic_data["i"] == i]
                    nValues = len(subset_hic_data)
                    if nValues == 0:
                        continue
                except:
                    continue

                predicted_hic_subset = subset_hic_data.copy(deep=True)
                if seq_lstm_init:
                    _, lstm_state = seq_lstm.initHidden()
                lstm_hidden = Variable(torch.tensor(lstm_hidden_states[i, :]).float().unsqueeze(0).unsqueeze(0)).cuda(
                    gpu_id)

                loss = 0
                for ei in range(0, nValues):

                    i_embed = torch.tensor(embed_rows.loc[subset_hic_data.iloc[ei]["i"]][:]).cuda(gpu_id)
                    j_embed = torch.tensor(embed_rows.loc[subset_hic_data.iloc[ei]["j"]][:]).cuda(gpu_id)
                    concat_embed = torch.cat((i_embed, j_embed), 0)
                    lstm_input = Variable(concat_embed.float().unsqueeze(0).unsqueeze(0)).cuda(gpu_id)

                    lstm_output, lstm_hidden, lstm_state = seq_lstm(lstm_hidden, lstm_state, lstm_input)

                    lstm_target = Variable(
                        torch.from_numpy(np.array(subset_hic_data.iloc[ei]["v"])).float().unsqueeze(0)).cuda(gpu_id)

                    if torch.isnan(lstm_target) or torch.isnan(lstm_output):
                        continue

                    lstm_prediction = lstm_output.squeeze(0).cpu().data.numpy()
                    predicted_hic_subset.at[ei, "v"] = lstm_prediction

                    loss += np.power((lstm_prediction - subset_hic_data.iloc[ei]["v"]), 2)

                mean_mse = loss / nValues
                mean_y = subset_hic_data["v"].mean()
                ss_reg = ((predicted_hic_subset["v"].sub(mean_y)) ** 2).sum()
                ss_tot = ((subset_hic_data["v"].sub(mean_y)) ** 2).sum()
                ss_res = ((subset_hic_data["v"] - predicted_hic_subset["v"]) ** 2).sum()
                r2 = 1 - (ss_res / ss_tot)

                iter_num += 1
                if iter_num % 500 == 0:
                    logger.info(
                        'Iter: {} - mse: {} - r2: {}'.format(iter_num, np.mean(monitor.mse_iter),
                                                             np.mean(monitor.r2_iter)))

                if mean_mse != 0:
                    monitor.monitor_mse_iter(callback, mean_mse, r2, iter_num)

                predicted_hic_data = predicted_hic_data.append(predicted_hic_subset, sort=True)

        except Exception as e:
            logger.error(traceback.format_exc())

        return predicted_hic_data


if __name__ == '__main__':
    setup_logging()

    chr = 21
    cell = "GM12878"
    model_dir = '/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/exp_run'
    config_base = 'config.yaml'
    result_base = 'images'
    cfg = get_config(model_dir, config_base, result_base)

    in_silico_ob = InSilico(cfg, cell, chr, mode='test')
    og_hic_df = in_silico_ob.get_og_hicmat()
    window_df = in_silico_ob.get_windowmat_hic(og_hic_df)
    in_silico_ob.plot_hic_mats(window_df)

    # path_ctcfko = cfg.hic_path + "GM12878/" + str(chr) + "/" + "predicted_exp_ctcfko_hic_" + str(chr) + ".npy"
    # path_og_pred_hic = cfg.hic_path + "GM12878/" + str(chr) + "/" + "predicted_exp_hic_test_" + str(chr) + ".npy"
    # hic_df = in_silico_ob.load_predicted_hic(path_og_pred_hic)
    # hic_df = in_silico_ob.converto_hicmat(hic_df)
    # in_silico_ob.plot_hic_mats(hic_df)

    #in_silico_ob.plot_knockout_results()

    print("done")
