import pandas as pd
import logging
import re
from train_fns.config import Config
import numpy as np
from os import listdir
from os.path import isfile, join
from common.log import setup_logging
import traceback
import math

logger = logging.getLogger(__name__)
pd.options.mode.chained_assignment = None


class DataPrepHic():

    def __init__(self, cfg, cell, chr, mode):
        self.cfg = cfg
        self.mode = mode
        self.hic_path = cfg.hic_path
        self.chr = chr
        self.res = cfg.resolution
        self.chr_pos_len = cfg.chr_len
        self.sizes_path = self.cfg.hic_path + self.cfg.sizes_file
        self.sizes = np.load(self.sizes_path).item()
        self.start_end_path = self.cfg.hic_path + self.cfg.start_end_file
        self.start_ends = np.load(self.start_end_path).item()
        self.cell = cell

    def get_data(self):
        hic_chr_path = self.hic_path + self.cell + '/' + self.chr + "/"
        hic_chr_txt = "hic_chr" + self.chr + ".txt"
        chr_hic_file = hic_chr_path + hic_chr_txt

        hic_data = pd.read_csv(chr_hic_file, sep="\s+", header=None)
        hic_data.columns = ["i", "j", "v"]
        hic_data["i"] = (hic_data["i"] / self.res).astype(int)
        hic_data["j"] = (hic_data["j"] / self.res).astype(int)
        hic_data = hic_data.sort_values(by=['i']).reset_index(drop=True)

        sorted_hic_data = pd.DataFrame(columns=["i", "j", "v", "diff"])

        start = self.start_ends["chr" + self.chr]["start"]
        stop = self.start_ends["chr" + self.chr]["stop"]
        for i in range(start, stop):

            try:
                subset_hic = hic_data.loc[hic_data["i"] == i]
                subset_hic = subset_hic.sort_values(by=['j']).reset_index(drop=True)
                subset_hic["diff"] = pd.to_numeric(abs(subset_hic["j"] - subset_hic["i"]))

                subset_hic_left = hic_data.loc[hic_data["j"] == i]
                subset_hic_left = subset_hic_left.sort_values(by=['i']).reset_index(drop=True)
                subset_hic_left = subset_hic_left.rename(columns={"i": "j", "j": "i", "v": "v"})
                subset_hic_left["diff"] = pd.to_numeric(abs(subset_hic_left["j"] - subset_hic_left["i"]))

                if self.cfg.distance_cut:
                    subset_hic = subset_hic.loc[subset_hic["diff"] < self.cfg.distance_cut_off_mb]
                    subset_hic_left = subset_hic_left.loc[subset_hic_left["diff"] < self.cfg.distance_cut_off_mb]

                if self.cfg.long_shot_sep and self.mode == "train":
                    subset_hic = subset_hic.loc[subset_hic["diff"] > self.cfg.long_shot_thresh]
                    subset_hic_left = subset_hic_left.loc[subset_hic_left["diff"] > self.cfg.long_shot_thresh]
                elif self.cfg.long_shot_sep and self.mode == "test":
                    subset_hic = subset_hic.loc[subset_hic["diff"] < self.cfg.long_shot_thresh]
                    subset_hic_left = subset_hic_left.loc[subset_hic_left["diff"] < self.cfg.long_shot_thresh]

                if self.cfg.diag_lstm:
                    subset_hic = subset_hic.append(subset_hic_left, sort=True)
                    subset_hic = subset_hic.sort_values(by=['diff']).drop_duplicates().reset_index(drop=True)
                else:
                    subset_hic = subset_hic.append(subset_hic_left, sort=True)
                    subset_hic = subset_hic.sort_values(by=['j']).drop_duplicates().reset_index(drop=True)

                # subset_hic["v"] = np.log(subset_hic["v"]) + 1
                subset_hic["v"] = self.contactProbabilities(subset_hic["v"])
                sorted_hic_data = sorted_hic_data.append(subset_hic, sort=True)

            except Exception as e:
                logger.error(traceback.format_exc())
                continue

        sorted_hic_data.reset_index(drop=True)

        cum_pos = self.get_cumpos()
        pos_columns = ["i", "j"]
        sorted_hic_data[pos_columns] += cum_pos

        return sorted_hic_data

    def get_cumpos(self):
        chr_num = int(self.chr)
        if chr_num == 1:
            cum_pos = 0
        else:
            key = "chr" + str(chr_num - 1)
            cum_pos = int(np.ceil(int(self.sizes[key]) / self.res))

        return cum_pos

    def contactProbabilities(self, M, delta=1e-10):
        coeff = np.nan_to_num(1 / (M + delta))
        PM = np.power(1 / np.exp(1), coeff)

        return PM


if __name__ == '__main__':
    setup_logging()

    chr = "21"
    cfg = Config()
    data_ob_hic = DataPrepHic(cfg, chr, mode='train')
    hic_data = data_ob_hic.get_data()
