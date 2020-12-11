import pandas as pd
import os
from train_fns.test_hic import get_config
from in_silico.insilico import InSilico
import numpy as np


class CTCF:
    def __init__(self, cfg, chr, cell, mode):
        self.file_path = os.path.join(cfg.downstream_dir, "ctcf")
        self.motif_path = os.path.join(cfg.downstream_dir, "ctcf", "fimo.tsv")
        self.meme_input = os.path.join(cfg.downstream_dir, "ctcf", "CTCF.meme")
        self.cfg = cfg
        self.chr = chr
        self.cell = cell
        self.insilico_ob = InSilico(cfg, cell, chr, mode)
        self.cum_pos = self.insilico_ob.hic_data_ob.get_cumpos()

    def get_ctcf_data(self):
        data = pd.read_csv(os.path.join(self.file_path, "chr" + str(self.chr) + ".bed"), sep="\t", header=None)
        ctcf_data = self.insilico_ob.alter_data(data)
        return ctcf_data

    def get_ctcf_motif(self):
        motif_data = pd.read_csv(self.motif_path, sep="\t", header=None)
        meme_input = pd.read_csv(self.meme_input, sep="\t", header=None)

    def alter_embed(self, ctcf_data, embed_rows, mode, control):

        if mode == "delete_ctcf":
            if control == "all":
                embed_rows.loc[ctcf_data["pos"], :] = pd.DataFrame(np.zeros((len(ctcf_data), len(embed_rows.columns))),
                                                                   columns=embed_rows.columns)
                embed_rows = embed_rows.fillna(0)
            elif control == "random":
                pass
            pass
        elif mode == "reverse_ctcf":
            if control == "all":
                pass
            elif control == "random":
                pass
            pass

        return embed_rows

    def infer_hic(self, embed_rows, lstm_hidden_states):

        predicted_hic_data = self.insilico_ob.predict_hic(self.cfg, self.chr,
                                                          lstm_hidden_states,
                                                          embed_rows)

        np.save(cfg.hic_path + "GM12878/" + str(chr) + "/" + "predicted_exp_ctcfko_hic_" + str(chr) + ".npy",
                predicted_hic_data)

        pass

    def analyse_knockout(self):

        predicted_hic = pd.DataFrame(np.load(
            cfg.hic_path + "GM12878/" + str(chr) + "/" + "predicted_exp_hic_test_" + str(chr) + ".npy"),
            columns=["diff", "pos", "j", "v"])
        ctcfko_hic = pd.DataFrame(np.load(
            cfg.hic_path + "GM12878/" + str(chr) + "/" + "predicted_exp_ctcfko_hic_" + str(chr) + ".npy"),
            columns=["diff", "pos", "j", "v"])

        ctcf_data = self.get_ctcf_data()
        ctcf_data["pos"] = ctcf_data["start"] + self.cum_pos

        predicted_df = pd.merge(predicted_hic, ctcf_data, on='pos')
        predicted_df = predicted_df.loc[predicted_df["diff"] <= 20]
        ctcfko_df = pd.merge(ctcfko_hic, ctcf_data, on='pos')
        ctcfko_df = ctcfko_df.loc[ctcfko_df["diff"] <= 20]

        predicted_probs = np.zeros((11,))
        ctcfko_probs = np.zeros((11,))
        for diff in range(0, 11):
            subset_predicted = predicted_df.loc[predicted_df["diff"] == diff]
            subset_ctcfko = ctcfko_df.loc[ctcfko_df["diff"] == diff]

            predicted_probs[diff] = subset_predicted["v"].mean()
            ctcfko_probs[diff] = subset_ctcfko["v"].mean()

        pass

    def perfom_ctcf_ko(self, exp_mode, exp_control):

        ctcf_data = self.get_ctcf_data()
        ctcf_data["pos"] = ctcf_data["start"] + self.cum_pos
        embed_rows = pd.DataFrame(self.insilico_ob.embed_rows)

        embed_rows = self.alter_embed(ctcf_data, embed_rows, exp_mode, exp_control)
        lstm_hidden_states = self.insilico_ob.lstm_hidden_states

        self.infer_hic(embed_rows, lstm_hidden_states)

        print("done")

        pass


if __name__ == '__main__':
    chr_list = [21, 22]
    cell = "GM12878"
    model_dir = '/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/exp_run'
    config_base = 'config.yaml'
    result_base = 'images'
    cfg = get_config(model_dir, config_base, result_base)

    exp_mode = "delete_ctcf"
    exp_control = "all"
    for chr in chr_list:
        print("Inferring with CHR {}".format(chr))
        ctcf_ob = CTCF(cfg, chr, cell, mode="test")
        # ctcf_ob.perfom_ctcf_ko(exp_mode, exp_control)

        ctcf_ob.analyse_knockout()

    print("done")
