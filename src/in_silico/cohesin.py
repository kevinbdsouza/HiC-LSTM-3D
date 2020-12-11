import pandas as pd
import os
import numpy as np
from train_fns.test_hic import get_config
from in_silico.insilico import InSilico


class Cohesin:
    def __init__(self, cfg, chr, cell, mode):
        self.cohesin_path = os.path.join(cfg.downstream_dir, "cohesin")
        self.rad21_file_name = "rad21.bed"
        self.smc3_file_name = "smc3.bed"
        self.pol2_file_name = "pol2.bam"
        self.cfg = cfg
        self.chr = chr
        self.insilico_ob = InSilico(cfg, cell, chr, mode)
        self.cum_pos = self.insilico_ob.hic_data_ob.get_cumpos()

    def get_cohesin_data(self):
        rad_data = pd.read_csv(os.path.join(self.cohesin_path, self.rad21_file_name), sep="\t", header=None)
        rad_data = rad_data.loc[rad_data[:][0] == "chr" + str(self.chr)]
        rad_data = self.insilico_ob.alter_data(rad_data)

        smc_data = pd.read_csv(os.path.join(self.cohesin_path, self.smc3_file_name), sep="\t", header=None)
        smc_data = smc_data.loc[smc_data[:][0] == "chr" + str(self.chr)]
        smc_data = self.insilico_ob.alter_data(smc_data)

        return rad_data, smc_data

    def alter_embed(self, cohesin_data, embed_rows, mode, control):

        if mode == "delete_cohesin":
            if control == "all":
                embed_rows.loc[cohesin_data["pos"], :] = pd.DataFrame(
                    np.zeros((len(cohesin_data), len(embed_rows.columns))),
                    columns=embed_rows.columns)
                embed_rows = embed_rows.fillna(0)
            elif control == "random":
                pass
            pass

        return embed_rows

    def infer_hic(self, embed_rows, lstm_hidden_states, complex):

        predicted_hic_data = self.insilico_ob.predict_hic(self.cfg, self.chr,
                                                          lstm_hidden_states,
                                                          embed_rows)

        np.save(
            cfg.hic_path + "GM12878/" + str(chr) + "/" + "predicted_exp_" + complex + "_ko_hic_" + str(chr) + ".npy",
            predicted_hic_data)

        pass

    def analyse_knockout(self):
        predicted_hic = pd.DataFrame(np.load(
            cfg.hic_path + "GM12878/" + str(chr) + "/" + "predicted_exp_hic_test_" + str(chr) + ".npy"),
            columns=["diff", "pos", "j", "v"])
        radko_hic = pd.DataFrame(np.load(
            cfg.hic_path + "GM12878/" + str(chr) + "/" + "predicted_exp_rad21_ko_hic_" + str(chr) + ".npy"),
            columns=["diff", "pos", "j", "v"])
        smc3ko_hic = pd.DataFrame(np.load(
            cfg.hic_path + "GM12878/" + str(chr) + "/" + "predicted_exp_smc3_ko_hic_" + str(chr) + ".npy"),
            columns=["diff", "pos", "j", "v"])

        rad_data, smc_data = self.get_cohesin_data()
        rad_data["pos"] = rad_data["start"] + self.cum_pos
        smc_data["pos"] = smc_data["start"] + self.cum_pos

        predicted_df = pd.merge(predicted_hic, rad_data, on='pos')
        predicted_df = predicted_df.loc[predicted_df["diff"] <= 20]
        radko_df = pd.merge(radko_hic, rad_data, on='pos')
        radko_df = radko_df.loc[radko_df["diff"] <= 20]
        smc3ko_df = pd.merge(smc3ko_hic, smc_data, on='pos')
        smc3ko_df = smc3ko_df.loc[smc3ko_df["diff"] <= 20]

        predicted_probs = np.zeros((11,))
        radko_probs = np.zeros((11,))
        smc3ko_probs = np.zeros((11,))
        for diff in range(0, 11):
            subset_predicted = predicted_df.loc[predicted_df["diff"] == diff]
            subset_radko = radko_df.loc[radko_df["diff"] == diff]
            subset_smc3ko = smc3ko_df.loc[smc3ko_df["diff"] == diff]

            predicted_probs[diff] = subset_predicted["v"].mean()
            radko_probs[diff] = subset_radko["v"].mean()
            smc3ko_probs[diff] = subset_smc3ko["v"].mean()
        pass

    def perfom_cohesin_ko(self, exp_mode, exp_control, complex):

        rad_data, smc_data = self.get_cohesin_data()
        rad_data["pos"] = rad_data["start"] + self.cum_pos
        smc_data["pos"] = smc_data["start"] + self.cum_pos

        if complex == "rad21":
            cohesin_data = rad_data
        elif complex == "smc3":
            cohesin_data = smc_data

        embed_rows = pd.DataFrame(self.insilico_ob.embed_rows)

        embed_rows = self.alter_embed(cohesin_data, embed_rows, exp_mode, exp_control)
        lstm_hidden_states = self.insilico_ob.lstm_hidden_states

        self.infer_hic(embed_rows, lstm_hidden_states, complex)

        print("done")

        pass


if __name__ == '__main__':
    chr_list = [21, 22]
    comlpex_list = ["smc3"]
    cell = "GM12878"
    mode = "test"
    model_dir = '/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/exp_run'
    config_base = 'config.yaml'
    result_base = 'images'
    cfg = get_config(model_dir, config_base, result_base)

    exp_mode = "delete_cohesin"
    exp_control = "all"
    for complex in comlpex_list:
        print("Inferring with complex {} KO".format(complex))
        for chr in chr_list:
            print("Inferring with CHR {}".format(chr))
            cohesin_ob = Cohesin(cfg, chr, cell, mode)
            # cohesin_ob.perfom_cohesin_ko(exp_mode, exp_control, complex)
            cohesin_ob.analyse_knockout()

    print("done")
