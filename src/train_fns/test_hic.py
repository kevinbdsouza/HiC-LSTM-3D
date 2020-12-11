from __future__ import division
from train_fns.data_prep_hic import DataPrepHic
from train_fns.monitor_testing import MonitorTesting
from train_fns.train_hic import unroll_loop
from eda.viz import Viz
from train_fns.model import Model
import logging
from common.log import setup_logging
from common import utils
import traceback
import pandas as pd
import torch
from torch.autograd import Variable
from keras.callbacks import TensorBoard
import numpy as np
import os

'''
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)
'''

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpu_id = 0
mode = "test"
exp = "hic_test"

logger = logging.getLogger(__name__)


def get_config(model_dir, config_base, result_base):
    seq_lstm__path = os.path.join(model_dir, '/seq_lstm.pth')
    config_path = os.path.join(model_dir, config_base)
    res_path = os.path.join(model_dir, result_base)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    cfg = utils.load_config_as_class(model_dir, config_path, seq_lstm__path, res_path)
    return cfg


def captum_test(cfg, model, subset_hic_data, feature_scores, row_num):
    torch.manual_seed(123)
    np.random.seed(123)
    nValues = len(subset_hic_data)

    hidden_test = torch.rand(nValues, 1, 8).cuda(gpu_id)
    state_test = torch.rand(nValues, 1, 8).cuda(gpu_id)
    hidden_baseline = torch.rand(1, 1, 8).cuda(gpu_id)
    state_baseline = torch.rand(1, 1, 8).cuda(gpu_id)
    input_baseline = torch.rand(1, 1, 32).cuda(gpu_id)
    baseline_stack = (hidden_baseline, state_baseline, input_baseline)

    final_input = torch.rand(1, 1, 32).cuda(gpu_id)
    for ei in range(0, nValues):
        try:
            i_pos, j_pos = torch.tensor(
                subset_hic_data.iloc[ei]["i"]).cuda(gpu_id), torch.tensor(
                subset_hic_data.iloc[ei]["j"]).cuda(gpu_id)
        except Exception as e:
            logger.error(traceback.format_exc())
            continue

        i_embed = model.seq_lstm.pos_embed(i_pos)
        j_embed = model.seq_lstm.pos_embed(j_pos)
        concat_embed = torch.cat((i_embed, j_embed), 0).unsqueeze(0).unsqueeze(0)

        final_input = torch.cat((final_input, concat_embed), 0)

    final_input = final_input[1:, :, :]
    input_stack = (hidden_test, state_test, final_input)

    ig = IntegratedGradients(model.seq_lstm)
    attributions, delta = ig.attribute(input_stack, baseline_stack, target=0, return_convergence_delta=True)

    attributions = torch.mean(attributions[2], 0)
    attributions = attributions.cpu().data.numpy()
    feature_scores[row_num, :] = attributions[0, :cfg.pos_embed_size]

    return feature_scores


def test_hic(cfg, cell, chr, lstm_hidden_states, embed_rows, feature_scores):
    data_ob = DataPrepHic(cfg, cell, mode='test', chr=str(chr))

    monitor = MonitorTesting(cfg)
    callback = TensorBoard(cfg.tensorboard_log_path)
    vizOb = Viz(cfg)

    hic_data = data_ob.get_data()

    model = Model(cfg, gpu_id)
    model.load_weights()
    model.set_callback(callback)

    predicted_hic_data = pd.DataFrame(columns=["i", "j", "v", "diff"])
    start = data_ob.start_ends["chr" + str(chr)]["start"] + data_ob.get_cumpos()
    stop = data_ob.start_ends["chr" + str(chr)]["stop"] + data_ob.get_cumpos()

    seq_lstm_init = True
    iter_num = 0

    seq_lstm_optimizer, criterion = None, None

    try:
        for i in range(start, stop):

            try:
                subset_hic_data = hic_data.loc[hic_data["i"] == i]
                nValues = len(subset_hic_data)
                if nValues == 0:
                    continue
            except:
                continue

            if exp == "hic_test":
                mse, hidden_state, seq_lstm_init, predicted_hic_subset, embed_row, r2 = unroll_loop(cfg,
                                                                                                    subset_hic_data,
                                                                                                    model.seq_lstm,
                                                                                                    seq_lstm_optimizer,
                                                                                                    criterion,
                                                                                                    seq_lstm_init, mode)

                lstm_hidden_states[i, :] = hidden_state
                embed_rows[i, :] = embed_row
                predicted_hic_data = predicted_hic_data.append(predicted_hic_subset, sort=True)

                iter_num += 1
                if iter_num % 500 == 0:
                    logger.info('Iter: {} - mse: {}'.format(iter_num, np.mean(monitor.mse_iter)))
                    # vizOb.plot_prediction(predicted_hic_subset, subset_hic_data, mse, iter_num)

                if mse != 0:
                    monitor.monitor_mse_iter(callback, mse, r2, iter_num)

            elif exp == "captum":
                feature_scores = captum_test(cfg, model, subset_hic_data, feature_scores, i)

    except Exception as e:
        logger.error(traceback.format_exc())

    return lstm_hidden_states, embed_rows, predicted_hic_data, feature_scores


if __name__ == '__main__':
    setup_logging()

    load_embed = False
    #test_chr = list(range(21, 23))
    # test_chr.remove(11)
    test_chr = [22]

    model_dir = '/home/kevindsouza/Documents/projects/hic_lstm/src/saved_model/model_lstm/exp_run'
    config_base = 'config.yaml'
    result_base = 'images'
    cfg = get_config(model_dir, config_base, result_base)

    if load_embed:
        lstm_hidden_states = np.load(model_dir + "/" + "lstm_hidden_states_test.npy")
        embed_rows = np.load(model_dir + "/" + "embed_rows_test.npy")
    else:
        lstm_hidden_states = np.zeros((cfg.chr_len, cfg.hidden_size_lstm))
        embed_rows = np.zeros((cfg.chr_len, cfg.pos_embed_size))

    feature_scores = np.zeros((cfg.chr_len, cfg.pos_embed_size))

    cell = "GM12878"
    for chr in test_chr:
        logger.info('Testing Start Chromosome: {}'.format(chr))

        lstm_hidden_states, embed_rows, predicted_hic_data, feature_scores = test_hic(cfg, cell, chr,
                                                                                      lstm_hidden_states,
                                                                                      embed_rows, feature_scores)

        if exp == "hic_test":
            np.save(cfg.hic_path + "GM12878/" + str(chr) + "/" + "predicted_exp_hic_test_" + str(chr) + ".npy",
                    predicted_hic_data)
            np.save(model_dir + "/" + "lstm_hidden_states_test.npy", lstm_hidden_states)
            np.save(model_dir + "/" + "embed_rows_test.npy", embed_rows)
        elif exp == "captum":
            np.save(model_dir + "/" + "ig_feature_scores.npy", feature_scores)

    print("done")
