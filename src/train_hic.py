from __future__ import division
from train_fns import config
from train_fns.data_prep_hic import DataPrepHic
from train_fns.monitor_training import MonitorTraining
from train_fns.model import Model
import logging
from common.log import setup_logging
import traceback
import pandas as pd
import torch
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
from keras.callbacks import TensorBoard
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

gpu_id = 1
mode = 'train'

logger = logging.getLogger(__name__)


def train_iter_hic(cfg, chr, lstm_hidden_states, embed_rows):
    data_ob = DataPrepHic(cfg, mode='train', chr=str(chr))

    model = Model(cfg, gpu_id)
    model.load_weights()
    monitor = MonitorTraining(cfg)

    callback = TensorBoard(cfg.tensorboard_log_path)
    monitor.save_config_as_yaml(cfg.config_file, cfg)

    seq_lstm_optimizer, criterion = model.compile_optimizer(cfg)
    model.set_callback(callback)

    hic_data = data_ob.get_data()
    predicted_hic_data = pd.DataFrame(columns=["i", "j", "v", "diff"])
    start = data_ob.start_ends["chr" + str(chr)]["start"] + data_ob.get_cumpos()
    stop = data_ob.start_ends["chr" + str(chr)]["stop"] + data_ob.get_cumpos()

    seq_lstm_init = True

    for epoch_num in range(cfg.num_epochs):

        logger.info('Epoch {}/{}'.format(epoch_num + 1, cfg.num_epochs))

        iter_num = 0

        try:
            for i in range(start, stop):

                try:
                    subset_hic_data = hic_data.loc[hic_data["i"] == i]
                    nValues = len(subset_hic_data)
                    if nValues == 0:
                        continue
                except:
                    continue

                loss, hidden_state, seq_lstm_init, predicted_hic_subset, embed_row = unroll_loop(cfg,
                                                                                                 subset_hic_data,
                                                                                                 model.seq_lstm,
                                                                                                 seq_lstm_optimizer,
                                                                                                 criterion,
                                                                                                 seq_lstm_init, mode)

                # logger.info('Hidden states: {}'.format(encoder_hidden_states_np))
                lstm_hidden_states[i, :] = hidden_state
                embed_rows[i, :] = embed_row
                predicted_hic_data = predicted_hic_data.append(predicted_hic_subset, sort=True)

                model.save_weights()

                iter_num += 1
                if iter_num % 500 == 0:
                    logger.info('Iter: {} - rec_loss: {}'.format(iter_num, np.mean(monitor.losses_iter)))

                if loss != 0:
                    monitor.monitor_loss_iter(callback, loss, iter_num)

            save_flag = monitor.monitor_loss_epoch(callback, epoch_num)
            if save_flag == 'True':
                model.save_weights()

        except Exception as e:
            logger.error(traceback.format_exc())
            model.save_weights()
            continue

    model.save_weights()
    logging.info('Training complete, exiting.')

    return lstm_hidden_states, embed_rows, predicted_hic_data


def unroll_loop(cfg, subset_hic_data, seq_lstm, seq_lstm_optimizer,
                criterion, seq_lstm_init, mode):
    if seq_lstm_init:
        lstm_hidden, lstm_state = seq_lstm.initHidden()
    if mode == "train":
        seq_lstm_optimizer.zero_grad()

    nValues = len(subset_hic_data)
    predicted_hic_subset = subset_hic_data

    loss = 0
    for ei in range(0, nValues):

        try:
            i_pos, j_pos = torch.tensor(
                subset_hic_data.iloc[ei]["i"]).cuda(gpu_id), torch.tensor(
                subset_hic_data.iloc[ei]["j"]).cuda(gpu_id)


        except Exception as e:
            logger.error(traceback.format_exc())
            continue

        i_embed = seq_lstm.pos_embed(i_pos)
        j_embed = seq_lstm.pos_embed(j_pos)
        concat_embed = torch.cat((i_embed, j_embed), 0)
        lstm_input = Variable(concat_embed.float().unsqueeze(0).unsqueeze(0)).cuda(gpu_id)

        lstm_output, lstm_hidden, lstm_state = seq_lstm(lstm_hidden, lstm_state, lstm_input)

        lstm_target = Variable(
            torch.from_numpy(np.array(subset_hic_data.iloc[ei]["v"])).float().unsqueeze(0)).cuda(gpu_id)

        if torch.isnan(lstm_target):
            continue

        lstm_prediction = lstm_output.squeeze(0).cpu().data.numpy()
        predicted_hic_subset.iloc[ei]["v"] = lstm_prediction

        if mode == "train":
            loss += criterion(lstm_output.squeeze(0).squeeze(0), lstm_target)
        else:
            loss += np.power((lstm_prediction - subset_hic_data.iloc[ei]["v"]), 2)

    if mode == "train":
        if loss != 0:
            loss.backward()
            clip_grad_norm_(seq_lstm.parameters(), max_norm=cfg.max_norm)
            seq_lstm_optimizer.step()
            mean_loss = loss.item() / nValues
        else:
            mean_loss = 0
    else:
        mean_loss = loss / nValues

    hidden_state_row = lstm_hidden.squeeze(0).cpu().data.numpy()
    embed_row = i_embed.cpu().data.numpy()
    # cell_state_row = lstm_state.squeeze(0).cpu().data.numpy()

    return mean_loss, hidden_state_row, seq_lstm_init, predicted_hic_subset, embed_row


if __name__ == '__main__':
    setup_logging()

    train_chr = range(2, 23, 2)
    # train_chr = [20, 22]

    cfg = config.Config()
    lstm_hidden_states = np.zeros((cfg.chr_len, cfg.hidden_size_lstm))
    embed_rows = np.zeros((cfg.chr_len, cfg.pos_embed_size))

    for chr in train_chr:
        lstm_hidden_states, embed_rows, predicted_hic_data = train_iter_hic(cfg, chr, lstm_hidden_states, embed_rows)

        np.save(cfg.hic_path + str(chr) + "/" + "predicted_hic.npy", predicted_hic_data)

    np.save(cfg.hic_path + "lstm_hidden_states.npy", lstm_hidden_states)
    np.save(cfg.hic_path + "embed_rows.npy", embed_rows)

    print("done")
