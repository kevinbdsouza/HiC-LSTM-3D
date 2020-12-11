from __future__ import division
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class MonitorTesting:

    def __init__(self, cfg):
        self.cfg = cfg
        self.mse_iter = []
        self.r2_iter = []

    def add_tf_summary(self, callback, loss_dict, seq_num):
        for name, value in loss_dict.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            callback.writer.add_summary(summary, seq_num)
            callback.writer.flush()

    def monitor_mse_iter(self, callback, mse, r2, iter_num):
        self.mse_iter.append(mse)
        self.r2_iter.append(r2)
        loss_dict_iter = {'mse_iter': mse, 'r2_iter': r2}
        self.add_tf_summary(callback, loss_dict_iter, iter_num)
