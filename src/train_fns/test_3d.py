import numpy as np
import pandas as pd
import torch

import config
from model import SeqLSTM
from data_utils import get_data_loader_chr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_model(model, criterion, cfg, cell, chr):
    # get data
    data_loader, samples = get_data_loader_chr(model, cfg, cell, chr)

    # test model
    predictions, test_error, values = model.test(data_loader)

    # save predictions
    # np.save(cfg.output_directory + "predictions_chr%s.npy" % str(chr), predictions)

    # select all samples of data that are not zero padding
    idx = np.where(np.sum(samples, axis=1) > 0)[0]
    output = pd.DataFrame(samples[idx, :].astype(int), columns=['chr', 'i', 'j'])
    output['pred'] = predictions[idx]
    output['v'] = values[idx]
    output['test_error'] = test_error[idx]

    output.to_csv(cfg.output_directory + "predictions_chr%s.csv" % str(chr), sep="\t")


if __name__ == '__main__':
    # test_chr = list(range(1, 11))
    test_chr = list(range(12, 23))
    # test_chr = [21]
    cfg = config.Config()
    cell = "GM12878"
    # model_name = sys.argv[1]
    model_name = "contact_prob"

    # initalize model
    model = SeqLSTM(cfg, device, model_name).to(device)

    # load model weights
    model.load_weights()

    # get criterion (loss)
    optimizer, criterion = model.compile_optimizer(cfg)

    for chr in test_chr:
        test_model(model, criterion, cfg, cell, chr)
