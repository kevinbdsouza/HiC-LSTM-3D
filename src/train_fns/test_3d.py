import train_fns.config as config
import numpy as np
import pandas as pd
import torch
from train_fns.model import HicLSTM3D
from train_fns.data_utils import get_data_loader_chr

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def model_test(model, chr, cell, cfg):
    # get hic data
    hic_loader, unsplit_ids = get_hic_loader_chr(chr, cell, cfg)

    # test model
    hic_labels, hic_predictions, pred_error = model.test_Hic3D(hic_loader)

    # select samples
    index = np.where(np.sum(unsplit_ids, axis=1) > 0)[0]
    output_df = pd.DataFrame(unsplit_ids[index, :].astype(int), columns=['chr', 'i', 'j'])
    output_df['pred'] = hic_predictions[index]
    output_df['value'] = hic_labels[index]
    output_df['prediction_error'] = pred_error[index]

    output_df.to_csv(cfg.output_directory + "predictions_chr%s.csv" % str(chr), sep="\t")


if __name__ == '__main__':
    # test_chr = list(range(12, 23))

    test_chr = [21]
    cfg = config.Config()
    cell = "GM12878"
    model_name = "contact_prob"

    model = HicLSTM3D(cfg, device, model_name).to(device)
    model.load_weights()
    optimizer, criterion = model.compile_optimizer()

    for chr in test_chr:
        model_test(model, chr, cell, cfg)
