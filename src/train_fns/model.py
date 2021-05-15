import numpy as np
import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
import train_fns.lstm as lstm
from captum.attr import LayerIntegratedGradients


class HicLSTM3D(nn.Module):
    def __init__(self, cfg, device, model_name):
        super(HicLSTM3D, self).__init__()
        self.device = device
        self.cfg = cfg
        self.hidden_size_lstm = cfg.hidden_size_lstm
        self.gpu_id = 0
        self.model_name = model_name

        self.pos_embed = nn.Embedding(cfg.genome_len, cfg.pos_embed_size).train()
        nn.init.normal_(self.pos_embed.weight)

        self.lstm = lstm.LSTM(cfg.input_size_lstm, cfg.hidden_size_lstm, batch_first=True)
        self.out = nn.Linear(cfg.hidden_size_lstm * cfg.sequence_length, cfg.output_size_lstm * cfg.sequence_length)

        if cfg.lstm_nontrain:
            self.lstm.requires_grad = False
            self.pos_embed.requires_grad = True
            self.out.requires_grad = True

    def forward(self, input):
        hidden, state = self._initHidden(input.shape[0])
        embeddings = self.pos_embed(input.long())
        embeddings = embeddings.view((input.shape[0], self.cfg.sequence_length, -1))
        output, _ = self.lstm(embeddings, (hidden, state))
        output = self.out(output.reshape(input.shape[0], -1))
        return output

    def ko_forward(self, embeddings):
        hidden, state = self._initHidden(embeddings.shape[0])
        output, _ = self.lstm(embeddings, (hidden, state))
        output = self.out(output.reshape(embeddings.shape[0], -1))
        return output

    def _initHidden(self, batch_size):
        h = Variable(torch.randn(1, batch_size, self.hidden_size_lstm)).to(self.device)
        c = Variable(torch.randn(1, batch_size, self.hidden_size_lstm)).to(self.device)

        return h, c

    def compile_optimizer(self, cfg):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        criterion = nn.MSELoss()

        return optimizer, criterion

    def load_weights(self):
        try:
            print('loading weights from {}'.format(self.cfg.model_dir))
            self.load_state_dict(torch.load(self.cfg.model_dir + self.model_name + '.pth'))
        except Exception as e:
            print("load weights exception: {}".format(e))


    def get_embeddings(self, indices):
        device = self.device
        indices = torch.from_numpy(indices).to(torch.int64).to(device)
        embeddings = self.pos_embed(indices)
        return embeddings.detach().cpu()

    def train_model(self, data_loader, criterion, optimizer, writer):
        device = self.device
        cfg = self.cfg
        num_epochs = cfg.num_epochs

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for i, (indices, values) in enumerate(tqdm(data_loader)):
                indices = indices.to(device)
                values = values.to(device)

                # Forward pass 
                output = self.forward(indices)
                loss = criterion(output, values)
                running_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.parameters(), max_norm=cfg.max_norm)
                optimizer.step()

                writer.add_scalar('training loss',
                                  loss, i + epoch * len(data_loader))

            print('Completed epoch %s' % str(epoch + 1))
            print('Average loss: %s' % (running_loss / len(data_loader)))

    def test(self, data_loader):
        device = self.device
        cfg = self.cfg
        num_outputs = cfg.sequence_length

        predictions = torch.empty(0, num_outputs).to(device)
        test_error = torch.empty(0).to(device)
        target_values = torch.empty(0, num_outputs).to(device)

        with torch.no_grad():
            self.eval()
            for i, (indices, values) in enumerate(tqdm(data_loader)):
                indices = indices.to(device)
                values = values.to(device)

                target_values = torch.cat((target_values, values), 0)

                lstm_output = self.forward(indices)
                predictions = torch.cat((predictions, lstm_output), 0)

                error = nn.MSELoss(reduction='none')(lstm_output, values)
                test_error = torch.cat((test_error, error), 0)

        predictions = torch.reshape(predictions, (-1, 1)).cpu().detach().numpy()
        test_error = torch.reshape(test_error, (-1, 1)).cpu().detach().numpy()
        target_values = torch.reshape(target_values, (-1, 1)).cpu().detach().numpy()

        return predictions, test_error, target_values

    def get_captum_ig(self, data_loader):
        device = self.device
        cfg = self.cfg
        num_outputs = cfg.sequence_length

        input_baseline = torch.rand(1, num_outputs, 2).float() * 1e5
        baseline_stack = (input_baseline.int().to(device))
        feature_scores = torch.empty(0, num_outputs).to(device)

        for i, (indices, values) in enumerate(tqdm(data_loader)):
            input_stack = (indices.to(device))
            ig_target = list(np.arange(len(indices)))
            ig_target = [int(x) for x in ig_target]
            ig = LayerIntegratedGradients(self, self.pos_embed)
            attributions, delta = ig.attribute(input_stack, baseline_stack, target=ig_target,
                                               return_convergence_delta=True)

            attributions = torch.mean(attributions[:, :, 0, :], 2)
            feature_scores = torch.cat((feature_scores, attributions), 0)

        feature_scores = torch.reshape(feature_scores, (-1, 1)).cpu().detach().numpy()

        return feature_scores

    def perform_ko(self, data_loader, embeddings):
        device = self.device
        cfg = self.cfg
        num_outputs = cfg.sequence_length
        ko_predictions = torch.empty(0, num_outputs).to(device)

        for i, (indices, values) in enumerate(tqdm(data_loader)):
            indices = indices.to(device)
            indices = indices.view(1, -1).squeeze(0).cpu()
            # indices = indices[indices.nonzero()].squeeze(1)
            embeddings = embeddings.loc[indices, :]
            embeddings = torch.tensor(embeddings.values).view(cfg.batch_size, num_outputs, cfg.input_size_lstm)

            with torch.no_grad():
                self.eval()
                lstm_output = self.ko_forward(embeddings)
                ko_predictions = torch.cat((ko_predictions, lstm_output), 0)

        ko_predictions = torch.reshape(ko_predictions, (-1, 1)).cpu().detach().numpy()

        return ko_predictions
