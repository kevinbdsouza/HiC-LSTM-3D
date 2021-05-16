import numpy as np
import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.clip_grad import clip_grad_norm_
import lstm
from captum.attr import LayerIntegratedGradients


class HicLSTM3D(nn.Module):
    def __init__(self, cfg, device, model_name):
        super(HicLSTM3D, self).__init__()
        self.cfg = cfg
        self.gpu_id = 0
        self.model_name = model_name
        self.device = device
        self.hidden_size_lstm = cfg.hidden_size_lstm

        self.ThreeD_embed = nn.Embedding(cfg.genome_len, cfg.pos_embed_size).train()
        nn.init.normal_(self.ThreeD_embed.weight)

        self.lnlstm = lstm.LSTM(cfg.input_size_lstm, cfg.hidden_size_lstm, batch_first=True)
        self.linear_out = nn.Linear(cfg.hidden_size_lstm * cfg.chunk_length, cfg.output_size_lstm * cfg.chunk_length)

        if cfg.lstm_nontrain:
            self.lnlstm.requires_grad = False
            self.ThreeD_embed.requires_grad = True
            self.linear_out.requires_grad = True

    def forward(self, input):
        hidden, state = self._initHidden(input.shape[0])
        representations = self.ThreeD_embed(input.long())
        representations = representations.view((input.shape[0], self.cfg.chunk_length, -1))
        output, _ = self.lnlstm(representations, (hidden, state))
        output = self.linear_out(output.reshape(input.shape[0], -1))
        return output

    def ko_forward(self, representations):
        hidden, state = self._initHidden(representations.shape[0])
        output, _ = self.lstm(representations, (hidden, state))
        output = self.linear_out(output.reshape(representations.shape[0], -1))
        return output

    def _initHidden(self, batch_size):
        h = Variable(torch.randn(1, batch_size, self.hidden_size_lstm)).to(self.device)
        c = Variable(torch.randn(1, batch_size, self.hidden_size_lstm)).to(self.device)

        return h, c

    def compile_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        criterion = nn.MSELoss()

        return optimizer, criterion

    def load_weights(self):
        try:
            print('loading weights from {}'.format(self.cfg.model_dir))
            self.load_state_dict(torch.load(self.cfg.model_dir + self.model_name + '.pth'))
        except Exception as e:
            print("load weights exception: {}".format(e))

    def get_representations(self, input_ids):
        device = self.device
        input_indices = torch.from_numpy(input_ids).to(torch.int64).to(device)
        representations = self.ThreeD_embed(input_indices)
        return representations.detach().cpu()

    def train_Hic3D(self, hic_loader, optimizer, criterion, sum_writer):
        device = self.device
        cfg = self.cfg
        epochs = cfg.num_epochs

        for ep in range(epochs):
            self.train()
            epoch_loss = 0.0
            for comb, (input_pos, hic_values) in enumerate(tqdm(hic_loader)):
                input_pos = input_pos.to(device)
                hic_values = hic_values.to(device)

                # Forward pass 
                output = self.forward(input_pos)
                loss = criterion(output, hic_values)
                epoch_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.parameters(), max_norm=cfg.max_norm)
                optimizer.step()

                sum_writer.add_scalar('training loss',
                                      loss, comb + ep * len(hic_loader))

            print('Epoch Number %s' % str(epoch + 1))
            print('Mean Epoch loss: %s' % (epoch_loss / len(hic_loader)))

    def test_Hic3D(self, hic_loader):
        device = self.device
        cfg = self.cfg
        chunk_length = cfg.chunk_length

        hic_predictions = torch.empty(0, chunk_length).to(device)
        pred_error = torch.empty(0).to(device)
        hic_labels = torch.empty(0, chunk_length).to(device)

        with torch.no_grad():
            self.eval()
            for comb, (input_pos, hic_values) in enumerate(tqdm(hic_loader)):
                input_pos = input_pos.to(device)
                hic_values = hic_values.to(device)

                hic_labels = torch.cat((hic_labels, hic_values), 0)

                lstm_output = self.forward(input_pos)
                hic_predictions = torch.cat((hic_predictions, lstm_output), 0)

                error = nn.MSELoss(reduction='none')(lstm_output, hic_values)
                pred_error = torch.cat((pred_error, error), 0)

        hic_predictions = torch.reshape(hic_predictions, (-1, 1)).cpu().detach().numpy()
        pred_error = torch.reshape(pred_error, (-1, 1)).cpu().detach().numpy()
        hic_labels = torch.reshape(hic_labels, (-1, 1)).cpu().detach().numpy()

        return hic_labels, hic_predictions, pred_error

    def get_captum_ig(self, hic_loader):
        device = self.device
        cfg = self.cfg
        chunk_length = cfg.sequence_length

        input_baseline = torch.rand(1, chunk_length, 2).float() * 1e5
        baseline_stack = (input_baseline.int().to(device))
        feature_scores = torch.empty(0, chunk_length).to(device)

        for comb, (input_pos, hic_values) in enumerate(tqdm(hic_loader)):
            input_stack = (input_pos.to(device))
            ig_target = list(np.arange(len(input_pos)))
            ig_target = [int(x) for x in ig_target]
            ig = LayerIntegratedGradients(self, self.ThreeD_embed)
            attributions, delta = ig.attribute(input_stack, baseline_stack, target=ig_target,
                                               return_convergence_delta=True)

            attributions = torch.mean(attributions[:, :, 0, :], 2)
            feature_scores = torch.cat((feature_scores, attributions), 0)

        feature_scores = torch.reshape(feature_scores, (-1, 1)).cpu().detach().numpy()

        return feature_scores

    def perform_ko(self, hic_loader, representations):
        device = self.device
        cfg = self.cfg
        chunk_length = cfg.chunk_length
        ko_predictions = torch.empty(0, chunk_length).to(device)

        for i, (input_pos, hic_values) in enumerate(tqdm(hic_loader)):
            input_pos = input_pos.to(device)
            input_pos = input_pos.view(1, -1).squeeze(0).cpu()
            # indices = indices[indices.nonzero()].squeeze(1)
            representations = representations.loc[input_pos, :]
            representations = torch.tensor(representations.values).view(cfg.batch_size, chunk_length,
                                                                        cfg.input_size_lstm)

            with torch.no_grad():
                self.eval()
                lstm_output = self.ko_forward(representations)
                ko_predictions = torch.cat((ko_predictions, lstm_output), 0)

        ko_predictions = torch.reshape(ko_predictions, (-1, 1)).cpu().detach().numpy()

        return ko_predictions





class HicLSTM3D_rep(nn.Module):
    def __init__(self, cfg, device, model_name):
        super(HicLSTM3D_rep, self).__init__()
        self.cfg = cfg
        self.gpu_id = 0
        self.model_name = model_name
        self.device = device
        self.hidden_size_lstm = cfg.hidden_size_lstm

        # self.ThreeD_embed = nn.Embedding(cfg.genome_len, cfg.pos_embed_size).train()
        # nn.init.normal_(self.ThreeD_embed.weight)

        self.pos_mean_embed = nn.Embedding(cfg.genome_len, cfg.pos_embed_size)
        self.pos_var_embed  = nn.Embedding(cfg.genome_len, cfg.pos_embed_size)

        nn.init.normal_(self.pos_mean_embed.weight)
        nn.init.normal_(self.pos_var_embed.weight)
        
        # self._initialize_rep_embed(cfg)

        self.lnlstm     = lstm.LSTM(cfg.input_size_lstm, cfg.hidden_size_lstm, batch_first=True)
        self.linear_out = nn.Linear(cfg.hidden_size_lstm * cfg.chunk_length, cfg.output_size_lstm * cfg.chunk_length)

        if cfg.lstm_nontrain:
            self.lnlstm.requires_grad = False
            # self.ThreeD_embed.requires_grad = True
            self.linear_out.requires_grad = True

    def forward(self, input):
        hidden, state   = self._initHidden(input.shape[0])
        # representations = self.ThreeD_embed(input.long())
        # representations = representations.view((input.shape[0], self.cfg.chunk_length, -1))

        mu      = self.pos_mean_embed(input)
        log_var = self.pos_var_embed( input)
        representations = self.reparameterize(mu, log_var)

        output, _ = self.lnlstm(representations, (hidden, state))
        output = self.linear_out(output.reshape(input.shape[0], -1))
        return output

    def ko_forward(self, representations):
        hidden, state = self._initHidden(representations.shape[0])
        output, _ = self.lstm(representations, (hidden, state))
        output    = self.linear_out(output.reshape(representations.shape[0], -1))
        return output
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def _initialize_rep_embed(self, cfg):
        init_pos_mean = np.random.multivariate_normal(np.zeros(cfg.pos_embed_size),
                                                              np.identity(cfg.pos_embed_size), cfg.genome_len)
        
        init_pos_var = np.random.multivariate_normal(np.zeros(cfg.pos_embed_size),
                                                              np.identity(cfg.pos_embed_size), cfg.genome_len)

        self.pos_mean_embed.weight.data.copy_(torch.from_numpy(init_pos_mean))
        self.pos_var_embed.weight.data.copy_(torch.from_numpy(init_pos_var))


    def _initHidden(self, batch_size):
        h = Variable(torch.randn(1, batch_size, self.hidden_size_lstm)).to(self.device)
        c = Variable(torch.randn(1, batch_size, self.hidden_size_lstm)).to(self.device)
        return h, c

    def compile_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.learning_rate)
        criterion = nn.MSELoss()

        return optimizer, criterion

    def load_weights(self):
        try:
            print('loading weights from {}'.format(self.cfg.model_dir))
            self.load_state_dict(torch.load(self.cfg.model_dir + self.model_name + '.pth'))
        except Exception as e:
            print("load weights exception: {}".format(e))

    def get_representations(self, input_ids):
        device = self.device
        input_indices   = torch.from_numpy(input_ids).to(torch.int64).to(device)
        representations = self.ThreeD_embed(input_indices)
        return representations.detach().cpu()

    def train_Hic3D(self, hic_loader, optimizer, criterion, sum_writer):
        device = self.device
        cfg = self.cfg
        epochs = cfg.num_epochs

        for ep in range(epochs):
            self.train()
            epoch_loss = 0.0
            for comb, (input_pos, hic_values) in enumerate(tqdm(hic_loader)):
                input_pos = input_pos.to(device)
                hic_values = hic_values.to(device)

                # Forward pass 
                output = self.forward(input_pos)
                loss = criterion(output, hic_values)
                epoch_loss += loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.parameters(), max_norm=cfg.max_norm)
                optimizer.step()

                sum_writer.add_scalar('training loss',
                                      loss, comb + ep * len(hic_loader))

            print('Epoch Number %s' % str(ep + 1))
            print('Mean Epoch loss: %s' % (epoch_loss / len(hic_loader)))

    def test_Hic3D(self, hic_loader):
        device = self.device
        cfg = self.cfg
        chunk_length = cfg.chunk_length

        hic_predictions = torch.empty(0, chunk_length).to(device)
        pred_error = torch.empty(0).to(device)
        hic_labels = torch.empty(0, chunk_length).to(device)

        with torch.no_grad():
            self.eval()
            for comb, (input_pos, hic_values) in enumerate(tqdm(hic_loader)):
                input_pos = input_pos.to(device)
                hic_values = hic_values.to(device)

                hic_labels = torch.cat((hic_labels, hic_values), 0)

                lstm_output = self.forward(input_pos)
                hic_predictions = torch.cat((hic_predictions, lstm_output), 0)

                error = nn.MSELoss(reduction='none')(lstm_output, hic_values)
                pred_error = torch.cat((pred_error, error), 0)

        hic_predictions = torch.reshape(hic_predictions, (-1, 1)).cpu().detach().numpy()
        pred_error = torch.reshape(pred_error, (-1, 1)).cpu().detach().numpy()
        hic_labels = torch.reshape(hic_labels, (-1, 1)).cpu().detach().numpy()

        return hic_labels, hic_predictions, pred_error




    # did not update yet
    def get_captum_ig(self, hic_loader):
        device = self.device
        cfg = self.cfg
        chunk_length = cfg.sequence_length

        input_baseline = torch.rand(1, chunk_length, 2).float() * 1e5
        baseline_stack = (input_baseline.int().to(device))
        feature_scores = torch.empty(0, chunk_length).to(device)

        for comb, (input_pos, hic_values) in enumerate(tqdm(hic_loader)):
            input_stack = (input_pos.to(device))
            ig_target = list(np.arange(len(input_pos)))
            ig_target = [int(x) for x in ig_target]
            # ig = LayerIntegratedGradients(self, self.ThreeD_embed)
            attributions, delta = ig.attribute(input_stack, baseline_stack, target=ig_target,
                                               return_convergence_delta=True)

            attributions   = torch.mean(attributions[:, :, 0, :], 2)
            feature_scores = torch.cat((feature_scores, attributions), 0)

        feature_scores = torch.reshape(feature_scores, (-1, 1)).cpu().detach().numpy()

        return feature_scores

    def perform_ko(self, hic_loader, representations):
        device = self.device
        cfg = self.cfg
        chunk_length = cfg.chunk_length
        ko_predictions = torch.empty(0, chunk_length).to(device)

        for i, (input_pos, hic_values) in enumerate(tqdm(hic_loader)):
            input_pos = input_pos.to(device)
            input_pos = input_pos.view(1, -1).squeeze(0).cpu()
            # indices = indices[indices.nonzero()].squeeze(1)
            representations = representations.loc[input_pos, :]
            representations = torch.tensor(representations.values).view(cfg.batch_size, chunk_length,
                                                                        cfg.input_size_lstm)

            with torch.no_grad():
                self.eval()
                lstm_output = self.ko_forward(representations)
                ko_predictions = torch.cat((ko_predictions, lstm_output), 0)

        ko_predictions = torch.reshape(ko_predictions, (-1, 1)).cpu().detach().numpy()

        return ko_predictions
