from torch import nn
import torch
from abc import ABCMeta, abstractmethod
import json
from nte.utils import get_md5_checksum
import datetime
import numpy as np
from nte import NTE_MODEL_PATH

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()
        self.name = config["model_name"]
        self.config = config
        self.timesteps = config["timesteps"]
        self.dependency_meta = self.config['dependency_meta']

    @abstractmethod
    def evaluate(self, data):
        pass

    def save(self):
        torch.save(self, f'{self.name}.ckpt')
        with open(f"{self.name}.meta", 'w') as f:
            json.dump({"md5": get_md5_checksum([f"{self.name}.ckpt"]), "timestamp": str(datetime.datetime.now()),
                       "dependency_meta": self.dependency_meta}, f)

    def load(self):
        pass


class RNN(nn.Module):
    def __init__(self, ninp, nhid, rnn_type, nlayers, nclasses, batch_first=True):
        super(RNN, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nhid = nhid
        self.rnn_type = rnn_type
        

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn_cell = getattr(nn, rnn_type)(ninp, nhid, nlayers, batch_first=True).to(device)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `rnn_type` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn_cell = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, batch_first=batch_first).to(device)
        self.fc1 = nn.Linear(nhid, nclasses)
        self = self.to(device)

    def forward(self, X):
        if len(X.shape)!=3:
            X = X.reshape([-1, X.shape[1], 1])        
        output, self.state = self.rnn_cell(X)
        y_hat = self.fc1(output[:, -1, :])
        return y_hat

    def evaluate(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        prediction_probabilities = torch.softmax(self.forward(data.reshape([-1, data.shape[1], 1])), dim=-1)
        predicted_value, predicted_class = torch.max(prediction_probabilities, 1)
        return predicted_value.cpu().detach().numpy(), predicted_class.cpu().detach().numpy(), prediction_probabilities.cpu().detach().numpy()


class Linear(Model):
    def __init__(self, config):
        super().__init__(config=config)
        self.timesteps = config["timesteps"]
        self.layers = nn.ModuleList([])
        self.sigmoid_activation = torch.nn.Sigmoid()
        self.softmax_activation = torch.nn.Softmax(dim=-1)
        for e, node in enumerate(self.config['dnn_config']["layers"]):
            prev_node = config["timesteps"] if e == 0 else self.config['dnn_config']["layers"][e - 1]
            self.layers.extend([nn.Linear(prev_node, node)])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.sigmoid_activation(layer(x))
        return self.layers[-1](x)

    def evaluate(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        prediction_probabilities = torch.softmax(self.forward(data.reshape([-1, self.timesteps])), dim=-1)
        predicted_value, predicted_class = torch.max(prediction_probabilities, 1)
        return predicted_value.cpu().detach().numpy(), predicted_class.cpu().detach().numpy(), prediction_probabilities.cpu().detach().numpy()


class CSaliencyNetwork(nn.Module):
    def __init__(self, N_TIMESTEPS):
        super(CSaliencyNetwork, self).__init__()
        self.timesteps = N_TIMESTEPS
        self.w = torch.nn.Parameter(data=torch.tensor(np.ones([N_TIMESTEPS]), dtype=torch.float32), requires_grad=True)
        self.register_parameter('w', self.w)

    def forward(self, time_series):
        return torch.sigmoid(self.w)


class CSaliencyLoss(nn.Module):
    def __init__(self, N_TIMESTEPS):
        self.timesteps = N_TIMESTEPS
        super(CSaliencyLoss, self).__init__()

    def forward(self, y_hat, y_star, saliency_values, x, x_star, alpha=1, beta=1, gamma=1):
        MSE = torch.nn.MSELoss(reduction="mean")
        loss_m = alpha * MSE(y_hat, y_star)  # Encourage y_hat to match y_star
        loss_budget = beta * (1 / self.timesteps) * torch.sum(
            saliency_values)  # Reduce the sum of the saliency values
        loss_variance = gamma * torch.abs(torch.sum(x - x_star))
        return loss_m + loss_budget + loss_variance, loss_m, loss_budget, loss_variance


class GLESaliencyNetwork(nn.Module):
    def __init__(self, N_TIMESTEPS):
        super(GLESaliencyNetwork, self).__init__()
        self.timesteps = N_TIMESTEPS
        self.w = torch.nn.Parameter(data=torch.tensor(np.random.normal(size=[N_TIMESTEPS]), dtype=torch.float32),
                                    requires_grad=True)
        self.register_parameter('w', self.w)

    def forward(self, time_series):
        return torch.sigmoid(self.w)


class GLESaliencyLoss(nn.Module):
    def __init__(self, N_TIMESTEPS):
        self.timesteps = N_TIMESTEPS
        super(GLESaliencyLoss, self).__init__()

    def forward(self, y_hat, y_star, saliency_values, x, x_star, alpha=1, beta=1, gamma=1):
        MSE = torch.nn.MSELoss(reduction="mean")
        loss_m = alpha * MSE(y_hat, y_star)  # Encourage y_hat to match y_star
        loss_budget = beta * self.timesteps / torch.sum(
            saliency_values)  # (1 / hyper_params.sequence_length) * torch.sum(saliency_values)#-torch.sum(torch.abs(noise)*saliency_values)#  # Reduce the sum of the saliency values
        loss_variance = gamma * torch.abs(torch.sum(x - x_star))
        return (loss_m + loss_budget + loss_variance), loss_m, loss_budget, loss_variance


class LSaliencyNetwork(nn.Module):
    def __init__(self, N_TIMESTEPS):
        super(LSaliencyNetwork, self).__init__()
        self.timesteps = N_TIMESTEPS
        self.linear = torch.nn.Linear(in_features=N_TIMESTEPS, out_features=N_TIMESTEPS)

    def forward(self, time_series):
        return torch.sigmoid(self.linear(time_series))


class RSaliencyNetwork(nn.Module):
    def __init__(self, N_TIMESTEPS):
        super(RSaliencyNetwork, self).__init__()
        self.timesteps = N_TIMESTEPS
        self.rnn = RNN(ninp=self.timesteps, nhid=self.timesteps, nlayers=1, nclasses=2, rnn_type='GRU')

    def forward(self, time_series):
        return self.rnn(time_series)


class LSaliencyLoss(nn.Module):
    def __init__(self, N_TIMESTEPS):
        self.timesteps = N_TIMESTEPS
        super(LSaliencyLoss, self).__init__()
        self.loss = torch.nn.MSELoss(reduction="mean")

    def forward(self, y_hat, y_star, saliency_values, x, x_star, alpha=1, beta=1, gamma=1):
        loss_m = alpha * self.loss(y_hat, y_star)  # Encourage y_hat to match y_star
        loss_budget = beta * (1 / self.timesteps) * torch.sum(
            saliency_values)  # Reduce the sum of the saliency values
        loss_variance = gamma * torch.abs(torch.sum(x - x_star))
        return loss_m + loss_budget + loss_variance, loss_m, loss_budget, loss_variance



class BlipV3DNNModel():
    def __new__(cls):
        return torch.load(NTE_MODEL_PATH+'/blipv3/blip_v3_dnn_mse.ckpt')


class BlipV3RNNModel():
    def __new__(cls):
        return torch.load(NTE_MODEL_PATH+'/blipv3/blip_v3_rnn_mse.ckpt')


class WaferDNNModel():
    def __new__(cls):
        return torch.load(NTE_MODEL_PATH+'/wafer/wafer_dnn_mse.ckpt')