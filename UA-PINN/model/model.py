import torch.nn.functional as F
from torch import nn
from utils.util import gaussian_likelihood_loss,negative_binomial_loss
from tqdm import tqdm
import numpy as np
import torch
import math
import os

class Gaussian(nn.Module):

    def __init__(self, hidden_size, output_size):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

        # initialize weights
        # nn.init.xavier_uniform_(self.mu_layer.weight)
        # nn.init.xavier_uniform_(self.sigma_layer.weight)

    def forward(self, h):  # h为神经网络隐藏层输出 (batch, hidden_size)
        sigma_t = F.softplus(self.sigma_layer(h), beta=1)
        sigma_t = torch.clamp(sigma_t, min=1e-6)
        mu_t = self.mu_layer(h)
        return mu_t, sigma_t  # (batch, output_size)


class NegativeBinomial(nn.Module):

    def __init__(self, input_size, output_size):
        '''
        Negative Binomial Supports Positive Count Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(NegativeBinomial, self).__init__()
        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)

    def forward(self, h):  # h为神经网络隐藏层输出 (batch, hidden_size)
        _, hidden_size = h.size()
        alpha_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        mu_t = torch.log(1 + torch.exp(self.mu_layer(h)))
        return mu_t, alpha_t  # (batch, output_size)


def gaussian_sample(mu, sigma):
    '''
    Gaussian Sample
    Args:
    ytrue (array like)
    mu (array like) # (num_ts, 1)
    sigma (array like): standard deviation # (num_ts, 1)
    gaussian maximum likelihood using log
        l_{G} (z|mu, sigma) = (2 * pi * sigma^2)^(-0.5) * exp(- (z - mu)^2 / (2 * sigma^2))
    '''
    # likelihood = (2 * np.pi * sigma ** 2) ** (-0.5) * \
    #         torch.exp((- (ytrue - mu) ** 2) / (2 * sigma ** 2))
    # return likelihood
    gaussian = torch.distributions.normal.Normal(mu, sigma)
    ypred = gaussian.rsample()
    return ypred


def negative_binomial_sample(mu, alpha):
    '''
    Negative Binomial Sample
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)
    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))
    minimize loss = - log l_{nb}
    Note: torch.lgamma: log Gamma function
    '''
    var = mu + mu * mu * alpha
    ypred = mu + torch.randn() * torch.sqrt(var)
    return ypred

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()
    @staticmethod
    def forward(x):
        return torch.sin(x)

class Encoder(nn.Module):
    def __init__(self,input_dim=17,output_dim=1,layers_num=4,hidden_dim=50,drop_out=0.2):
        super(Encoder, self).__init__()
        assert layers_num >= 2, "layers must be greater than 2"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_num = layers_num
        self.hidden_dim = hidden_dim
        self.layers = []

        for i in range(layers_num):
            if i == 0:
                self.layers.append(nn.Linear(input_dim,hidden_dim))
                self.layers.append(Sin())
            elif i == layers_num-1:
                self.layers.append(nn.Linear(hidden_dim,output_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim,hidden_dim))
                self.layers.append(Sin())
                self.layers.append(nn.Dropout(p=drop_out))
        self.net = nn.Sequential(*self.layers)
        self._init()

    def _init(self):
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self,x):
        x = self.net(x)
        return x

class Predictor(nn.Module):
    def __init__(self,input_dim=40,output_dim=10):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(input_dim,32),
            Sin(),
            nn.Linear(32,output_dim)
        )
        self.input_dim = input_dim
    def forward(self,x):
        return self.net(x)

class DeepAR(nn.Module):

    def __init__(self, input_dim, encoder_dim, num_layers, likelihood, device):
        super(DeepAR, self).__init__()

        # network
        self.encoder = Encoder(input_dim=input_dim, output_dim=encoder_dim, layers_num=num_layers, hidden_dim=60, drop_out=0.2)
        if likelihood == "g":
            self.likelihood_layer = Gaussian(encoder_dim, 1)
        elif likelihood == "nb":
            self.likelihood_layer = NegativeBinomial(encoder_dim, 1)
        self.likelihood = likelihood
        self.device = device

    def forward(self, input_x):
        encoded_x = self.encoder(input_x)
        mu, sigma = self.likelihood_layer(encoded_x)
        if self.likelihood == "g":
            y_sample = gaussian_sample(mu, sigma)
        elif self.likelihood == "nb":
            alpha_t = sigma
            mu_t = mu
            y_sample = negative_binomial_sample(mu_t, alpha_t)
        else:
            raise ValueError("likelihood must be g or nb")
        return y_sample, mu, sigma
    def train_one_epoch(self,train_loader, optimizer):
        losses = []
        self.train()
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_sample, mu, sigma = self.forward(x)
            if self.likelihood == "g":
                loss = gaussian_likelihood_loss(y, mu, sigma)
            elif self.likelihood == "nb":
                loss = negative_binomial_loss(y, mu, sigma)
            else:
                raise ValueError("likelihood must be g or nb")
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return sum(losses) / len(losses) if losses else 0.0
    def Train(self, train_loader, optimizer, scheduler, num_epoches):
        losses = []
        loop = tqdm(range(num_epoches),desc="Training Progress")
        for epoch in loop:
            loss = self.train_one_epoch(train_loader, optimizer)
            scheduler.step()
            losses.append(loss)
            loop.set_description(f'Train Epoch [{epoch + 1}/{num_epoches}]')
            loop.set_postfix(loss=loss)
        return losses
    def test(self,test_Loader,n_samples):
        self.eval()
        results_pred = []
        with torch.no_grad():
            loop = tqdm(range(n_samples), desc="Test Progress")
            for sample in loop:
                result_pred = []
                result_true = []
                for Xte, yte in test_Loader:
                    Xte = Xte.to(self.device)
                    y_sample, mu, sigma = self.forward(Xte)
                    ypred = y_sample.cpu().numpy()
                    result_pred.append(ypred)
                    result_true.append(yte)
                result_pred = np.concatenate(result_pred, axis=0)
                result_true = np.concatenate(result_true, axis=0)
                results_pred.append(result_pred)
                loop.set_description(f'Test sample [{sample + 1}/{n_samples}]')
            results_pred = np.concatenate(results_pred, axis=1)
        return results_pred, result_true

    def save(self, save_path):
        model = {'encoder': self.encoder.state_dict(),
                 'likelihood_layer': self.likelihood_layer.state_dict()}
        torch.save(model, os.path.join(save_path, 'model.pth'))

    def load(self,load_path):
        checkpoint = torch.load(os.path.join(load_path, 'model.pth'), map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.likelihood_layer.load_state_dict(checkpoint['likelihood_layer'])


class UA_PINN(nn.Module):

    def __init__(self, input_dim, encoder_dim, num_layers, likelihood, device):
        super(UA_PINN, self).__init__()
        # network
        self.encoder = Encoder(input_dim=input_dim, output_dim=encoder_dim, layers_num=num_layers, hidden_dim=60, drop_out=0.2)
        self.predictor = Predictor(input_dim=encoder_dim,output_dim=1)
        self.dynamical_F = Encoder(input_dim=2*input_dim+1, output_dim=1, layers_num=num_layers, hidden_dim=60, drop_out=0.2)
        if likelihood == "g":
            self.likelihood_layer = Gaussian(encoder_dim, 1)
        elif likelihood == "nb":
            self.likelihood_layer = NegativeBinomial(encoder_dim, 1)
        self.pinn_loss_func = nn.MSELoss()
        self.pred_loss_func = nn.MSELoss()
        self.relu = nn.ReLU()
        self.likelihood = likelihood
        self.device = device

    def forward(self, input_x):
        x = input_x[:, :-1].clone().detach().requires_grad_(True)  # 叶子，可对 x 求偏导
        t = input_x[:, -1:].clone().detach().requires_grad_(True)  # 叶子，可对 t 求偏导

        z = torch.cat([x, t], dim=1)
        encoded_x = self.encoder(z)  # 依赖 x,t
        mu, sigma = self.likelihood_layer(encoded_x)
        if self.likelihood == "g":
            y_sample = gaussian_sample(mu, sigma)
        elif self.likelihood == "nb":
            alpha_t = sigma
            mu_t = mu
            y_sample = negative_binomial_sample(mu_t, alpha_t)
        else:
            raise ValueError("likelihood must be g or nb")
        # 版本二
        u = y_sample

        # 一次性对 (t, x) 求梯度，避免重复反传 & 图释放问题
        u_t, u_x = torch.autograd.grad(
            outputs=u.sum(),
            inputs=(t, x),
            create_graph=True,
            retain_graph=True  # 若后续还会对同一图继续求导，保留；否则可去掉
        )

        f_value = self.dynamical_F(torch.cat([input_x, u, u_x, u_t], dim=1))
        f = u_t - f_value
        y_pred = u
        return y_pred, f, y_sample, mu, sigma

    def train_one_epoch(self,train_loader, optimizer):
        self.train()
        losses = []
        for x_prev, y_prev, x_next, y_next in train_loader:
            x_prev = x_prev.to(self.device)
            y_prev = y_prev.to(self.device)
            x_next = x_next.to(self.device)
            y_next = y_next.to(self.device)
            y_prev_pred, f_prev, y_prev_sample, mu_prev, sigma_prev = self.forward(x_prev)
            y_next_pred, f_next, y_next_sample, mu_next, sigma_next = self.forward(x_next)
            if self.likelihood == "g":
                loss_prev = gaussian_likelihood_loss(y_prev, mu_prev, sigma_prev)
                loss_next = gaussian_likelihood_loss(y_next, mu_next, sigma_next)
            elif self.likelihood == "nb":
                loss_prev = negative_binomial_loss(y_prev, mu_prev, sigma_prev)
                loss_next = negative_binomial_loss(y_next, mu_next, sigma_next)
            else:
                raise ValueError("likelihood must be g or nb")
            # PINN损失
            loss_pred = 0.5 * self.pred_loss_func(y_prev_pred,y_prev) + 0.5 * self.pred_loss_func(y_next_pred,y_next)
            f_target = torch.zeros_like(f_prev)
            loss_pinn = 0.5 * self.pinn_loss_func(f_prev, f_target) + 0.5 * self.pinn_loss_func(f_next, f_target)
            # 约束损失
            loss_constraint = self.relu(torch.mul(mu_prev - mu_next, y_next - y_prev)).mean()
            loss = loss_prev + loss_next + 0.2 * loss_pinn + 0.2 * loss_pred + 0.2 * loss_constraint
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return sum(losses) / len(losses) if losses else 0.0
    def Train(self, train_loader, optimizer, scheduler, num_epoches):
        losses = []
        loop = tqdm(range(num_epoches),desc="Training Progress")
        for epoch in loop:
            loss = self.train_one_epoch(train_loader, optimizer)
            scheduler.step()
            losses.append(loss)
            loop.set_description(f'Train Epoch [{epoch + 1}/{num_epoches}]')
            loop.set_postfix(loss=loss)
        return losses

    def test(self, test_Loader, n_samples):
        self.eval()
        results_pred = []
        result_true = []
        with torch.no_grad():
            loop = tqdm(range(n_samples), desc="Test Progress")
            for sample in loop:
                result_pred = []
                result_true = []
                for Xte, yte, _, _ in test_Loader:
                    Xte = Xte.to(self.device)
                    y_pred, y_sample, mu, sigma = self.predict(Xte)
                    ypred = y_sample.cpu().numpy()
                    result_pred.append(ypred)
                    result_true.append(yte)
                result_pred = np.concatenate(result_pred, axis=0)
                result_true = np.concatenate(result_true, axis=0)
                results_pred.append(result_pred)
                loop.set_description(f'Test sample [{sample + 1}/{n_samples}]')
            results_pred = np.concatenate(results_pred, axis=1)
        return results_pred, result_true

    def predict(self, input_x):
        encoded_x = self.encoder(input_x)
        u = self.predictor(encoded_x)
        mu, sigma = self.likelihood_layer(encoded_x)
        if self.likelihood == "g":
            y_sample = gaussian_sample(mu, sigma)
        elif self.likelihood == "nb":
            alpha_t = sigma
            mu_t = mu
            y_sample = negative_binomial_sample(mu_t, alpha_t)
        else:
            raise ValueError("likelihood must be g or nb")
        y_pred = u
        return y_pred, y_sample, mu, sigma

    def save(self, save_path):
        model = {'encoder': self.encoder.state_dict(),
                 'predictor': self.predictor.state_dict(),
                'dynamical_F': self.dynamical_F.state_dict(),
                 'likelihood_layer': self.likelihood_layer.state_dict()}
        torch.save(model, os.path.join(save_path, 'model.pth'))

    def load(self, load_path):
        checkpoint = torch.load(os.path.join(load_path, 'model.pth'), map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        self.dynamical_F.load_state_dict(checkpoint['dynamical_F'])
        self.likelihood_layer.load_state_dict(checkpoint['likelihood_layer'])

