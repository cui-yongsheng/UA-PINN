import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import os

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

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()
    @staticmethod
    def forward(x):
        return torch.sin(x)

class Predictor(nn.Module):
    def __init__(self,input_dim=40):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(input_dim,32),
            Sin(),
            nn.Linear(32,1)
        )
        self.input_dim = input_dim
    def forward(self,x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self,input_dim):
        super(MLP, self).__init__()
        self.encoder = Encoder(input_dim=input_dim, output_dim=16, layers_num=3, hidden_dim=64, drop_out=0.2)
        self.predictor = Predictor(input_dim=16)
        self.loss_func = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self,input_x):
        encoded_x = self.encoder(input_x)
        predictions = self.predictor(encoded_x)
        return predictions

    def train_one_epoch(self,train_loader, optimizer):
        losses = []
        self.train()
        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            y_pred = self.forward(x)
            loss = self.loss_func(y_pred, y)
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

    def test(self, test_Loader,  n_samples):
        self.eval()
        result_pred = []
        result_true = []
        with torch.no_grad():
            for Xte, yte in test_Loader:
                Xte = Xte.to(self.device)
                ypred = self.forward(Xte)
                ypred = ypred.cpu().numpy()
                result_pred.append(ypred)
                result_true.append(yte)
            result_pred = np.concatenate(result_pred, axis=0)
            result_true = np.concatenate(result_true, axis=0)
        return result_pred, result_true

    def save(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'model.pth'))
    def load(self, load_path):
        self.load_state_dict(torch.load(os.path.join(load_path, 'model.pth'), map_location=self.device))