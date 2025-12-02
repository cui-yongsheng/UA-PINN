import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import os

class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return out

class CNN(nn.Module):
    def __init__(self,input_dim):
        super(CNN, self).__init__()
        self.layer1 = ResBlock(input_channel=1, output_channel=24, stride=1)  # N,8,17
        self.layer2 = ResBlock(input_channel=24, output_channel=28, stride=2)  # N,16,9
        self.layer3 = ResBlock(input_channel=28, output_channel=4, stride=3)  # N,24,5
        flattened_dim = self._calculate_layer6_input_dim(input_dim)
        self.layer6 = nn.Linear(flattened_dim, 1)
        self.loss_func = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _calculate_layer6_input_dim(self, input_dim):
        """
        自动计算经过各层卷积后最终的特征维度，用于初始化 layer6 的输入大小
        """
        # 模拟数据通过网络层的变化
        with torch.no_grad():
            x = torch.randn(1, 1, input_dim)  # 假设batch_size为1，channel为1
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            # 计算展平后的维度
            flattened_dim = x.view(x.size(0), -1).size(1)

        return flattened_dim


    def forward(self, input_x):
        n,l = input_x.shape[0],input_x.shape[1]
        out = input_x.view(n,1,l)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer6(out.view(n,-1))
        return out.view(n,1)

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

    def test(self, test_Loader, n_samples):
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
