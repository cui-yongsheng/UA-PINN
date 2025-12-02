import numpy as np
import pandas as pd
import shutil
import random
import torch
import time
import math
import os

def quantile_loss(ytrue, ypred, qs):
    '''
    Quantile loss version 2
    Args:
    ytrue (batch_size, output_horizon)
    ypred (batch_size, output_horizon, num_quantiles)
    '''
    L = np.zeros_like(ytrue)
    for i, q in enumerate(qs):
        yq = ypred[:, :, i]
        diff = yq - ytrue
        L += np.max(q * diff, (q - 1) * diff)
    return L.mean()

class StandardScaler:

    def fit_transform(self, y):
        self.mean = np.mean(y)
        self.std = np.std(y) + 1e-4
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        return y * self.std + self.mean

    def transform(self, y):
        return (y - self.mean) / self.std


class MaxScaler:

    def fit_transform(self, y):
        self.max = np.max(y)
        return y / self.max

    def inverse_transform(self, y):
        return y * self.max

    def transform(self, y):
        return y / self.max


class MeanScaler:

    def fit_transform(self, y):
        self.mean = np.mean(y)
        return y / self.mean

    def inverse_transform(self, y):
        return y * self.mean

    def transform(self, y):
        return y / self.mean


class LogScaler:

    def fit_transform(self, y):
        return np.log1p(y)

    def inverse_transform(self, y):
        return np.expm1(y)

    def transform(self, y):
        return np.log1p(y)

def gaussian_likelihood_loss(z, mu, sigma):
    '''
    Gaussian Liklihood Loss
    Args:
    z (tensor): true observations, shape (num_ts, num_periods)
    mu (tensor): mean, shape (num_ts, num_periods)
    sigma (tensor): standard deviation, shape (num_ts, num_periods)
    likelihood:
    (2 pi sigma^2)^(-1/2) exp(-(z - mu)^2 / (2 sigma^2))
    log likelihood:
    -1/2 * (log (2 pi) + 2 * log (sigma)) - (z - mu)^2 / (2 sigma^2)
    '''
    # 方式 1
    # c = 0.5 * math.log(2 * math.pi)
    # negative_likelihood = torch.log(sigma + 1) + (z - mu) ** 2 / (2 * sigma ** 2) + c
    # 方式 2
    distribution = torch.distributions.normal.Normal(mu, sigma)
    negative_likelihood = -distribution.log_prob(z)
    return negative_likelihood.mean()

def negative_binomial_loss(ytrue, mu, alpha):
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
    batch_size, seq_len = ytrue.size()
    likelihood = torch.lgamma(ytrue + 1. / alpha) - torch.lgamma(ytrue + 1) - torch.lgamma(1. / alpha) \
                 - 1. / alpha * torch.log(1 + alpha * mu) \
                 + ytrue * torch.log(alpha * mu / (1 + alpha * mu))
    return - likelihood.mean()

def set_random_seed(seed):
    """
    设置全局随机种子以确保实验可复现性

    Args:
        seed (int): 随机种子值
    """
    # 设置Python内置random模块的种子
    random.seed(seed)

    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # 设置cuDNN相关参数确保确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_save_path():
    # 月日建立一级目录，时分秒是另外1一级目录
    save_path = os.path.join(
        os.getcwd(),
        "results",
        time.strftime("%m-%d", time.localtime()),
        time.strftime("%H-%M-%S", time.localtime())
    )
    os.makedirs(save_path)
    return save_path

def save_code(save_path):
    for file in os.listdir(os.getcwd()):
        if file.endswith(".py"):
            shutil.copy(file, os.path.join(save_path, file))


def save_code(save_path):
    """
    递归复制项目中所有.py文件到保存路径，保持目录结构，排除__init__.py文件和results目录
    Args:
        save_path (str): 目标保存路径
    """
    # 使用shutil.copytree配合ignore参数实现递归复制
    save_path = os.path.join(save_path, "src")
    current_dir = os.getcwd()
    # 遍历当前目录及子目录，复制所有.py文件
    for root, dirs, files in os.walk(current_dir):
        # 排除__pycache__和results目录
        dirs[:] = [d for d in dirs if d not in ('__pycache__', 'results', 'plot','plot_cycle')]
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                # 构造源文件路径
                src_path = os.path.join(root, file)
                # 计算相对于当前工作目录的路径
                rel_path = os.path.relpath(src_path, current_dir)
                # 构造目标文件路径
                dst_path = os.path.join(save_path, rel_path)
                # 创建目标目录
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                # 复制文件
                shutil.copy2(src_path, dst_path)

def get_directory_by_model_and_dataset(csv_file_path, model, dataset_name):
    """
    根据model和dataset_name获取对应的directory值

    参数:
    csv_file_path (str): CSV文件路径
    model (str): 模型名称
    dataset_name (str): 数据集名称

    返回:
    str or None: 对应的directory值，如果未找到则返回None
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 根据model和dataset_name筛选行
    filtered_df = df[(df['model'] == model) & (df['dataset_name'] == dataset_name)]

    # 如果找到了匹配的行，返回directory值
    if not filtered_df.empty:
        return filtered_df.iloc[0]['directory']
    else:
        return None



if __name__ == "__main__":
    pass