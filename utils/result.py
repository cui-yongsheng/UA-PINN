from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

import numpy as np
import torch
import os
import pickle
import json
import math
def plot_loss(losses,save_path):
    plt.plot(range(len(losses)), losses, "k-")
    plt.xlabel("Period")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(save_path, "loss_plot.png"))
    plt.show()

def plot_prediction(results_pred,result_true,save_path):
    p50 = np.quantile(results_pred, 0.5, axis=1)  # (num_samples, seq_len)
    p90 = np.quantile(results_pred, 0.9, axis=1)  # (num_samples, seq_len)
    p10 = np.quantile(results_pred, 0.1, axis=1)  # (num_samples, seq_len)
    num_test = len(p50)
    plt.plot(range(num_test), p50, "r-")  # 绘制50%分位数曲线
    # 绘制10%-90%分位数阴影d
    plt.fill_between(x=range(num_test), y1=p10, y2=p90, alpha=0.5)
    plt.title('Prediction uncertainty')
    plt.plot(range(num_test), result_true)
    plt.legend(["P50 forecast", "P10-P90 quantile", "true"], loc="upper left")
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymax)
    plt.xlabel("Periods")
    plt.ylabel("Y")
    plt.savefig(os.path.join(save_path, "prediction_plot.png"))
    plt.show()

class ResultManager:
    """
    结果管理类，用于保存和读取实验结果
    """

    def __init__(self, save_path):
        """
        初始化结果管理器

        Args:
            save_path (str): 结果保存路径
        """
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def save_results(self, **kwargs):
        """
        保存结果数据，支持任意数量的关键字参数

        Args:
            **kwargs: 需要保存的数据，以关键字形式传入
        """
        for key, value in kwargs.items():
            self._save_single_result(key, value)

    def _save_single_result(self, name, data):
        """
        保存单个结果数据

        Args:
            name (str): 数据名称
            data: 需要保存的数据
        """
        if isinstance(data, np.ndarray):
            # 保存numpy数组为.npy格式
            np.save(os.path.join(self.save_path, f"{name}.npy"), data)
        elif isinstance(data, (dict, list, tuple)) and all(
                isinstance(x, (int, float, str, bool)) or
                (isinstance(x, (list, tuple)) and all(isinstance(y, (int, float, str, bool)) for y in x)) or
                (isinstance(x, dict) and all(isinstance(v, (int, float, str, bool)) for v in x.values()))
                for x in (data.values() if isinstance(data, dict) else data)
                if isinstance(data, dict) or not isinstance(x, dict)):
            # 保存简单数据结构为JSON格式
            with open(os.path.join(self.save_path, f"{name}.json"), 'w') as f:
                json.dump(data, f)
        else:
            # 保存复杂数据结构为pickle格式
            with open(os.path.join(self.save_path, f"{name}.pkl"), 'wb') as f:
                pickle.dump(data, f)

    def load_results(self, *names):
        """
        读取结果数据

        Args:
            *names: 需要读取的数据名称列表

        Returns:
            dict: 包含读取数据的字典
        """
        results = {}
        for name in names:
            file_found = False
            # 尝试不同的文件格式
            for ext in ['.npy', '.json', '.pkl']:
                file_path = os.path.join(self.save_path, f"{name}{ext}")
                if os.path.exists(file_path):
                    results[name] = self._load_single_result(file_path, ext)
                    file_found = True
                    break

            if not file_found:
                print(f"Warning: {name} not found in {self.save_path}")

        return results

    def _load_single_result(self, file_path, ext):
        """
        读取单个结果文件

        Args:
            file_path (str): 文件路径
            ext (str): 文件扩展名

        Returns:
            读取的数据
        """
        if ext == '.npy':
            return np.load(file_path)
        elif ext == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif ext == '.pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)

    def list_available_results(self):
        """
        列出所有可用的结果文件（不包含扩展名）

        Returns:
            list: 可用结果名称列表
        """
        files = os.listdir(self.save_path)
        names = set()
        for file in files:
            name, ext = os.path.splitext(file)
            if ext in ['.npy', '.json', '.pkl']:
                names.add(name)
        return sorted(list(names))

def eval_result(results_pred, true, up, low):
    if isinstance(true, torch.Tensor):
        true = true.cpu().numpy()
    pred_mid = np.quantile(results_pred, 0.5, axis=1, keepdims=True)  # (num_samples, 1)
    pred_up = np.quantile(results_pred, up, axis=1, keepdims=True)  # (num_samples, 1)
    pred_low = np.quantile(results_pred, low, axis=1, keepdims=True)  # (num_samples, 1)
    u = up - low
    # 1. PICP(Prediction Interval Coverage Probability) 要求在区间分位数内部
    PICP = np.mean((true > pred_low) & (true < pred_up))
    # 2. PINAW(Prediction Interval Normalized Averaged Width) 用于定义区间的狭窄程度，在保证准确性的前提下越小越好
    overall_range = np.max(true) - np.min(true)
    PINAW = np.mean((pred_up - pred_low) / overall_range)
    # 3. CWC(coverage width-based criterion) 综合考虑区间覆盖率和狭窄程度, 越小越好
    g = 90  # 取值在50-100
    error = math.exp(-g * (PICP - u))
    if PICP >= u:
        r = 0
    else:
        r = 1
    CWC = PINAW * (1 + r * error)
    # MSE
    mse = mean_squared_error(true, pred_mid)
    # 计算 RMSE
    rmse = np.sqrt(mse)
    # 计算 MAE
    mae = mean_absolute_error(true, pred_mid)
    # 计算 R²
    r2 = r2_score(true, pred_mid)
    return PICP, PINAW, CWC, mse, rmse, mae, r2


def kde_analysis_precise(results_pred, true, up, low):
    """
    使用KDE对results_pred的每一行进行精确分析，输出最大概率值及10%和90%分位点
    Args:
        results_pred (np.ndarray): 形状为(N, 100)的预测结果数组
        true (np.ndarray, optional): 形状为(N,)的真实值数组
    Returns:
        dict: 包含kde_max_values, p10_values, p90_values等分析结果的字典
    """
    N = results_pred.shape[0]
    # 存储结果的数组
    kde_max_values = np.zeros((N,1))  # KDE最大值点
    p10_values = np.zeros((N,1))  # 10%分位点
    p50_values = np.zeros((N,1))  # 50%分位点（中位数）
    p90_values = np.zeros((N,1))  # 90%分位点
    # 对每一行进行KDE分析
    for i in range(N):
        if (i + 1) % 50 == 0:
            print(f"处理进度: {i + 1}/{N}")
        # 获取当前行的数据
        row_data = results_pred[i, :]
        # 使用更精细的网格和交叉验证选择最优带宽
        xmin, xmax = row_data.min() - 1, row_data.max() + 1
        xs = np.linspace(xmin, xmax, 400)  # 更精细的网格
        params = {"bandwidth": np.logspace(-1.2, 0.6, 20)}  # 更宽的带宽范围和更多选项
        grid = GridSearchCV(KernelDensity(kernel="gaussian"), params, cv=5)
        grid.fit(row_data[:, None])
        kde_skl = grid.best_estimator_
        log_pdf = kde_skl.score_samples(xs[:, None])
        pdf_skl = np.exp(log_pdf)
        # 计算KDE最大值点
        max_kde_idx = np.argmax(pdf_skl)
        kde_max_values[i] = xs[max_kde_idx]
        # 使用概率密度函数计算分位点（通过数值积分得到CDF）
        dx = xs[1] - xs[0]
        cdf = np.cumsum(pdf_skl) * dx
        # 归一化CDF
        cdf = cdf / cdf[-1]
        # 通过插值找到精确的分位点
        def find_quantile(quantile_value):
            # 找到最接近目标分位数的索引
            idx = np.argmin(np.abs(cdf - quantile_value))
            # 如果正好匹配，直接返回
            if cdf[idx] == quantile_value:
                return xs[idx]
            # 否则进行线性插值
            elif cdf[idx] > quantile_value:
                # 在idx-1和idx之间插值
                if idx > 0:
                    frac = (quantile_value - cdf[idx - 1]) / (cdf[idx] - cdf[idx - 1])
                    return xs[idx - 1] + frac * (xs[idx] - xs[idx - 1])
                else:
                    return xs[idx]
            else:
                # 在idx和idx+1之间插值
                if idx < len(cdf) - 1:
                    frac = (quantile_value - cdf[idx]) / (cdf[idx + 1] - cdf[idx])
                    return xs[idx] + frac * (xs[idx + 1] - xs[idx])
                else:
                    return xs[idx]
        p10_values[i,0] = find_quantile(0.1)
        p50_values[i,0] = find_quantile(up)
        p90_values[i,0] = find_quantile(low)
    # 构建结果字典
    result_dict = {
        "kde_max": kde_max_values,
        "p10": p10_values,
        "p50": p50_values,
        "p90": p90_values
    }

    # 计算误差
    if isinstance(true, torch.Tensor):
        true = true.cpu().numpy()
    kde_max_errors = np.abs(kde_max_values - true)
    pred_mid = np.quantile(results_pred, 0.5, axis=1, keepdims=True)  # (num_samples, 1)
    p50_errors = np.abs(pred_mid - true)
    result_dict["true"] = true
    result_dict["kde_max_errors"] = kde_max_errors
    result_dict["p50_errors"] = p50_errors
    # 打印统计信息
    print(f"\n=== 精确KDE分析结果 ===")
    print(f"KDE最大值平均误差: {np.mean(kde_max_errors):.4f} ± {np.std(kde_max_errors):.4f}")
    print(f"中位数平均误差: {np.mean(p50_errors):.4f} ± {np.std(p50_errors):.4f}")

    return result_dict


if __name__ == '__main__':
    pass


