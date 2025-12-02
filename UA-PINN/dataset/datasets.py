from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch
import os

class DataList:
    def __init__(self, root, few_shot=False):
        self.root = root
        self.few_shot = few_shot    # 设置小样本训练
    def get_NASA_list(self):
        data_root = os.path.join(self.root, "NASA data")
        files = os.listdir(data_root)
        test_id = ['B0025', 'B0029', 'B0033', 'B0037', 'B0041', 'B0045', 'B0049', 'B0053']
        train_list = []
        test_list = []
        for f in files:
            if f[:-4] in test_id:
                f_root = os.path.join(data_root, f)
                test_list.append(f_root)
            else:
                f_root = os.path.join(data_root, f)
                train_list.append(f_root)
        if self.few_shot:
            train_list = []
            train_id = ['B0018', 'B0026', 'B0030', 'B0034', 'B0038', 'B0042', 'B0046', 'B0050', 'B0054']
            for f in files:
                if f[:-4] in train_id:
                    f_root = os.path.join(data_root, f)
                    train_list.append(f_root)
        return train_list, test_list

    def get_SUN_list(self):
        data_root = os.path.join(self.root, "SUN data")
        files = os.listdir(data_root)
        test_id = ['data_part_1','data_part_2']
        train_list = []
        test_list = []
        for f in files:
            if f[:-4] in test_id:
                f_root = os.path.join(data_root, f)
                test_list.append(f_root)
            else:
                f_root = os.path.join(data_root, f)
                train_list.append(f_root)
        if self.few_shot:
            train_list = []
            train_id = ['data_part_4']
            for f in files:
                if f[:-4] in train_id:
                    f_root = os.path.join(data_root, f)
                    train_list.append(f_root)
        return train_list, test_list

    def load_data(self, dataset_name):
        method_name = f'get_{dataset_name}_list'
        if not hasattr(self, method_name):
            raise ValueError(f'Dataset {dataset_name} not found')
        return getattr(self, method_name)()

def load_one_battery(path):
    data = pd.read_csv(path)
    return data
def pre_process(data,normalize=True):
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    data = data.reset_index(drop=True)
    out_index = []
    for col in data.columns:
        ser1 = data[col]
        rule = (ser1.mean() - 3 * ser1.std() > ser1) | (ser1.mean() + 3 * ser1.std() < ser1)
        index = np.arange(ser1.shape[0])[rule]
        out_index.extend(index)
    out_index = list(set(out_index))
    data = data.drop(out_index, axis=0)
    data = data.reset_index(drop=True)
    if normalize:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        return data, scaler
    return data

def load_data(data_list,normalize=True,shuffle=False):
    data_frames = []
    scaler = None
    for data_path in data_list:
        data = load_one_battery(data_path)
        if 'SMART data' not in data_path:
            data.insert(data.shape[1] - 1, 'cycle index', np.arange(data.shape[0]))
        data,scaler = pre_process(data,normalize=normalize)
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        data_frames.append(data)
    data_all = pd.concat(data_frames, axis=0, ignore_index=True)
    if shuffle:
        data_all = data_all.sample(frac=1, random_state=42).reset_index(drop=True)
    return data_all,scaler


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset_name, train=True, few_shot=False):
        self.root = root
        self.train = train
        self.train_list,self.test_list = DataList(root,few_shot).load_data(dataset_name)
        if train and few_shot:
            assert 'few_shot 不影响测试集大小'
        self.train_data,_ = load_data(self.train_list)
        self.test_data,_ = load_data(self.test_list)
        # 预先转换为numpy数组并设定数据类型
        self.train_features = torch.from_numpy(self.train_data.iloc[:, :-1].values.astype(np.float32))
        self.train_targets = torch.from_numpy(self.train_data.iloc[:, -1].values.astype(np.float32)).unsqueeze(1)
        self.test_features = torch.from_numpy(self.test_data.iloc[:, :-1].values.astype(np.float32))
        self.test_targets = torch.from_numpy(self.test_data.iloc[:, -1].values.astype(np.float32)).unsqueeze(1)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    def __getitem__(self, idx):
        if self.train:
            return self.train_features[idx], self.train_targets[idx]
        else:
            return self.test_features[idx], self.test_targets[idx]

class PINNDataset(torch.utils.data.Dataset):
    def __init__(self, root, dataset_name, train=True, few_shot=False):
        self.root = root
        self.train = train
        self.train_list,self.test_list = DataList(root,few_shot).load_data(dataset_name)
        if train and few_shot:
            assert 'few_shot 不影响测试集大小'
        self.train_data_prev,_ = load_data(self.train_list,shuffle=False)
        self.train_data_next,_ = load_data(self.train_list,shuffle=True)
        self.test_data_prev,_ = load_data(self.test_list,shuffle=False)
        self.test_data_next,_ = load_data(self.test_list,shuffle=True)
        # 预先转换为numpy数组并设定数据类型
        self.train_features_prev = torch.from_numpy(self.train_data_prev.iloc[:, :-1].values.astype(np.float32))
        self.train_targets_prev = torch.from_numpy(self.train_data_prev.iloc[:, -1].values.astype(np.float32)).unsqueeze(1)
        self.test_features_prev = torch.from_numpy(self.test_data_prev.iloc[:, :-1].values.astype(np.float32))
        self.test_targets_prev = torch.from_numpy(self.test_data_prev.iloc[:, -1].values.astype(np.float32)).unsqueeze(1)
        self.train_features_next = torch.from_numpy(self.train_data_next.iloc[:, :-1].values.astype(np.float32))
        self.train_targets_next = torch.from_numpy(self.train_data_next.iloc[:, -1].values.astype(np.float32)).unsqueeze(1)
        self.test_features_next = torch.from_numpy(self.test_data_next.iloc[:, :-1].values.astype(np.float32))
        self.test_targets_next = torch.from_numpy(self.test_data_next.iloc[:, -1].values.astype(np.float32)).unsqueeze(1)

    def __len__(self):
        if self.train:
            return len(self.train_data_prev)
        else:
            return len(self.test_data_prev)
    def __getitem__(self, idx):
        if self.train:
            return self.train_features_prev[idx], self.train_targets_prev[idx], self.train_features_next[idx], self.train_targets_next[idx]
        else:
            return self.test_features_prev[idx], self.test_targets_prev[idx], self.test_features_next[idx], self.test_targets_next[idx]