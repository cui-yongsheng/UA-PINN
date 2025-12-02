# 导入相关包
from torch.optim import Adam
from dataset.datasets import Dataset,PINNDataset
from model.model import DeepAR, UA_PINN
from model.CNN import CNN
from model.MLP import MLP
from utils.args import get_args,save_args
from utils.util import set_random_seed,save_code,get_save_path
from utils.result import plot_loss,plot_prediction,ResultManager,eval_result
import numpy as np
import torch
import os

def main():
    # 参数解析
    args = get_args()
    args.save_path = get_save_path()
    save_args(args)
    set_random_seed(args.seed)
    save_code(args.save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 结果保存
    result_save_path = os.path.join(args.save_path, 'result')
    os.makedirs(result_save_path, exist_ok=True)
    # 不同模型对应不同数据加载和处理方式
    if args.model == 'UA_PINN':
        train_dataset = PINNDataset(root='./data', dataset_name=args.dataset_name, train=True, few_shot=args.few_shot)
        test_dataset = PINNDataset(root='./data', dataset_name=args.dataset_name, train=False)
        input_dim = train_dataset[0][0].shape[0]
        model = UA_PINN(input_dim=input_dim,
            encoder_dim=args.encoder_dim,
            num_layers=args.num_layers,
            likelihood=args.likelihood,
            device = device
        ).to(device)
    elif args.model == 'DeepAR' or args.model == 'MLP' or args.model == 'CNN':
        train_dataset = Dataset(root='./data', dataset_name=args.dataset_name, train=True, few_shot=args.few_shot)
        test_dataset = Dataset(root='./data', dataset_name=args.dataset_name, train=False)
        input_dim = train_dataset[0][0].shape[0]
        if args.model == 'DeepAR':
            model = DeepAR(input_dim=input_dim,
                encoder_dim=args.encoder_dim,
                num_layers=args.num_layers,
                likelihood=args.likelihood,
                device = device
            ).to(device)
    else:
        raise ValueError("Invalid model type")

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size,shuffle=False)
    # 模型训练
    if args.model == 'CNN' or args.model == 'MLP':
        results_pred = []
        for sample in range(args.n_samples):
            set_random_seed(sample+2)
            if args.model == 'CNN':
                model = CNN(input_dim=input_dim).to(device)
            elif args.model == 'MLP':
                model = MLP(input_dim=input_dim).to(device)
            optimizer = Adam(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.99 ** args.num_epochs)
            losses = model.Train(train_loader, optimizer, scheduler, args.num_epochs)
            result_pred, result_true = model.test(test_loader, args.n_samples)
            results_pred.append(result_pred)
            print(f'Sample [{sample + 1}/{args.n_samples}]')
        results_pred = np.concatenate(results_pred, axis=1)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.99 ** args.num_epochs)
        losses = model.Train(train_loader, optimizer, scheduler, args.num_epochs)
        results_pred, result_true = model.test(test_loader, args.n_samples)
        model.save(result_save_path)
    manage = ResultManager(result_save_path)
    manage.save_results(results_pred=results_pred,result_true=result_true,losses=losses)
    # 结果绘制
    plot_loss(losses, args.save_path)
    plot_prediction(results_pred, result_true, args.save_path)
    PICP, PINAW, CWC, mse, rmse, mae, r2 = eval_result(results_pred, result_true, 0.9, 0.1)
    # 创建结果字典
    results_dict = {
        'PICP': float(PICP),
        'PINAW': float(PINAW),
        'CWC': float(CWC),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'MSE': float(mse)
    }
    manage.save_results(results_dict=results_dict)


if __name__ == '__main__':
    main()
