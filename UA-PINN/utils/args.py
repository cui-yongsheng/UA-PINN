# args.py
import argparse
import json
import os

def get_args():
    parser = argparse.ArgumentParser(description='DeepAR Model Arguments')
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    # 数据集参数
    parser.add_argument('--dataset_name', type=str, default='NASA', help='数据集名称')
    parser.add_argument('--few_shot', action='store_true', help='是否使用小样本训练')
    # 模型参数
    parser.add_argument('--model', type=str, default='CNN', help='模型名称')
    parser.add_argument('--num_layers', type=int, default=3, help='模型层数')
    parser.add_argument('--encoder_dim', type=int, default=32, help='编码器维度')
    parser.add_argument('--likelihood', type=str, default='g', help='likelihood类型')
    # 其他参数
    parser.add_argument('--show_plot', action='store_true', default=True, help='是否显示结果')
    parser.add_argument('--train_batch_size', type=int, default=1024, help='批次大小')
    parser.add_argument('--test_batch_size', type=int, default=4096, help='批次大小')
    parser.add_argument('--n_samples', type=int, default=1, help='样本数量')
    parser.add_argument('--seed', type=int, default=2, help='随机种子')
    args = parser.parse_args()
    return args


def save_args(args):
    # 将参数保存到文件
    filename = os.path.join(args.save_path, 'args.json')
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Arguments saved to {filename}")
if __name__ == '__main__':
    args = get_args()
    print(args)

