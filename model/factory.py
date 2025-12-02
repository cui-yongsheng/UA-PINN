from model.model import DeepAR,UA_PINN
from model.CNN import CNN
from model.MLP import MLP
from utils.args import get_args
import torch
def build(model_name="UA_PINN"):
    if model_name == "MLP":
        return MLP(input_dim=17)
    elif model_name == "CNN":
        return CNN(input_dim=17)
    elif model_name == "DeepAR":
        args = get_args()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return DeepAR(input_dim=17,
                   encoder_dim=args.encoder_dim,
                   num_layers=args.num_layers,
                   likelihood=args.likelihood,
                   device=device)

    elif model_name == "UA_PINN":
        args = get_args()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return UA_PINN(input_dim=17,
                   encoder_dim=args.encoder_dim,
                   num_layers=args.num_layers,
                   likelihood=args.likelihood,
                   device=device
                   ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
