import argparse
import torch
import random
import numpy as np
import os




def fix_seed(seed):
    """

    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='WEANet: Wavelet Inductive Bias meets Polyphonic Convolutional Core')


    parser.add_argument('--is_training', type=int, default=1, help='status: 1 for training, 0 for testing')
    parser.add_argument('--task_name', type=str, default='forecasting',
                        help='task: forecasting, imputation, classification')
    parser.add_argument('--dataset', type=str, default='ETTh1', help='dataset name')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model weights')

    parser.add_argument('--d_depth', type=int, default=3, help='Network Depth (D), default optimal is 3')
    parser.add_argument('--pcc_kernel', type=int, default=3, help='PCC Kernel Size (k), default optimal is 3')
    parser.add_argument('--polyphony_factor', type=int, default=2,
                        help='Polyphony factor (K), theoretically motivated as 2')
    parser.add_argument('--n_bases', type=int, default=4, help='Number of wavelet bases used for WFE initialization')
    parser.add_argument('--bottleneck_dim', type=int, default=64, help='Dimension of the PCC bottleneck space')


    parser.add_argument('--lambda_recon', type=float, default=0.1,
                        help='Reconstruction weight (\lambda) for Dual-Objective Loss')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')


    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--seed', type=int, default=2026, help='random seed for reproducibility')

    args = parser.parse_args()

    fix_seed(args.seed)
    print(f"Set random seed to {args.seed} for strict reproducibility.")

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu:
        print(f"Use GPU: cuda:{args.gpu}")
    else:
        print("Use CPU")


    if args.is_training:
        print(f">>>>>>> Start training : {args.dataset}_{args.task_name} >>>>>>>")


        print(f">>>>>>> Start testing : {args.dataset}_{args.task_name} >>>>>>>")

    else:
        print(f">>>>>>> Start testing (Inference Only) : {args.dataset}_{args.task_name} >>>>>>>")


if __name__ == '__main__':
    main()