import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str,
                        help="name of the model")
    parser.add_argument("--seed", help="seed for RNG", type=int, default=0)

    # parameters
    parser.add_argument("--nb_samp", type=int, default=16000, help="Number of input samples")
    parser.add_argument("--first_conv", type=int, default=1024, help="Number of filter coefficients")
    parser.add_argument("--in_channels", type=int, default=1, help="Input channels")
    parser.add_argument("--filts", type=int, default=[20, [20, 20], [20, 128], [128, 128]], help="Number of filters channel in residual blocks")
    parser.add_argument("--blocks", type=int, default=[2, 4], help="")
    parser.add_argument("--nb_fc_node", type=int, default=1024, help="Fully connected nodes")
    parser.add_argument("--gru_node", type=int, default=1024, help="GRU nodes")
    parser.add_argument("--nb_gru_layer", type=int, default=3, help="GRU layers")
    parser.add_argument("--nb_classes", type=int, default=2, help="Output classes")

    parser.add_argument("--win_len", type=int, required=False, default=2, help="Window length")
    parser.add_argument("--runpath", type=str, required=False, default="/nas/home/aorsatti/Pycharm/Tesi/data/trained_models", help="Results directory to be loaded")
    parser.add_argument("--gpu", type=int, required=False, default=-1, help="Index of GPU to use (None for CPU, -1 for least used GPU)")
    parser.add_argument("--outpath", type=str, required=False, default="/nas/home/aorsatti/Pycharm/Tesi/data/trained_models", help="Results directory")
    parser.add_argument("--batch_size", type=int, required=False, default=64, choices=[1, 16, 32, 64, 128, 256, 512], help="Batch size")
    parser.add_argument("--epochs", type=int, required=False, default=50, help="Max iterations number")
    parser.add_argument("--label_smoothing", type=float, required=False, default=0.0, help="Cross Entropy Loss label smoothing")
    parser.add_argument("--lr", type=float, required=False, default=0.0001, help="")

    return parser.parse_args()