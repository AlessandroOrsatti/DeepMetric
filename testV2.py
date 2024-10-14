import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import utils
from sklearn.metrics import roc_auc_score, roc_curve, auc, balanced_accuracy_score
import model as rawnet
#import yaml
from metrics import *
from loops import test_loop
import src
from argparser import parse_args
import parameters as param
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def main(args):

    model_path = os.path.join(param.models_dir, "rawnet_log_2s_03.pth")
    results_path = os.path.join(param.results_dir, "results_new_dataset.csv")
    emb_path = os.path.join(param.embeddings_dir, "embeddings_new_dataset")

    os.makedirs(emb_path, exist_ok=True)

    print(model_path)
    print(results_path)
    print(emb_path)
    
    # select the computation device:
    src.torch_utils.set_gpu(-1)

    # set backend here to create GPU processes
    src.torch_utils.set_backend()
    src.torch_utils.set_seed()
    
    # define the computation platform for torch:
    platform = src.torch_utils.platform()

    # Load test set, initialize Dataloaders
    targets_test, file_test, path_test, original_test = utils.genDataFrame(
        os.path.join(param.csv_dir, 'modified_dataset2.csv'),
        is_eval=True, is_mod=True)
    test_set1 = utils.LoadEvalData(file_IDs=file_test[0:int(len(file_test)/2)], labels=targets_test, audio_path=path_test, win_len=args.win_len)
    test_set2 = utils.LoadEvalData(file_IDs=file_test[int(len(file_test)/2):], labels=targets_test, audio_path=path_test, win_len=args.win_len)
    test_dataloader1 = DataLoader(test_set1, batch_size=1, shuffle=True, drop_last=True, num_workers=2)
    test_dataloader2 = DataLoader(test_set2, batch_size=1, shuffle=True, drop_last=True, num_workers=2)

    # Initialize model
    # RawNet config with yaml file
    # dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'
    # with open(dir_yaml, 'r') as f_yaml:
    #     parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)

    # Initialize model
    model = rawnet.RawNet(args, platform)
    # Move the model on gpu
    model = model.to(platform)
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    # define the optimization parameters
    loss_fn = nn.CrossEntropyLoss()

    df = pd.DataFrame(columns=param.test_dataframe)
    #df = pd.read_csv(results_path)

    accuracy1, test_loss1, df = test_loop(test_dataloader1, model, loss_fn, platform, df, original_test, emb_dir=emb_path)
    df.to_csv(results_path, index= False)

    accuracy2, test_loss2, df = test_loop(test_dataloader2, model, loss_fn, platform, df, original_test, emb_dir=emb_path)
    df.to_csv(results_path, index= False)

    df = addCosineDistance(df, emb_dir=emb_path)
    df.to_csv(results_path, index=False)

    df = addEuclideanDistance(df, emb_dir=emb_path)
    df.to_csv(results_path, index= False)

    df = addManhattanDistance(df, emb_dir=emb_path)
    df.to_csv(results_path, index=False)

    #df = addDotProduct(df, emb_dir=emb_path)
    #df.to_csv(results_path, index=False)

    #df = addSquaredEuclideanDistance(df,emb_dir=emb_path)
    #df.to_csv(results_path, index=False)

    print("Done")

    df_filtered = df[df['original']=='-']
    plt.figure(figsize=(8, 8))
    utils.plot_roc_curve(df_filtered['label'], df_filtered['prediction'], legend='Original score')
    utils.plot_roc_curve(df_filtered['label'], -df_filtered['cosine distance'], legend='Cosine distance')
    utils.plot_roc_curve(df_filtered['label'], -df_filtered['euclidean distance'], legend='Euclidean distance')
    utils.plot_roc_curve(df_filtered['label'], -df_filtered['manhattan distance'], legend='Manhattan distance')
    plt.title(f"ROC curve for new dataset")
    plt.savefig('roc4.pdf')
    plt.show()

if __name__ == '__main__':
    args = parse_args()
    main(args)


