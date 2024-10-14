import os
import numpy as np
import pandas
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
import utils
import src

import model as rawnet
import parameters as param
from loops import train_loop, val_loop, trainTriplet_loop
from argparser import parse_args

def main(args):

    # save args to outpath, for reproducibility
    os.makedirs(args.outpath, exist_ok=True)  # set to True to enable overwriting
    src.torch_utils.write_args(filename=os.path.join(args.outpath, "args.txt"),
                   args=args)
    model_path = os.path.join(param.models_dir, "rawnet_log_2s_03.pth")
    history_path = os.path.join(args.outpath, "history.csv")
    
    # select the computation device:
    src.torch_utils.set_gpu(args.gpu)

    # set backend here to create GPU processes
    src.torch_utils.set_backend()
    src.torch_utils.set_seed()
    
    # define the computation platform for torch:
    platform = src.torch_utils.platform()

    # Load training and validation set, initialize Dataloaders
    targets_trn, file_train, path_trn, _ = utils.genDataFrame(os.path.join(param.dataset_csv_path, 'ASVspoof2019.LA.cm.train.trn.txt'), is_train=True)
    targets_dev, file_dev, path_dev, _ = utils.genDataFrame(os.path.join(param.dataset_csv_path, 'ASVspoof2019.LA.cm.dev.trl.txt'), is_train=True)

    train_set = utils.LoadTrainData(file_IDs=file_train, labels=targets_trn, audio_path=path_trn, win_len=args.win_len)
    dev_set = utils.LoadTrainData(file_IDs=file_dev, labels=targets_dev, audio_path=path_dev, win_len=args.win_len)

    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    dev_dataloader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)

    del train_set, targets_trn
    del dev_set, targets_dev

    # RawNet config
    # dir_yaml = os.path.splitext('model_config_RawNet')[0] + '.yaml'
    # with open(dir_yaml, 'r') as f_yaml:
    #     parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)

    # Initialize model
    model = rawnet.RawNet(args, platform)
    # Move the model on gpu either with
    model = model.to(platform)
    
    # define the optimization parameters
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Scheduler for reducing the learning rate on plateau
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, mode='min', verbose=True)


    no_improvement = 0     # n of epochs with no improvements
    patience = 40          # max n of epoch with no improvements
    min_val_loss = np.inf
    history = []
    for t in range(args.epochs):

        print(f"Epoch {t+1}\n-------------------------------")
        # Model training
        train_acc, train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, platform)
        val_acc, val_loss = val_loop(dev_dataloader, model, loss_fn, platform)
        
        # SCHEDULER CALLBACK
        lr_scheduler.step(val_loss)    # call the scheduler to reduce the lr if val loss is in plateau
        history.append({"epoch": t,
                        "val_loss": val_loss,
                        "loss": train_loss,
                        "val_score": val_acc,
                        "score": train_acc,
                        "lr": optimizer.param_groups[0]['lr']})
        
        # MODEL CHECKPOINT CALLBACK
        if val_loss < min_val_loss:
        # if val_acc > best_acc:
            # Callback for weight saving
            torch.save(model.state_dict(), model_path)
            no_improvement = 0
            min_val_loss = val_loss
        else:                           # No improvement in the new epoch
            no_improvement += 1
            
        if t > 5 and no_improvement == patience:    # Patience reached
            print(f'Early stopped at epoch {t}')
            # Save history for early stopping
            df = pandas.DataFrame(history)
            df.to_csv(history_path)
            break

    print("Done!")
    # Save history
    df = pandas.DataFrame(history)
    df.to_csv(history_path)
    print("Model saved!")

if __name__ == '__main__':
    args = parse_args()
    main(args)

