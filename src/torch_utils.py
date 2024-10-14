
import os
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

import torch
import GPUtil

import copy
import json
from argparse import Namespace
from pathlib import Path
from typing import Union


def read_args(filename: Union[str, Path]) -> Namespace:
    """Read a file containing a dict of arguments, and store them in a Namespace"""
    args = Namespace()
    with open(filename, 'r') as fp:
        args.__dict__.update(json.load(fp))
    return args


def write_args(filename: Union[str, Path], args: Namespace, indent: int = 2, excluded_keys=[]) -> None:
    """
    Write a Namespace arguments to a file.
    """
    args = copy.deepcopy(args)
    if len(excluded_keys) != 0:
        for key in excluded_keys:
            if key in args:
                args.__delattr__(key)
            else:
                print(f"{key} not in args, skipping")
                pass

    with open(filename, 'w') as fp:
        json.dump(args.__dict__, fp, indent=indent)


def torch_on_cuda():
    return os.environ["CUDA_VISIBLE_DEVICES"] and torch.cuda.is_available()


def set_gpu(id=-1):
    """
    Set tensor computation device.

    :param id: CPU or GPU device id (None for CPU, -1 for the device with lowest memory usage, or the ID)

    hint: use gpustat (pip install gpustat) in a bash CLI, or gputil (pip install gputil) in python.

    """
    if id is None:
        # CPU only
        print('GPU not selected')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
    else:
        # -1 for automatic choice
        device = id if id != -1 else GPUtil.getFirstAvailable(order='memory')[0]
        try:
            name = GPUtil.getGPUs()[device].name
        except IndexError:
            print('The selected GPU does not exist. Switching to the most available one.')
            device = GPUtil.getFirstAvailable(order='memory')[0]
            name = GPUtil.getGPUs()[device].name
        print('GPU selected: %d - %s' % (device, name))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)


def set_backend():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def dtype():
    if torch_on_cuda():
        return torch.cuda.FloatTensor
    else:
        return torch.FloatTensor


def platform():
    if torch_on_cuda():
        # watch out! cuda for torch is 0 because it is the first torch can see! It is not the os.environ one!
        device = "cuda:0"
    else:
        device = "cpu"
    return torch.device(device)


def load_weights(model: torch.nn.Module, weights_path: str) -> torch.nn.Module:
    state_tmp = torch.load(weights_path, map_location='cpu')
    if 'net' not in state_tmp.keys():
        state = OrderedDict({'net': OrderedDict()})
        [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
    else:
        state = state_tmp
    
    incomp_keys = model.load_state_dict(state['net'], strict=True)
    print(incomp_keys)
    
    return model


def save_model(net: torch.nn.Module, optimizer: torch.optim.Optimizer,
               train_loss: float, valid_loss: float,
               train_score: float, valid_score: float,
               batch_size: int, epoch: int, path: str):
    path = str(path)
    state = dict(net=net.state_dict(),
                 opt=optimizer.state_dict(),
                 train_loss=train_loss,
                 valid_loss=valid_loss,
                 train_score=train_score,
                 valid_score=valid_score,
                 batch_size=batch_size,
                 epoch=epoch)
    torch.save(state, path)

