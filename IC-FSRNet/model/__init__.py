import os
import torch
from importlib import import_module
import torch.nn as nn
def init_model(model, args):
    device = torch.device(args.device)
    if len(args.cuda_name)>1:
        model = nn.DataParallel(model).to(device)
    else:
        model.to(device)
    return model

def get_model(args):
    module = import_module('model.' + args.model.lower())
    return init_model(module.make_model(args), args)
