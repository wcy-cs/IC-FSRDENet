import os
from option import args
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
import torch
import torch.optim as optim
import torch.nn as nn
import model
from data import dataset
import torchvision
from torch.utils.data import DataLoader

import util
net = model.get_model(args)
testdata = dataset.Data(root=os.path.join(args.dir_data, args.data_test), args=args, train=False)
testset = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=1)
pretrained_dict = torch.load('pretrained_model/pretrained_model.pth')
net.load_state_dict(pretrained_dict)
save_name = "result-test"
os.makedirs(os.path.join(args.save_path, args.writer_name, save_name), exist_ok=True)
with torch.no_grad():
    net.eval()
    for batch, (lr, hr, filename) in enumerate(testset):
        lr, hr = util.prepare(lr), util.prepare(hr)
        sr = net(lr)
        torchvision.utils.save_image(sr[0],
                                 os.path.join(args.save_path, args.writer_name, save_name,
                                              '{}'.format(str(filename[0])[:-4] + ".png")))
    print('Test Over')
