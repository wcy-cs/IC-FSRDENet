import os
from option import args
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
import torch
import torch.optim as optim
import torch.nn as nn
import model
from data import dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import util

net = model.get_model(args)
writer = SummaryWriter('./logs/{}'.format(args.writer_name))
traindata = dataset.Data(root=os.path.join(args.dir_data, args.data_train), args=args, train=True)
valdata = dataset.Data(root=os.path.join(args.dir_data, args.data_val), args=args, train=False)
trainset = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=32)
valset = DataLoader(valdata, batch_size=1, shuffle=False, num_workers=1)
criterion1 = nn.L1Loss()
optimizer = optim.Adam(params=net.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-8)
best_val = 0.0
for i in range(args.epochs):
    net.train()
    train_loss = 0
    for batch, (lr,  hr, _) in enumerate(trainset):
        lr, hr = util.prepare(lr), util.prepare(hr)
        sr = net(lr)
        l1_loss = criterion1(sr, hr)
        train_loss = train_loss + l1_loss.item()
        optimizer.zero_grad()
        l1_loss.backward()
        optimizer.step()
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i)
    print("Epoch：{} loss: {:.3f}".format(i+1, train_loss/(len(trainset)) * 255))
    writer.add_scalar('train_loss', train_loss /(len(trainset)) * 255, i)
    os.makedirs(os.path.join(args.save_path, args.writer_name), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, args.writer_name, 'model'), exist_ok=True)

    net.eval()
    val_psnr_dic = 0
    val_ssim_dic = 0
    for batch, (lr,  hr, filename) in enumerate(valset):
        lr, hr = util.prepare(lr), util.prepare(hr)
        sr = net(lr)
        psnr_c, ssim_c = util.calc_metrics(hr[0].data.cpu(), sr[0].data.cpu())
        val_psnr_dic = val_psnr_dic + psnr_c
        val_ssim_dic = val_ssim_dic + ssim_c
    writer.add_scalar("val_psnr_DIC", val_psnr_dic / len(valset), i)
    writer.add_scalar("val_ssim_DIC", val_ssim_dic / len(valset), i)
    if  val_psnr_dic / len(valset) > best_val:
        best_val =  val_psnr_dic / len(valset)
        torch.save(net.state_dict(),
                   os.path.join(args.save_path, args.writer_name, 'model', 'epoch_best_{}.pth'.format(i + 1)))
