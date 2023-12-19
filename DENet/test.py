import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
from tensorboardX import SummaryWriter
import core.metrics as Metrics
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/denet.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)

    # model
    diffusion = Model.create_model(opt)

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']



    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    avg_psnr = 0.0
    avg_ssim = 0.0
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _, val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        filename = val_data['filename']
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()

        hr_img = Metrics.tensor2img(visuals['HR']) 
        lr_img = Metrics.tensor2img(visuals['LR']) 
        fake_img = Metrics.tensor2img(visuals['INF'])  

        sr_img = visuals['SR']  
        Metrics.save_img(
            Metrics.tensor2img(sr_img[-1]), '{}/{}.png'.format(result_path, str(filename[0][:-4])))
    print("Test Over!")
