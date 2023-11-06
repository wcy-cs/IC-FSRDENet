from skimage import measure
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import math
import cv2
import os
def prepare(arg):
    if torch.cuda.is_available():
        # print(1)
        arg = arg.cuda(0)
    return arg


def cal_ssim(img1, img2):
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    img1_np = img1_np.transpose(1, 2, 0)
    img2_np = img2_np.transpose(1, 2, 0)
    # print(img1_np)
    return structural_similarity(img1_np, img2_np, data_range=1, multichannel=True)



def convert_rgb_to_y(tensor):
    image = tensor[0].cpu().numpy().transpose(1,2,0)#.detach()

    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    #xform = np.array([[65.481, 128.553, 24.966]])
    #y_image = image.dot(xform.T) + 16.0

    xform = np.array([[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]])
    y_image = image.dot(xform.T) + 16.0

    return y_image


def calc_psnr(sr, hr, scale, rgb_range, dataset=None, facebb=[]):
    # Y channel
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    shave = scale
    facebb = facebb[0].numpy()
    if diff.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        diff = diff.mul(convert).sum(dim=1)

    _, _, w, h = hr.size()
    x1 = max(int(facebb[0]), shave)
    x2 = min(int(facebb[2]), w-shave)
    y1 = max(int(facebb[1]), shave)
    y2 = min(int(facebb[3]), h-shave)

    image1 = convert_rgb_to_y(sr)
    image2 = convert_rgb_to_y(hr)
    image1 = image1[y1:y2, x1:x2, :]
    image2 = image2[y1:y2, x1:x2, :]
    psnr = peak_signal_noise_ratio(image1, image2, data_range=rgb_range)
    ssim = structural_similarity(image1, image2, win_size=11, gaussian_weights=True, multichannel=True, K1=0.01, K2=0.03,
                        sigma=1.5, data_range=rgb_range)
    return psnr, ssim

def rgb2ycbcr(img, only_y=True):
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                                [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def calc_metrics(img1, img2, crop_border=8, test_Y=True):

    img1 = np.transpose(img1, (1, 2, 0))
    img2 = np.transpose(img2, (1, 2, 0))
    img1 = np.array(img1)
    img2 = np.array(img2)
    if test_Y and img1.shape[2] == 3:
        im1_in = rgb2ycbcr(img1)
        im2_in = rgb2ycbcr(img2)
    else:
        im1_in = img1
        im2_in = img2
    if im1_in.ndim == 3:
        cropped_im1 = im1_in[:, crop_border:-crop_border, crop_border:-crop_border]
        cropped_im2 = im2_in[:, crop_border:-crop_border, crop_border:-crop_border]
    elif im1_in.ndim == 2:
        cropped_im1 = im1_in[crop_border:-crop_border, crop_border:-crop_border]
        cropped_im2 = im2_in[crop_border:-crop_border, crop_border:-crop_border]
    else:
        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im1_in.ndim))

    psnr = calc_psnr(cropped_im1 * 255, cropped_im2 * 255)
    ssim = calc_ssim(cropped_im1 * 255, cropped_im2 * 255)

    return psnr, ssim
