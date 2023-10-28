#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-22 22:07:08
from warnings import warn

import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from skimage.util import crop
from skimage.util.dtype import dtype_range
from torch.autograd import Function as autoF
from scipy.special import gammaln
from skimage import img_as_ubyte
import numpy as np
import sys
from math import floor
import numpy as np
import torch
from functools import partial

import scipy.io as sio
import skimage.io

import cv2
import numpy as np

def compare_ssim(X, Y, win_size=None, gradient=False,
                 data_range=None, multichannel=False, gaussian_weights=False,
                 full=False, **kwargs):
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if multichannel:
        # loop over channels
        args = dict(win_size=win_size,
                    gradient=gradient,
                    data_range=data_range,
                    multichannel=False,
                    gaussian_weights=gaussian_weights,
                    full=full)
        args.update(kwargs)
        nch = X.shape[-1]
        mssim = np.empty(nch)
        if gradient:
            G = np.empty(X.shape)
        if full:
            S = np.empty(X.shape)
        for ch in range(nch):
            ch_result = compare_ssim(X[..., ch], Y[..., ch], **args)
            if gradient and full:
                mssim[..., ch], G[..., ch], S[..., ch] = ch_result
            elif gradient:
                mssim[..., ch], G[..., ch] = ch_result
            elif full:
                mssim[..., ch], S[..., ch] = ch_result
            else:
                mssim[..., ch] = ch_result
        mssim = mssim.mean()
        if gradient and full:
            return mssim, G, S
        elif gradient:
            return mssim, G
        elif full:
            return mssim, S
        else:
            return mssim

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if gaussian_weights:
        # Set to give an 11-tap filter with the default sigma of 1.5 to match
        # Wang et. al. 2004.
        truncate = 3.5

    if win_size is None:
        if gaussian_weights:
            # set win_size used by crop to match the filter size
            r = int(truncate * sigma + 0.5)  # radius as in ndimage
            win_size = 2 * r + 1
        else:
            win_size = 7  # backwards compatibility

    if np.any((np.asarray(X.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent.  If the input is a multichannel "
            "(color) image, set multichannel=True.")

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        if X.dtype != Y.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "X.dtype.")
        dmin, dmax = dtype_range[X.dtype.type]
        data_range = dmax - dmin

    ndim = X.ndim

    if gaussian_weights:
        filter_func = gaussian_filter
        filter_args = {'sigma': sigma, 'truncate': truncate}
    else:
        filter_func = uniform_filter
        filter_args = {'size': win_size}

    # ndimage filters need floating point data
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(X * X, **filter_args)
    uyy = filter_func(Y * Y, **filter_args)
    uxy = filter_func(X * Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim
    mssim = crop(S, pad).mean()

    if gradient:
        # The following is Eqs. 7-8 of Avanaki 2009.
        grad = filter_func(A1 / D, **filter_args) * X
        grad += filter_func(-S / B2, **filter_args) * Y
        grad += filter_func((ux * (A2 - A1) - uy * (B2 - B1) * S) / D,
                            **filter_args)
        grad *= (2 / X.size)

        if full:
            return mssim, grad, S
        else:
            return mssim, grad
    else:
        if full:
            return mssim, S
        else:
            return mssim
def ssim_index(im1, im2):
    '''
    Input:
        im1, im2: np.uint8 format
    '''
    if im1.ndim == 2:
        out = compare_ssim(im1, im2, data_range=1, gaussian_weights=True,
                                                    use_sample_covariance=False, multichannel=False)
    elif im1.ndim == 3:
        out = compare_ssim(im1, im2, data_range=1, gaussian_weights=True,
                                                 use_sample_covariance=False, multichannel=True)
    elif im1.ndim==4:
        out = compare_ssim(im1, im2,  gaussian_weights=True,
                           use_sample_covariance=False, multichannel=True)
    else:
        sys.exit('Please input the corrected images')
    return out

def im2patch(im, pch_size, stride=1):
    '''
    Transform image to patches.
    Input:
        im: 3 x H x W or 1 X H x W image, numpy format
        pch_size: (int, int) tuple or integer
        stride: (int, int) tuple or integer
    '''
    if isinstance(pch_size, tuple):
        pch_H, pch_W = pch_size
    elif isinstance(pch_size, int):
        pch_H = pch_W = pch_size
    else:
        sys.exit('The input of pch_size must be a integer or a int tuple!')

    if isinstance(stride, tuple):
        stride_H, stride_W = stride
    elif isinstance(stride, int):
        stride_H = stride_W = stride
    else:
        sys.exit('The input of stride must be a integer or a int tuple!')

    C, H, W = im.shape
    num_H = len(range(0, H-pch_H+1, stride_H))
    num_W = len(range(0, W-pch_W+1, stride_W))
    num_pch = num_H * num_W
    pch = np.zeros((C, pch_H*pch_W, num_pch), dtype=im.dtype)
    kk = 0
    for ii in range(pch_H):
        for jj in range(pch_W):
            temp = im[:, ii:H-pch_H+ii+1:stride_H, jj:W-pch_W+jj+1:stride_W]
            pch[:, kk, :] = temp.reshape((C, num_pch))
            kk += 1

    return pch.reshape((C, pch_H, pch_W, num_pch))

class Bandwise(object):
    def __init__(self, index_fn):
        self.index_fn = index_fn

    def __call__(self, X, Y):
        C = X.shape[-3]
        bwindex = []
        for ch in range(C):
            x = torch.squeeze(X[...,ch,:,:].data).cpu().numpy()
            y = torch.squeeze(Y[...,ch,:,:].data).cpu().numpy()
            index = self.index_fn(x, y)
            bwindex.append(index)
        return bwindex

cal_bwssim = Bandwise(partial(structural_similarity))
cal_bwpsnr = Bandwise(partial(peak_signal_noise_ratio, data_range=1))
def cal_sam(X, Y, eps=1e-8):
    X = torch.squeeze(X.data).cpu().numpy()
    Y = torch.squeeze(Y.data).cpu().numpy()
    tmp = (np.sum(X*Y, axis=0) + eps) / (np.sqrt(np.sum(X**2, axis=0)) + eps) / (np.sqrt(np.sum(Y**2, axis=0)) + eps)
    tmp = np.clip(tmp, -1, 1)
    result = np.mean(np.real(np.arccos(tmp)))

    return result
def MAD(X,Y):
    X = torch.squeeze(X.data).cpu().numpy()
    Y = torch.squeeze(Y.data).cpu().numpy()
    W=X.shape[-1]
    H=X.shape[-2]
    C=X.shape[-3]
    # num=C*H*W
    diff=np.fabs(X - Y)
    # sum=np.sum(diff)
    # print(sum)
    # print(C*H*W)
    mean_val=np.mean(diff)
    #print(mean_val)
    return mean_val

def MSIQA(X, Y):
    psnr = np.mean(cal_bwpsnr(X, Y))
    ssim = np.mean(Bandwise(ssim_index)(X, Y))
    sam = cal_sam(X, Y)
    mad=MAD(X,Y)
    return psnr, ssim, sam,mad

def MSIQA_Batch(X, Y):
    psnr = np.mean(cal_bwpsnr(X, Y))
    ssim = np.mean(Bandwise(ssim_index)(X, Y))
    # ssim = np.mean(cal_bwssim(X, Y))
    sam = cal_sam(X, Y)
    mad=MAD(X,Y)
    return psnr, ssim, sam,mad




def batch_PSNR(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    PSNR = 0
    print(Img.shape)
    print(imclean.shape)
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:,:], Img[i,:,:,:,:])
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    SSIM = 0
    print(Img.shape)
    print(Iclean.shape)
    for i in range(Img.shape[0]):
        print("评测中去噪后图像的形状{0}".format(Iclean[i,: ,:, :, :].shape))
        SSIM += ssim_index(np.transpose(Iclean[i,:,:,:,:],(1,2,3,0)), np.transpose(Img[i,:,:,:,:],(1,2,3,0)))
    return (SSIM/Img.shape[0])

