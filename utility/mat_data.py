"""generate testing mat dataset"""
import os
import numpy as np
import h5py
from os.path import join, exists
from scipy.io import loadmat, savemat

from util import crop_center,crop_custom, Visualize3D, minmax_normalize, rand_crop,BandMinMaxQuantileStateful
from PIL import Image
import torch
from spectral import *
from skimage import io
import cv2 

data_path = '/cvlabdata1/ly/hsi_data/kaist/ori_dataset/'  #change to datadir

def hdr_to_mat():
    imgpath = data_path+'/Hyperspectral_Project/Apex/APEX_OSD_V1_calibr_cube'
    
    img = open_image(imgpath+'.hdr')
    img = img.load()
    img = img.transpose((2,0,1))

    apex210 = img[:210]
    print('load hdr image and save as mat from ',img.shape, ' to ', apex210.shape)
   # savemat(data_path+"Hyperspectral_Project/apex_210.mat", {'data': apex210})

def create_big_apex_dataset():
    #hdr_to_mat()  #process the hdr file
    total_num = 20
    print('processing---')
    all_data = loadmat(data_path+'Hyperspectral_Project/apex_210.mat')['data']
    print(all_data.shape)
    save_dir = data_path+'Hyperspectral_Project/apex_crop/'
    for i in range(total_num):
        data = rand_crop(all_data, 512, 512)
        savemat(save_dir+str(i)+'.mat',{'data': data})
        print(i)

def create_kaist_dataset(datadir, fnames, newdir, book, matkey,noisykey,func=None, load=h5py.File):
    if not exists(newdir):
        os.mkdir(newdir)

    for i, fn in enumerate(fnames):
        print('generate data(%d/%d)' %(i+1, len(fnames)))
        filepath = join(datadir, fn)
        try:
            mat = loadmat(filepath)
            fns = fn[:-4] + book + fn[-4:]

            data = mat[matkey][...]
            noisy = mat[noisykey][...]

            data_tmp = data.transpose((2,1,0))
            noisy_tmp = noisy.transpose((2,1,0))

            data = func(data_tmp)
            noisy = func(noisy_tmp)

            data_tmp = data.transpose((2,1,0))
            noisy_tmp = noisy.transpose((2,1,0))
            # data_hwc = data.transpose((2,1,0))
            # noisy = noisy.transpose((2,1,0))
            savemat(join(newdir, fns), {'gt': data_tmp ,'input' : noisy_tmp})
            # try:
            #     Image.fromarray(np.array(data_hwc*255,np.uint8)[:,:,20]).save(data_path+'icvl_test_512_png/{}.png'.format(os.path.splitext(fn)[0]))
            # except Exception as e:
            #     print(e)
        except:
            print('open error for {}'.format(filepath))
            continue

def create_mat_dataset(datadir, fnames, newdir, matkey, func=None, load=h5py.File):
    if not exists(newdir):
        os.mkdir(newdir)

    for i, fn in enumerate(fnames):
        print('generate data(%d/%d)' %(i+1, len(fnames)))
        filepath = join(datadir, fn)
        try:
            mat = load(filepath)
            data = func(mat[matkey][...])
            data_hwc = data.transpose((2,1,0))
            savemat(join(newdir, fn), {'data': data_hwc})
            # try:
            #     Image.fromarray(np.array(data_hwc*255,np.uint8)[:,:,20]).save(data_path+'icvl_test_512_png/{}.png'.format(os.path.splitext(fn)[0]))
            # except Exception as e:
            #     print(e)
        except:
            print('open error for {}'.format(fn))
            continue

def create_WDC_dataset():
    imgpath = data_path+'Hyperspectral_Project/dc.tif'
    imggt = io.imread(imgpath)
    # 转为mat
    imggt = torch.tensor(imggt, dtype=torch.float)
    test = imggt[:, 600:800, 50:250].clone()
    train_0 = imggt[:, :600, :].clone()
    train_1 = imggt[:, 800:, :].clone()
    val = imggt[:, 600:656, 251:].clone()

    normalizer = BandMinMaxQuantileStateful()

    # fit train
    normalizer.fit([train_0, train_1])
    train_0 = normalizer.transform(train_0).cpu().numpy()
    train_1 = normalizer.transform(train_1).cpu().numpy()

    # fit test
    normalizer.fit([test])
    test = normalizer.transform(test).cpu().numpy()

    # val test
    normalizer.fit([val])
    val = normalizer.transform(val).cpu().numpy()

    savemat("WDC/train/train_0.mat", {'data': train_0})
    savemat("WDC/train/train_1.mat", {'data': train_1})
    savemat("WDC/test/test.mat", {'data': test})
    savemat("WDC/val/val.mat", {'data': val})

def create_icvl_sr():
    basedir = data_path
    datadir = join(basedir, 'ori_test_mat')
    newdir = join(basedir, 'icvl_test_512')
    fnames = os.listdir(datadir)
    
    def func(data):
        data = np.rot90(data, k=-1, axes=(1,2))
        
        data = crop_center(data, 512, 512)
        
        data = minmax_normalize(data)
        return data
    
    create_mat_dataset(datadir, fnames, newdir, 'rad', func=func)

def create_kaist_2048():
    basedir = data_path
    datadir = join(basedir, 'kaist_test_mat')
    newdir = join(basedir, 'kaist_test_2048')
    fnames = os.listdir(datadir)
    
    def func(data):
        # data = np.rot90(data, k=-1, axes=(1,2))
        
        data = crop_center(data, 2048, 2048)
        
        data = minmax_normalize(data)
        return data
    
    create_mat_dataset(datadir, fnames, newdir, 'data', func=func)

def create_kaist_1024_4():

    basedir = data_path
    datadir = join(basedir, 'test_noisy_2048')
    newdir = join(basedir, 'test_noisy_2048_4')

    dirs = [[-1024,-1024],[1024,-1024],[-1024,1024],[1024,1024]]
    # NorthWest ,NorthEast ,SouthWest ,SouthEast
    books = ['NW','NE','SW','SE']

    for i in range(4):
        create_kaist_1024(dirs[i],books[i],datadir,newdir)
    
def montage_kaist_2048():
    
    oridir = join(data_path, 'test_noisy_2048')
    datadir = join(data_path, 'kaist_test_1024_4_complex/512_noniid')
    newdir = join(data_path,'montage_kaist_2048')

    if not exists(newdir):
        os.mkdir(newdir)

    book = ['NW','NE','SW','SE']
    result_gt = np.zeros([2048,2048,31], dtype=np.float64)
    result_noise = np.zeros([2048,2048,31], dtype=np.float64)
    fnames = os.listdir(oridir)

    for j,fn in enumerate(fnames):
        img = loadmat(join(datadir,tmpdata))
        for i in range(4):
            tmpdata =  fn[:-4] + book[i] + fn[-4:]
            gt_data = img['gt'][...]
            noise_data = img['input'][...]
            if book[i] == 'NW':
                result_gt[0:1024,0:1024,:] = gt_data
                result_noise[0:1024,0:1024,:] = noise_data
            elif book[i] == 'NE':
                result_gt[0:1024,1024:2048,:] = gt_data
                result_noise[0:1024,1024:2048,:] = noise_data
            elif book[i] == 'SW':
                result_gt[1024:2048,0:1024,:] = gt_data
                result_noise[1024:2048,0:1024,:] = noise_data
            elif book[i] == 'SE':
                result_gt[1024:2048,1024:2048,:] = gt_data
                result_noise[1024:2048,1024:2048,:] = noise_data
        savemat(join(newdir, fn), {'gt': result_gt ,'input' : result_noise})

def resolve_kaist_2048():
        
    datadir = join(data_path, 'kaist_2048_complex/512_impulse')
    newdir = join(data_path, 'kaist_1024_complex/1024_impulse')

    if not exists(newdir):
        os.mkdir(newdir)

    book = ['NW','NE','SW','SE']

    fnames = os.listdir(datadir)

    for j,fn in enumerate(fnames):
        print('generate data(%d/%d)' %(j+1, len(fnames)))
        img = loadmat(join(datadir,fn))
        gt_data = img['gt'][...]
        noise_data = img['input'][...]
        for i in range(4):
            tmpdata =  fn[:-4] + book[i] + fn[-4:]
            if book[i] == 'NW':
                result_gt = gt_data[0:1024,0:1024,:]
                result_noise = noise_data[0:1024,0:1024,:]
                savemat(join(newdir, tmpdata), {'gt': result_gt ,'input' : result_noise})
            elif book[i] == 'NE':
                result_gt = gt_data[0:1024,1024:2048,:]
                result_noise = noise_data[0:1024,1024:2048,:]
                savemat(join(newdir, tmpdata), {'gt': result_gt ,'input' : result_noise})
            elif book[i] == 'SW':
                result_gt = gt_data[1024:2048,0:1024,:]
                result_noise = noise_data[1024:2048,0:1024,:]
                savemat(join(newdir, tmpdata), {'gt': result_gt ,'input' : result_noise})
            elif book[i] == 'SE':
                result_gt = gt_data[1024:2048,1024:2048,:]
                result_noise = noise_data[1024:2048,1024:2048,:]
                savemat(join(newdir, tmpdata), {'gt': result_gt ,'input' : result_noise})

def create_kaist_1024(dir,book,datadir,newdir):

    fnames = os.listdir(datadir)
    
    def func(data):
        # data = np.rot90(data, k=-1, axes=(1,2))
        
        data = crop_custom(data, 1024, 1024 , dir)
        
        data = minmax_normalize(data)
        return data
    
    create_kaist_dataset(datadir, fnames, newdir, book, 'gt','input', func=func)


def create_Urban_test():
    imgpath = '/data/HSI_Data/Hyperspectral_Project/Urban_F210.mat'
    img = loadmat(imgpath)
    print(img.keys())
    imgg  = img['Y'].reshape((210,307,307))
    
    imggt = imgg.astype(np.float32)
    print(imggt.shape)
    norm_gt = imggt.transpose((1,2,0))
    cut_gt = norm_gt[:304,:304,:]
    cut_gt = minmax_normalize(cut_gt)
    savemat("/data/HSI_Data/Hyperspectral_Project/Urban_304.mat", {'gt': cut_gt})

def create_WDC_test():
    imgpath = '/data1/jiahua/test_data/wdc_mat/dc.mat'
    img = loadmat(imgpath)
    # print(img.keys())
    imgg  = img['DC']
    # for i in range(161):
    #     test = imgg[i:i+31, 572:828, 22:278]
    #     test = minmax_normalize(test)
    #     test = test.transpose((1,2,0))
    #     # norm_gt = imggt.transpose((1,2,0))
    #     # cut_gt = norm_gt[:304,:304,:]
    #     # cut_gt = minmax_normalize(cut_gt)
    #     savemat("/data1/jiahua/ly/wdc_test/wdc_test"+ str(i) + ".mat", {'data': test})
    #     print(i)
    
    test = imgg[:, 572:828, 22:278]
    test = minmax_normalize(test)
    test = test.transpose((1,2,0))
    savemat("/data1/jiahua/ly/wdc/wdc.mat", {'data': test})

def create_pavia_test():
    imgpath = '/data1/jiahua/test_data/pavia_mat/Pavia.mat'
    img = loadmat(imgpath)
    # print(img.keys())
    imgg  = img['pavia']

    # for i in range(71):
    #     test = imgg[356:740, 115:499, i:i+31]
    #     # test = test.transpose((2,1,0))
    #     test = minmax_normalize(test)
    #     # norm_gt = imggt.transpose((1,2,0))
    #     # cut_gt = norm_gt[:304,:304,:]
    #     # cut_gt = minmax_normalize(cut_gt)
    #     savemat("/data1/jiahua/ly/pavia_test/pavia_test"+ str(i) + ".mat", {'data': test})
    #     print(i)

    test = imgg[:, :,:]

    test = minmax_normalize(test)
    test = test * 255
    cv2.imwrite('test.png', test[:,:,1:4])
    savemat("/data1/jiahua/ly/pavia/pavia.mat", {'data': test})

def create_india_test():
    imgpath = '/data1/jiahua/test_data/indian_mat/indian_pines.mat'
    img = loadmat(imgpath)
    print(img.keys())
    imgg  = img['pavia']
    # for i in range(169):
    #     test = imgg[8:136:,8:136,i:i+31]
    #     # test = test.transpose((2,1,0))
    #     test = minmax_normalize(test)
    #     # norm_gt = imggt.transpose((1,2,0))
    #     # cut_gt = norm_gt[:304,:304,:]
    #     # cut_gt = minmax_normalize(cut_gt)
    #     savemat("/data1/jiahua/ly/india_test/india_test"+ str(i) + ".mat", {'data': test})
    #     print(i)
    test = imgg[8:136:,8:136,:]
    # test = minmax_normalize(imgg)
    savemat("/data1/jiahua/ly/india/india.mat", {'data': test})

    test = loadmat("/data1/jiahua/ly/india/india.mat")
    print('')

def create_urban_test():
    # 228,228,210
    imgpath = '/data1/jiahua/test_data/urban_mat/noisy.mat'
    img = loadmat(imgpath)
    print(img.keys())
    imgg  = img['urban']
    # for i in range(169):
    #     test = imgg[8:136:,8:136,i:i+31]
    #     # test = test.transpose((2,1,0))
    #     test = minmax_normalize(test)
    #     # norm_gt = imggt.transpose((1,2,0))
    #     # cut_gt = norm_gt[:304,:304,:]
    #     # cut_gt = minmax_normalize(cut_gt)
    #     savemat("/data1/jiahua/ly/india_test/india_test"+ str(i) + ".mat", {'data': test})
    #     print(i)
    test = imgg[0:224:,0:224,:]
    # test = minmax_normalize(imgg)
    savemat("/data1/jiahua/ly/urban/urban_224.mat", {'data': test})

    test = loadmat("/data1/jiahua/ly/urban_test/urban_179_210.mat")
    print('')


if __name__ == '__main__':
    #create_big_apex_dataset()
    # create_icvl_sr()
    # create_kaist_2048()
    # create_kaist_1024_4()
    # montage_kaist_2048()
    # resolve_kaist_2048()
    #create_WDC_dataset()
    #create_Urban_test()
    #hdr_to_mat()
    # create_WDC_test()
    # create_pavia_test()
    # create_india_test()
    create_urban_test()

