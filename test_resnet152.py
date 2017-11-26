#!python2
#encoding=utf-8
'''
Created on Nov 6, 2017

@author: kohill


Test a model wheather it works.
'''
import os
os.environ.setdefault("MXNET_CUDNN_AUTOTUNE_DEFAULT", "0")
import mxnet as mx
import numpy as np
import cv2,os,math
import matplotlib.pyplot as plt

test_origin_openpose = False


save_prefix = "/data1/yks/models/openpose/realtimePose" if test_origin_openpose else "model/vggpose"
test_img_filename = "sample_image/multiperson.jpg"
test_images_path = "/data1/yks/dataset/ai_challenger/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911/"
epoch = 0 if test_origin_openpose else 2300
batch_size = 1
from modelCPM import *
import modelresnet
from config.config import config
from log import log_init
import logging
log_init("vgg.log")
import mxnet as mx
from pprint import pprint
def show_heatmap(img_path,heatmaps):
    img = cv2.imread(img_path)
    img = copy.copy(img[:,:,(2,1,0)])

    fig, axes = plt.subplots(5, 7, figsize=(20, 16),
                         subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.01, wspace=0.05)
    print (heatmaps.shape)
    for i in range(5):
        for j in range(7):
            if i*7 +j < heatmaps.shape[2]:
                axes[i][j].imshow(heatmaps[:,:,i*7 + j])
    axes[-1][-1].imshow(img)
    axes[-2][-1].imshow(np.max(np.array(heatmaps[0:-2]),axis = 2))
    plt.show()
    pass
def show_pafmap(img_path,pafmaps):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(368,368))
    img = copy.copy(img[:,:,(2,1,0)])

    fig, axes = plt.subplots(5, 5, figsize=(20, 20),
                         subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.01, wspace=0.05)        
    pafmaps_s = []
    print(pafmaps.shape)
    for i in range(5):
        for j in range(5):
            if i*5 +j < pafmaps.shape[2]//2:
                pafmaps_s.append(np.sqrt(pafmaps[:,:,(i*5 + j)*2] **2+pafmaps[:,:,(i*5 + j)*2 +1] **2))
                axes[i][j].imshow(pafmaps_s[-1])
    axes[-1][-1].imshow(img)
    axes[-1][-2].imshow(np.max(np.array(pafmaps_s),axis =0))
    plt.show()
def getHeatAndPAF(cmodel,img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(368,368))
#         img = padimg(img,368)
    imgs_transpose = np.transpose(np.float32(img[:,:,:]), (2,0,1)) /256 - 0.5 if test_origin_openpose else np.transpose(np.float32(img[:,:,:]), (2,0,1))
    imgs_batch = mx.io.DataBatch([mx.nd.array([imgs_transpose[:,:,:]])])
    cmodel.forward(imgs_batch)


    result = cmodel.get_outputs() 
    for r in result:
        print r.shape
       
    heatmap = np.moveaxis(result[1].asnumpy()[0], 0, -1)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)   
    
    pafmap = np.moveaxis(result[0].asnumpy()[0], 0, -1)
    pafmap = cv2.resize(pafmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)    
    
        
    return img_path,img,heatmap,pafmap
def get_module(prefix = save_prefix,
               batch_size = 1, 
               start_epoch = epoch,
               reinit = False,
               gpus = [0]):
    if test_origin_openpose:
        sym = poseSymbol_test()
    else:
        from symbol.resnet_v1_101_deeplab import resnet_v1_101_deeplab
        Sym = resnet_v1_101_deeplab()
        sym = Sym.get_symbol(num_classes=14,is_train  = False)
    _,args,auxes = mx.model.load_checkpoint(prefix , epoch=start_epoch)
    model = mx.mod.Module(symbol=sym, context=[mx.gpu(g) for g in gpus],
                                  label_names=['heatmaplabel',
                                 'partaffinityglabel',
                                 'heatweight',
                                 'vecweight']
                          )
    model.bind(data_shapes=[('data', (batch_size, 3, 368, 368))],)
    model.init_params(arg_params=args, aux_params=auxes, allow_missing=False,allow_extra = False)
    return model

if __name__ == "__main__":
    mod = get_module()
    
    for x,y,z in os.walk(test_images_path):
        for name in z:
            img_path,img,heatmaps,paf = getHeatAndPAF(mod,os.path.join(x,name))
            show_heatmap(img_path, heatmaps)
            show_pafmap(img_path,paf)
    pass
