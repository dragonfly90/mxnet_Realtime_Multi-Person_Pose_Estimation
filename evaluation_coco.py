## Include mxnet path: you should include your mxnet local path, if mxnet path is global, 
## you don't need to include it.
import sys
sys.path.append('../../practice_demo')
import os
import copy
import re
import json
from collections import namedtuple
from scipy.ndimage.filters import gaussian_filter

import mxnet as mx

import numpy as np
from numpy import ma

from google.protobuf import text_format

import cv2 as cv
import scipy
import PIL.Image
import math
import time
import skimage.io as io

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from GenerateLabelCPM import *
from modelCPM import *


Point = namedtuple('Point', 'x y')
crop_size_x = 368
crop_size_y = 368
center_perterb_max = 40

#use_caffe = True
scale_prob = 1
scale_min = 0.5
scale_max = 1.1
target_dist = 0.6

modelId = 1
# set this part
param = dict()
# GPU device number (doesn't matter for CPU mode)
GPUdeviceNumber = 0
# Select model (default: 5)
param['modelID'] = modelId
# Use click mode or not. If yes (1), you will be asked to click on the center
# of person to be pose-estimated (for multiple people image). If not (0),
# the model will simply be applies on the whole image.
param['click'] = 1
# Scaling paramter: starting and ending ratio of person height to image
# height, and number of scales per octave
# warning: setting too small starting value on non-click mode will take
# large memory
# CPU mode or GPU mode
param['use_gpu'] = 1
param['test_mode'] = 3
param['vis'] = 1
param['octave'] = 6
param['starting_range'] = 0.8
param['ending_range'] = 2
param['min_num'] = 4
param['mid_num'] = 10
# the larger the crop_ratio, the smaller the windowsize
param['crop_ratio'] = 2.5  # 2
param['bbox_ratio'] = 0.25 # 0.5
# applyModel_max
param['max'] = 0
# use average heatmap
param['merge'] = 'avg'

def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h%stride==0) else stride - (h % stride) # down
    pad[3] = 0 if (w%stride==0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1,:,:]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:,0:1,:]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1,:,:]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:,-2:-1,:]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad

class DataBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = [data]
        self.label = [label]
        self.pad = pad

def applyDNN(oriImg, images, sym1, arg_params1, aux_params1):
    
    imageToTest_padded, pad = padRightDownCorner(images, 8, 128)
    transposeImage = np.transpose(np.float32(imageToTest_padded[:,:,:]), (2,0,1))/256 - 0.5
    testimage = transposeImage
    # print testimage.shape
    # cmodel = mx.mod.Module(symbol=sym1, label_names=[])
    # cmodel = mx.mod.Module(symbol=sym1, label_names=[], context=mx.gpu(0))
    cmodel = mx.mod.Module(symbol=sym1, context = mx.gpu(3), label_names=[])
    # cmodel = mx.mod.Module(symbol = sym1, label_names = []) 
    cmodel.bind(data_shapes=[('data', (1, 3, testimage.shape[1], testimage.shape[2]))])
    cmodel.init_params(arg_params=arg_params1, aux_params=aux_params1)
    # print 'init_params failed'
    onedata = DataBatch(mx.nd.array([testimage[:,:,:]]), 0)
    #print 'batch'
    cmodel.forward(onedata)
    #print 'forward'
    result=cmodel.get_outputs()
    
    heatmap = np.moveaxis(result[1].asnumpy()[0], 0, -1)
    heatmap = cv.resize(heatmap, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC) # INTER_LINEAR
    heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
    
    pagmap = np.moveaxis(result[0].asnumpy()[0], 0, -1)
    pagmap = cv.resize(pagmap, (0,0), fx=8, fy=8, interpolation=cv.INTER_CUBIC)
    pagmap = pagmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
    pagmap = cv.resize(pagmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)
    
    # print heatmap.shape
    # print pagmap.shape
    return heatmap, pagmap

def applyModel(oriImg, param, sym, arg_params, aux_params):
    model = param['model']
    model = model[param['modelID']]
    boxsize = model['boxsize']
    # print(model)
    # print(boxsize)
    
    makeFigure = 0 
    numberPoints = 1
    
    '''
    octave = param['octave']
    starting_range = param['starting_range']
    ending_range = param['ending_range']
    '''
    '''
    octave = 6
    starting_range = 0.85 # 0.25 0.7
    ending_range = 1.5    # 1.2 1.8
    starting_scale = boxsize*1.0/(oriImg.shape[0]*ending_range)
    ending_scale = boxsize*1.0/(oriImg.shape[0]*starting_range)
    
    # print starting_scale, ending_scale
    
    multiplier = list()
    current_scale = math.log(starting_scale, 2)
    while current_scale < math.log(ending_scale, 2):
        # print current_scale
        multiplier.append(pow(2, current_scale))
        current_scale = current_scale+(1.0/octave)
    '''
    boxsize = 368
    scale_search = 0.5, 1.0, 1.5, 2.0
    multiplier = [x * boxsize / oriImg.shape[0] for x in scale_search]
    # print multiplier
    
    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
    pag_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))
    for i in range(len(multiplier)):
        # print i
        cscale = multiplier[i]
        imageToTest = cv.resize(oriImg, (0,0), fx=cscale, fy=cscale, interpolation=cv.INTER_CUBIC)
        heatmap, pagmap = applyDNN(oriImg, imageToTest, sym, arg_params, aux_params)
        # print(heatmap.shape)
        # print(pagmap.shape)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        pag_avg = pag_avg + pagmap / len(multiplier)
        # print 'add one layer'
    return heatmap_avg, pag_avg

def connect56LineVec(oriImg, param, sym, arg_params, aux_params):
    heatmap_avg, paf_avg = applyModel(oriImg, param, sym, arg_params, aux_params)
    # print 'heatmap, paf'
    all_peaks = []
    peak_counter = 0

    for part in range(19-1):
        x_list = []
        y_list = []
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        cid = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (cid[i],) for i in range(len(cid))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)
    # find connection in the specified sequence, center 29 is in the position 15
    limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
               [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
               [1,16], [16,18], [3,17], [6,18]]
    # the middle joints heatmap correpondence
    mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
              [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
              [55,56], [37,38], [45,46]]

    connection_all = []
    special_k = []
    special_non_zero_index = []
    mid_num = 10

    
    for k in range(len(mapIdx)):
        score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]-1]
        candB = all_peaks[limbSeq[k][1]-1]
        # print(k)
        # print(candA)
        # print('---------')
        # print(candB)
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    try:
                        vec = np.subtract(candB[j][:2], candA[i][:2])
                        # print('vec: ',vec)
                        norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                        # print('norm: ', norm)
                        vec = np.divide(vec, norm)
                        # print('normalized vec: ', vec)
                        startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
                                       np.linspace(candA[i][1], candB[j][1], num=mid_num))
                        # print('startend: ', startend)
                        vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
                                          for I in range(len(startend))])
                        # print('vec_x: ', vec_x)
                        vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
                                          for I in range(len(startend))])
                        # print('vec_y: ', vec_y)
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                        # print(score_midpts)
                        # print('score_midpts: ', score_midpts)
                        score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/norm-1, 0)

                        # print('score_with_dist_prior: ', score_with_dist_prior)
                        criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                        # print('score_midpts > param["thre2"]: ', len(np.nonzero(score_midpts > param['thre2'])[0]))
                        criterion2 = score_with_dist_prior > 0

                        if criterion1 and criterion2:
                            # print('match')
                            # print(i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2])
                            connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])
                    except:
                        print 'error rendering'
                    # print('--------end-----------')
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            # print('-------------connection_candidate---------------')
            # print(connection_candidate)
            # print('------------------------------------------------')
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    # print('----------connection-----------')
                    # print(connection)
                    # print('-------------------------------')
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        # elif(nA != 0 or nB != 0):
        else:
            special_k.append(k)
            special_non_zero_index.append(indexA if nA != 0 else indexB)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 20))

    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            try:
                partAs = connection_all[k][:,0]
                partBs = connection_all[k][:,1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])): #= 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)): #1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if(subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2: # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        print "found = 2"
                        membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0: #merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else: # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
            except:
                print "not link"

    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 3 or subset[i][-2]/subset[i][-1] < 0.2:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    
    return candidate, subset

# Load parameters
output_prefix='../realtimePose'
sym, arg_params, aux_params = mx.model.load_checkpoint(output_prefix, 0)

# ground truth
annFile = '/data/datasets/COCO/person_keypoints_trainval2014/person_keypoints_val2014.json'
cocoGt = COCO(annFile)
cats = cocoGt.loadCats(cocoGt.getCatIds())
catIds = cocoGt.getCatIds(catNms=['person'])
imgIds = cocoGt.getImgIds(catIds=catIds )

# Test parameters
if modelId == 1:
    param['scale_search'] = [0.5, 1, 1.5, 2]
    param['thre1'] = 0.1
    param['thre2'] = 0.05 
    param['thre3'] = 0.5 

    param['model'] = dict()
    param['model'][1] = dict()
    param['model'][1]['caffemodel'] = '../model/_trained_COCO/pose_iter_440000.caffemodel'
    param['model'][1]['deployFile'] = '../model/_trained_COCO/pose_deploy.prototxt'
    param['model'][1]['description'] = 'COCO Pose56 Two-level Linevec'
    param['model'][1]['boxsize'] = 368
    param['model'][1]['padValue'] = 128
    param['model'][1]['np'] = 18
    param['model'][1]['part_str'] = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 
                             'Lsho', 'Lelb', 'Lwri', 
                             'Rhip', 'Rkne', 'Rank',
                             'Lhip', 'Lkne', 'Lank',
                             'Leye', 'Reye', 'Lear', 'Rear', 'pt19']
if modelId == 2:
    param['scale_search'] = [0.7, 1, 1.3]
    param['thre1'] = 0.05
    param['thre2'] = 0.01 
    param['thre3'] = 3
    param['thre4'] = 0.1

    param['model'] = dict()
    param['model'][2] = dict()
    param['model'][2]['caffemodel'] = '../model/_trained_MPI/pose_iter_146000.caffemodel'
    param['model'][2]['deployFile'] = '../model/_trained_MPI/pose_deploy.prototxt'
    param['model'][2]['description'] = 'COCO Pose56 Two-level Linevec'
    param['model'][2]['boxsize'] = 368
    param['model'][2]['padValue'] = 128
    param['model'][2]['np'] = 18
    param['model'][2]['part_str'] = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri',  
                                     'Lsho', 'Lelb', 'Lwri', 
                                     'Rhip', 'Rkne', 'Rank', 
                                     'Lhip', 'Lkne', 'Lank', 
                                     'Leye', 'Reye', 'Lear', 'Rear', 'pt19']

# candidate
starttime = time.time()
orderCOCO = [1,0,7,9,11, 6,8,10,13,15,17,12,14,16,3,2,5,4]
myjsonValidate = list(dict())
count = 0
imgIds_num = 2644
#imgIds_num = 100
not_working_num = 0
notworkingimageIds = []
# imgIds_num = len(imgIds)
for i in range(imgIds_num):
    print 'image: ', i
    img = cocoGt.loadImgs(imgIds[i])[0]
    cimg = io.imread('/data/guest_users/liangdong/liangdong/practice_demo/val2014/'+img['file_name'])
    
    if len(cimg.shape)==2:       
        cimgRGB = np.zeros((cimg.shape[0], cimg.shape[1], 3))
        for i in range(3):
            cimgRGB[:, :, i] = cimg
            print cimgRGB.shape
    else:
        cimgRGB = cimg
        
    # print 'image shape'
    # print cimg.shape
    image_id = img['id']
    try:
        candidate, subset = connect56LineVec(cimgRGB, param, sym, arg_params, aux_params)
        print subset
        
        for j in range(len(subset)):
            category_id = 1
            keypoints = np.zeros(51)
            score = 0
            for part in range(18):
                if part == 1:
                    continue
                index = int(subset[j][part])
                if index > 0:
                    realpart = orderCOCO[part]-1
                    if part == 0:
                        keypoints[realpart*3] = candidate[index][0]
                        keypoints[realpart*3+1] = candidate[index][1]
                        keypoints[realpart*3+2] = 1
                        # score = score + candidate[index][2]
                    else:
                        keypoints[(realpart)*3] = candidate[index][0]
                        keypoints[(realpart)*3+1] = candidate[index][1]
                        keypoints[(realpart)*3+2] = 1
                        # score = score + candidate[index][2]
                        
            keypoints_list = keypoints.tolist()
            current_dict = {'image_id' : image_id,
                            'category_id' : category_id,
                            'keypoints' : keypoints_list,
                            'score' : subset[j][-2]}
            myjsonValidate.append(current_dict)
            count = count + 1
            
    except:
        print 'train image not working'
        print image_id
        notworkingimageIds.append(image_id)
        not_working_num = not_working_num+1

annType = ['segm','bbox','keypoints']
annType = annType[2]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print 'Running demo for *%s* results.'%(annType)

import json
with open('evaluationResultFixed.json', 'w') as outfile:
    json.dump(myjsonValidate, outfile)
resJsonFile = 'evaluationResultFixed.json'
cocoDt2 = cocoGt.loadRes(resJsonFile)

image_ids = []
for i in range(imgIds_num):
    img = cocoGt.loadImgs(imgIds[i])[0]
    image_ids.append(img['id'])

print 'len: ', len(image_ids)
print 'not working number ', not_working_num
print  notworkingimageIds
# running evaluation
cocoEval = COCOeval(cocoGt, cocoDt2, annType)
cocoEval.params.imgIds  = image_ids
cocoEval.evaluate()
cocoEval.accumulate()
k = cocoEval.summarize()
endtime = time.time()
print k
print (endtime-starttime)/60.0
