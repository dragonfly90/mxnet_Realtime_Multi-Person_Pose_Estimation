## author: Liang Dong
## Generate heat map and part affinity map

import os
import mxnet as mx
import numpy as np
import copy
import re
from google.protobuf import text_format
import json
import cv2 as cv

import scipy
import PIL.Image
import math
import time
from PIL import Image, ImageDraw

from config.config import config
from collections import namedtuple
Point = namedtuple('Point', 'x y')


def rotatePoint(R, pointDict):
    NewPoint = {'x':0, 'y':0}
    NewPoint['x'] = R[0,0]*pointDict['x']+R[0,1]*pointDict['y']+R[0,2]
    NewPoint['y'] = R[1,0]*pointDict['x']+R[1,1]*pointDict['y']+R[1,2]
    return NewPoint

def readmeta(data):
    meta = copy.deepcopy(data)
    meta['img_size'] = {'width': data['img_width'], 'height': data['img_height']}

    joint_self = data['joint_self']
    meta['joint_self'] = {'joints': list(), 'isVisible': list()}
    for i in range(len(joint_self)):
        currentdict = {'x': joint_self[i][0], 'y': joint_self[i][1]}
        meta['joint_self']['joints'].append(currentdict)

        # meta['joint_self']['isVisible'].append(joint_self[i][2])
        if joint_self[i][2] == 3:
            meta['joint_self']['isVisible'].append(3)
        elif joint_self[i][2] == 2:
            meta['joint_self']['isVisible'].append(2)
        elif joint_self[i][2] == 0:
            meta['joint_self']['isVisible'].append(0)
        else:
            meta['joint_self']['isVisible'].append(1)
            if (meta['joint_self']['joints'][i]['x'] < 0 or meta['joint_self']['joints'][i]['y'] < 0
                or meta['joint_self']['joints'][i]['x'] >= meta['img_size']['width'] or
                        meta['joint_self']['joints'][i]['y'] >= meta['img_size']['height']):
                meta['joint_self']['isVisible'][i] = 2

    for key in data['joint_others']:
        joint_other = data['joint_others'][key]
        #print joint_other
        meta['joint_others'][key] = {'joints': list(), 'isVisible': list()}

        for i in range(len(joint_self)):
            currentdict = {'x': joint_other[i][0], 'y': joint_other[i][1]}
            meta['joint_others'][key]['joints'].append(currentdict)

            # meta['joint_self']['isVisible'].append(joint_self[i][2])
            if joint_other[i][2] == 3:
                meta['joint_others'][key]['isVisible'].append(3)
            elif joint_other[i][2] == 2:
                meta['joint_others'][key]['isVisible'].append(2)
            elif joint_other[i][2] == 0:
                meta['joint_others'][key]['isVisible'].append(0)
            else:
                meta['joint_others'][key]['isVisible'].append(1)
                if (meta['joint_others'][key]['joints'][i]['x'] < 0 or meta['joint_others'][key]['joints'][i]['y'] < 0
                    or meta['joint_others'][key]['joints'][i]['x'] >= meta['img_size']['width'] or
                            meta['joint_others'][key]['joints'][i]['y'] >= meta['img_size']['height']):
                    meta['joint_others'][key]['isVisible'][i] = 2

    return meta


def TransformJointsSelf(meta):
    jo = meta['joint_self'].copy()
    newjo = {'joints': list(), 'isVisible': list()}
    COCO_to_ours_1 = [1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
    COCO_to_ours_2 = [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]

   
    for i in range(18):
        currentdict = {'x': (jo['joints'][COCO_to_ours_1[i] - 1]['x'] + jo['joints'][COCO_to_ours_2[i] - 1]['x']) * 0.5,
                       'y': (jo['joints'][COCO_to_ours_1[i] - 1]['y'] + jo['joints'][COCO_to_ours_2[i] - 1]['y']) * 0.5}
        newjo['joints'].append(currentdict)

        if (jo['isVisible'][COCO_to_ours_1[i] - 1] == 2 or jo['isVisible'][COCO_to_ours_2[i] - 1] == 2):
            newjo['isVisible'].append(2)
        elif (jo['isVisible'][COCO_to_ours_1[i] - 1] == 3 or jo['isVisible'][COCO_to_ours_2[i] - 1] == 3):
            newjo['isVisible'].append(3)
        else:
            isVisible = jo['isVisible'][COCO_to_ours_1[i] - 1] and jo['isVisible'][COCO_to_ours_2[i] - 1]
            newjo['isVisible'].append(isVisible)

    
    meta['joint_self'] = newjo


def TransformJointsOther(meta):
    for key in meta['joint_others']:
        jo = meta['joint_others'][key].copy()

        newjo = {'joints': list(), 'isVisible': list()}
        COCO_to_ours_1 = [1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]
        COCO_to_ours_2 = [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]

        
        for i in range(18):
            currentdict = {
                'x': (jo['joints'][COCO_to_ours_1[i] - 1]['x'] + jo['joints'][COCO_to_ours_2[i] - 1]['x']) * 0.5,
                'y': (jo['joints'][COCO_to_ours_1[i] - 1]['y'] + jo['joints'][COCO_to_ours_2[i] - 1]['y']) * 0.5}
            newjo['joints'].append(currentdict)

            if ((jo['isVisible'][COCO_to_ours_1[i] - 1] == 2) or (jo['isVisible'][COCO_to_ours_2[i] - 1] == 2)):
                newjo['isVisible'].append(2)
            elif ((jo['isVisible'][COCO_to_ours_1[i] - 1] == 3) or (jo['isVisible'][COCO_to_ours_2[i] - 1] == 3)):
                newjo['isVisible'].append(3)
            else:
                isVisible = jo['isVisible'][COCO_to_ours_1[i] - 1] and jo['isVisible'][COCO_to_ours_2[i] - 1]
                newjo['isVisible'].append(isVisible)

        
        meta['joint_others'][key] = newjo


def TransformMetaJoints(meta):
    TransformJointsSelf(meta)
    TransformJointsOther(meta)

def augmentation_scale(meta, oriImg, maskmiss):
    newmeta = copy.deepcopy(meta)
    dice2 = np.random.uniform()
    scale_multiplier = (config.TRAIN.scale_max - config.TRAIN.scale_min) * dice2 + config.TRAIN.scale_min
    if newmeta['scale_provided'] > 0:
        scale_abs = config.TRAIN.target_dist / newmeta['scale_provided']
        scale = scale_abs * scale_multiplier
    else:
        scale = 1
    
    if config.TRAIN.scale_set == False:
        scale = 1

    resizeImage = cv.resize(oriImg, (0, 0), fx=scale, fy=scale)
    maskmiss_scale = cv.resize(maskmiss, (0,0), fx=scale, fy=scale)
    
    newmeta['objpos'][0] *= scale
    newmeta['objpos'][1] *= scale

    for i in range(len(meta['joint_self']['joints'])):
        newmeta['joint_self']['joints'][i]['x'] *= scale
        newmeta['joint_self']['joints'][i]['y'] *= scale

    for i in meta['joint_others']:
        for j in range(len(meta['joint_others'][i]['joints'])):
            newmeta['joint_others'][i]['joints'][j]['x'] *= scale
            newmeta['joint_others'][i]['joints'][j]['y'] *= scale

    return (newmeta, resizeImage, maskmiss_scale)


def onPlane(p, img_size):
    if (p[0] < 0 or p[1] < 0):
        return False
    if (p[0] >= img_size[0] - 1 or p[1] >= img_size[1]):
        return False
    return True

def augmentation_flip(meta, croppedImage, maskmiss):
    dice = np.random.uniform()
    newmeta = copy.deepcopy(meta)
    if config.TRAIN.flip and dice > 0.5: 
        flipImage = cv.flip(croppedImage, 1)
        maskmiss_flip = cv.flip(maskmiss, 1)

        newmeta['objpos'][0] =  newmeta['img_width'] - 1- newmeta['objpos'][0]

        for i in range(len(meta['joint_self']['joints'])):
            newmeta['joint_self']['joints'][i]['x'] = newmeta['img_width'] - 1- newmeta['joint_self']['joints'][i]['x']

        for i in meta['joint_others']:
            for j in range(len(meta['joint_others'][i]['joints'])):
                newmeta['joint_others'][i]['joints'][j]['x'] = newmeta['img_width'] - 1 - newmeta['joint_others'][i]['joints'][j]['x']
    else:
        flipImage = croppedImage.copy()
        maskmiss_flip = maskmiss.copy()
        
    return (newmeta, flipImage, maskmiss_flip)

def augmentation_rotate(meta, flipimage, maskmiss):
    newmeta = copy.deepcopy(meta)
    #print maskmiss.dtype
    
    dice2 = np.random.uniform()
    degree = (dice2 - 0.5)*2*config.TRAIN.max_rotate_degree 
    
    #print degree
    center = (368/2, 368/2)
    
    R = cv.getRotationMatrix2D(center, degree, 1.0)
    
    rotatedImage = cv.warpAffine(flipimage, R, (368,368))
    maskmiss_rotated = cv.warpAffine(maskmiss, R, (368,368))
    
    for i in range(len(meta['joint_self']['joints'])):
        newmeta['joint_self']['joints'][i] = rotatePoint(R, newmeta['joint_self']['joints'][i])

    for i in meta['joint_others']:
        for j in range(len(meta['joint_others'][i]['joints'])):
            newmeta['joint_others'][i]['joints'][j] = rotatePoint(R, newmeta['joint_others'][i]['joints'][j])
    
    return (newmeta, rotatedImage, maskmiss_rotated)

def augmentation_crop(meta, oriImg, maskmiss):
    # dice_x = np.random.uniform()
    # dice_y = np.random.uniform()
    dice_x = 0.5
    dice_y = 0.5
    # float dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    # float dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
    crop_x = config.TRAIN.crop_size_x
    crop_y = config.TRAIN.crop_size_y

    x_offset = int((dice_x - 0.5) * 2 * config.TRAIN.center_perterb_max)
    y_offset = int((dice_y - 0.5) * 2 * config.TRAIN.center_perterb_max)

    #print x_offset, y_offset
    newmeta2 = copy.deepcopy(meta)

    center = [x_offset + meta['objpos'][0], y_offset + meta['objpos'][1]]

    offset_left = -int(center[0] - crop_x / 2)
    offset_up = -int(center[1] - crop_y / 2)

    img_dst = np.full((crop_y, crop_x, 3), 128, dtype=np.uint8)
    maskmiss_croppad = np.full((crop_y, crop_x, 3), False, dtype=np.uint8)
    for i in range(crop_y):
        for j in range(crop_x):
            coord_x_on_img = int(center[0] - crop_x / 2 + j)
            coord_y_on_img = int(center[1] - crop_y / 2 + i)
            # print coord_x_on_img, coord_y_on_img
            if (onPlane([coord_x_on_img, coord_y_on_img], [oriImg.shape[1], oriImg.shape[0]])):
                img_dst[i, j, :] = oriImg[coord_y_on_img, coord_x_on_img, :]
                maskmiss_croppad[i, j] = maskmiss[coord_y_on_img, coord_x_on_img]
                
    newmeta2['objpos'][0] += offset_left
    newmeta2['objpos'][1] += offset_up

    for i in range(len(meta['joint_self']['joints'])):
        newmeta2['joint_self']['joints'][i]['x'] += offset_left
        newmeta2['joint_self']['joints'][i]['y'] += offset_up

    for i in meta['joint_others']:
        for j in range(len(meta['joint_others'][i]['joints'])):
            newmeta2['joint_others'][i]['joints'][j]['x'] += offset_left
            newmeta2['joint_others'][i]['joints'][j]['y'] += offset_up

    return (newmeta2, img_dst, maskmiss_croppad)


def putGaussianMaps(entry, rows, cols, center,
                    stride, grid_x, grid_y, sigma):
    start = stride / 2.0 - 0.5
    center = Point(int(center.x), int(center.y))

    for g_y in range(grid_y):
        for g_x in range(grid_x):
            x = start + g_x * stride
            y = start + g_y * stride
            d2 = (x - center.x) * (x - center.x) + (y - center.y) * (y - center.y)
            exponent = d2 / 2.0 / sigma / sigma
            if (exponent > 4.6052):
                continue
            entry[g_y, g_x] += np.exp(-exponent)
            if (entry[g_y, g_x] > 1):
                entry[g_y, g_x] = 1
    return entry


def putVecMaps(entryX, entryY, count, center1, center2, stride, grid_x, grid_y, sigma, thre):
    if (type(center1) != Point):
        raise Exception
    if (type(center2) != Point):
        raise Exception
    centerA = Point(0.125 * center1.x, 0.125 * center1.y)
    centerB = Point(0.125 * center2.x, 0.125 * center2.y)
    # print centerA, centerB
    bc = Point(centerB.x - centerA.x, centerB.y - centerA.y)
    # print grid_x, grid_y
    min_x = max(int(round(min(centerA.x, centerB.x)) - thre), 0)
    max_x = min(int(round(max(centerA.x, centerB.x))) + thre, grid_x)
    min_y = max(int(round(min(centerA.y, centerB.y) - thre)), 0)
    max_y = min(int(round(max(centerA.y, centerB.y) + thre)), grid_y)

    # print('-------------')
    # print(min_x, max_x)
    # print(min_y, max_y)
    norm_bc = np.sqrt(bc.x * bc.x + bc.y * bc.y)
    if norm_bc == 0:
        return 
    #print '----norm_bc------'
    #print norm_bc
    
    bc = Point(bc.x / norm_bc, bc.y / norm_bc)

    for g_y in range(min_y, max_y):
        for g_x in range(min_x, max_x):
            ba = Point(g_x - centerA.x, g_y - centerA.y)
            dist = abs(ba.x * bc.y - ba.y * bc.x)
            # print dist
            if (dist <= thre):
                cnt = count[g_y, g_x]
                if (cnt == 0):
                    entryX[g_y, g_x] = bc.x
                    entryY[g_y, g_x] = bc.y

                else:
                    entryX[g_y, g_x] = (entryX[g_y, g_x] * cnt + bc.x) / (cnt + 1)
                    entryY[g_y, g_x] = (entryY[g_y, g_x] * cnt + bc.y) / (cnt + 1)
                    count[g_y, g_x] = cnt + 1


def generateLabelMap(img_aug, meta):
    thre = 1
    crop_size_width = 368
    crop_size_height = 368

    augmentcols = 368
    augmentrows = 368
    stride = 8
    grid_x = augmentcols / stride
    grid_y = augmentrows / stride
    sigma = 7.0

    heat_map = list()
    for i in range(19):
        heat_map.append(np.zeros((crop_size_width / stride, crop_size_height / stride)))

    for i in range(18):
        center = Point(meta['joint_self']['joints'][i]['x'], meta['joint_self']['joints'][i]['y'])

        if (meta['joint_self']['isVisible'][i] <= 1):
            putGaussianMaps(heat_map[i], 368, 368, center, stride,
                            grid_x, grid_y, sigma)

        for j in meta['joint_others']:
            center = Point(meta['joint_others'][j]['joints'][i]['x'],
                           meta['joint_others'][j]['joints'][i]['y'])
            if (meta['joint_others'][j]['isVisible'][i] <= 1):
                putGaussianMaps(heat_map[i], 368, 368, center, stride,
                                grid_x, grid_y, sigma)

    ### put background channel
    
    for g_y in range(grid_y):
        for g_x in range(grid_x):
            maximum=0
            for i in range(18):
                if maximum<heat_map[i][g_y, g_x]:
                    maximum = heat_map[i][g_y, g_x]
            heat_map[18][g_y,g_x]=max(1.0-maximum,0.0)
                          
    # mid_1 = [2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16]
    # mid_2 = [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]
    
    mid_1 = [2, 2, 3, 4, 6, 7, 2, 9, 10, 2, 12, 13, 2, 1, 15, 1, 16, 3, 6]
    mid_2 = [3, 6, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 1, 15, 17, 16, 18, 17, 18]
    
    thre = 1

    pag_map = list()
    for i in range(38):
        pag_map.append(np.zeros((46, 46)))

    for i in range(19):
        count = np.zeros((46, 46))
        jo = meta['joint_self']

        if (jo['isVisible'][mid_1[i] - 1] <= 1 and jo['isVisible'][mid_2[i] - 1] <= 1):
            center1 = Point(jo['joints'][mid_1[i] - 1]['x'], jo['joints'][mid_1[i] - 1]['y'])
            center2 = Point(jo['joints'][mid_2[i] - 1]['x'], jo['joints'][mid_2[i] - 1]['y'])
            putVecMaps(pag_map[2 * i], pag_map[2 * i + 1], count, center1,
                       center2, stride, 46, 46, sigma, thre)

        for j in meta['joint_others']:
            jo = meta['joint_others'][j]
            if (jo['isVisible'][mid_1[i] - 1] <= 1 and jo['isVisible'][mid_2[i] - 1] <= 1):
                center1 = Point(jo['joints'][mid_1[i] - 1]['x'], jo['joints'][mid_1[i] - 1]['y'])
                center2 = Point(jo['joints'][mid_2[i] - 1]['x'], jo['joints'][mid_2[i] - 1]['y'])
                putVecMaps(pag_map[2 * i], pag_map[2 * i + 1], count, center1,
                           center2, stride, 46, 46, sigma, thre)

    return (heat_map, pag_map)

def getMask(meta):
    nx, ny = meta['img_width'], meta['img_height']
    maskall = np.zeros((ny, nx))

    try: 
        if(len(meta['segmentations']) > 0):
            for i in range(len(meta['segmentations'])): 
                seg = meta['segmentations'][i]
                if len(seg) > 0:
                    nlen = len(seg[0])
                    if nlen > 5:
                        poly = zip(seg[0][0:nlen+2:2], seg[0][1:nlen+1:2])
                        img = Image.new("L", [nx, ny], 0)
                        ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
                        mask = np.array(img)
                        maskall = np.logical_or(mask, maskall)
    except:
        print 'full mask'
    
    return np.logical_not(maskall)

def getImageandLabel(iterjson):

    meta = readmeta(iterjson)
    
    TransformMetaJoints(meta)

    oriImg = cv.imread(meta['img_paths'])
    maskmiss = getMask(meta)
    maskmiss = maskmiss.astype(np.uint8)
    
    newmeta, resizeImage, maskmiss_scale = augmentation_scale(meta, oriImg, maskmiss)

    newmeta2, croppedImage, maskmiss_cropped = augmentation_crop(newmeta, resizeImage,
                                                                   maskmiss_scale)
    
    newmeta3, rotatedImage, maskmiss_rotate= augmentation_rotate(newmeta2, croppedImage, 
                                                                maskmiss_cropped)
    
    newmeta4, flipImage, maskmiss_flip = augmentation_flip(newmeta3, rotatedImage, maskmiss_rotate)

    heatmap, pagmap = generateLabelMap(flipImage, newmeta4)

    return (flipImage, maskmiss_flip, heatmap, pagmap)

'''

with open('pose_io/data.json', 'r') as f:
    data = json.load(f)

meta = readmeta(data['5'])
TransformMetaJoints(meta)

cimagepath = meta['img_paths']
print cimagepath
oriImg = cv.imread(cimagepath)  # B,G,R order

import matplotlib

get_ipython().magic(u'matplotlib inline')
import pylab as plt

type(oriImg)
f = plt.imshow(oriImg[:, :, [2, 1, 0]])  # reorder it before displaying
x = [i['x'] for i in meta['joint_self']['joints']]
y = [i['y'] for i in meta['joint_self']['joints']]
plt.plot(x, y, 'r.')

newmeta, resizeImage = augmentation_scale(meta, oriImg)


f = plt.imshow(resizeImage[:, :, [2, 1, 0]])  # reorder it before displaying
x = [i['x'] for i in newmeta['joint_self']['joints'][8:9]]
y = [i['y'] for i in newmeta['joint_self']['joints'][8:9]]
plt.plot(x, y, 'r.')


f = plt.imshow(resizeImage[:, :, [2, 1, 0]])  # reorder it before displaying
x = [i['x'] for i in newmeta['joint_self']['joints']]
y = [i['y'] for i in newmeta['joint_self']['joints']]
plt.plot(x, y, 'r.')

newmeta2, croppedImage = augmentation_crop(newmeta, resizeImage)


img_dst = np.full((368, 368, 3), 128)
print img_dst.shape


resizeImage[1, 1, :]


img_dst[1, 1, :] = resizeImage[1, 1, :]


croppedImage[10, 100, :]


f = plt.imshow(croppedImage[:, :, [2, 1, 0]])  # reorder it before displaying
x = [i['x'] for i in newmeta2['joint_self']['joints']]
y = [i['y'] for i in newmeta2['joint_self']['joints']]
plt.plot(x, y, 'r.')


newmeta['joint_self']['joints']

# In[232]:

from collections import namedtuple

Point = namedtuple('Point', 'x y')
center = Point(5, 6)

crop_size_x = 368
crop_size_y = 368
center_perterb_max = 40

heatmap, pag_map = generateLabelMap(croppedImage, newmeta2)

'''

'''
### show heat map

plt.figure();  # plt.axis('off')

for i in range(0, 1):
    plt.imshow(cv.resize(heatmap[i], (368, 368)), alpha=(1 - 1.0 * i / 10))

plt.imshow(croppedImage, alpha=0.5)

plt.show()

'''


'''
### show part affinity graph


from numpy import ma

U = cv.resize(pag_map[4], (368, 368))
V = cv.resize(pag_map[5], (368, 368))

X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
M = np.zeros(U.shape, dtype='bool')
# M[U**2 + V**2 < 0.5 * 0.5] = True
U = ma.masked_array(U, mask=M)
V = ma.masked_array(V, mask=M)

plt.figure()
plt.imshow(croppedImage, alpha=.5)
s = 5
Q = plt.quiver(X[::s, ::s], Y[::s, ::s], U[::s, ::s], V[::s, ::s],
               scale=50, headaxislength=4, alpha=.5, width=0.001, color='r')

fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10, 10)

'''