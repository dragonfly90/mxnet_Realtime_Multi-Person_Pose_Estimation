#encoding=utf-8
'''
Created on 2017年11月14日

@author: kohill
'''
import numpy as np
import cv2,math
import mxnet as mx
import matplotlib.pyplot as plt
import test.visualization as visual
import networkx as nx
import os
from pprint import pprint
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = '0'
test_images_path = "/data1/yks/dataset/ai_challenger/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911/"
test_images_path = "/data1/yks/dataset/openpose_dataset/mpi/images"
def _get_module(sym,save_prefix,save_epoch,gpu = 6):
    _, newargs,aux_args = mx.model.load_checkpoint(save_prefix, save_epoch)        
    model = mx.mod.Module(symbol=sym, context=[mx.gpu(gpu)],label_names = [])
    model.bind(data_shapes=[('data', (1, 3, 368, 368))],for_training = False)
    model.init_params(arg_params=newargs, aux_params=aux_args, allow_missing=False,allow_extra=True)
    return model
def get_module():
    from symbol.resnet_v1_101_deeplab_dcn_paf import resnet_v1_101_deeplab_dcn_paf as resnet_dcn_pafmap
    from symbol.resnet import get_symbol    
    sym_heatmap = get_symbol(14,num_layers = 152,image_shape = "3,368,368",for_training = False)
#     sym_heatmap = resnet_v1_101_deeplab_dcn()
#     sym_heatmap = sym_heatmap.get_symbol(num_classes = 14,is_train = False)
    sym_pafmap = resnet_dcn_pafmap()
    sym_pafmap = sym_pafmap.get_symbol(num_classes = 38,is_train = False)
    model_pafmap = _get_module(sym_pafmap, "models/paf/paf_wildpose", 30400, 6)
    model_heatmap = _get_module(sym_heatmap, "models/resnet/wildpose", 9900, 7)
    return model_heatmap,model_pafmap
def getHeatAndPAF(model_heatmap,model_pafmap,img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img,(368,368))
#         img = padimg(img,368)
    imgs_transpose = np.transpose(np.float32(img[:,:,:]), (2,0,1))
    imgs_batch = mx.io.DataBatch([mx.nd.array([imgs_transpose[:,:,:]])])
    model_heatmap.forward(imgs_batch)
    model_pafmap.forward(imgs_batch)

    result = model_heatmap.get_outputs()    
    offmap = np.moveaxis(result[1].asnumpy()[0], 0, -1)
    offmap = cv2.resize(offmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)   
    heatmap = np.moveaxis(result[0].asnumpy()[0], 0, -1)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)   
    
    result = model_pafmap.get_outputs()
    pafmap = np.moveaxis(result[0].asnumpy()[0], 0, -1)
    pafmap = cv2.resize(pafmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)    
    
        
    return img_path,img,heatmap,offmap,pafmap

def conv(offsetx,offsety,heatmap,radius = 12,dataset_radius = 12):
    '''
    Todo: It's need to remove this.(try training a small convolution network.
    '''
    heatmap /= np.max(heatmap)
    
    output = np.zeros_like(heatmap,dtype = np.float32)
    
    for i in range(offsetx.shape[0]):
        for j in range(offsetx.shape[1]):
            try:
                x = int(round(offsety[i,j]* dataset_radius + i ))
                y = int(round(offsetx[i,j]* dataset_radius + j ))
                output[x,y] += 1
            except IndexError:
                pass

    output =   np.logical_or((output >=  6) , (heatmap >= 0.5)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(8,8))
    output = cv2.erode(output,kernel)
    _,contours,_ = cv2.findContours(output,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        center = rect[0]
        points.append([center[1],center[0]])
    return output,points
def parse_keypoints(heatmaps,offset_scores):
    offmaps= []
    key_points = []
    for i in range(offset_scores.shape[2]//2):
        offsetx = offset_scores[:,:,i*2]
        offsety = offset_scores[:,:,i*2 + 1]
        offmap,points = conv(offsetx,offsety,heatmaps[:,:,i])
        offmaps.append(offmap)
        for m,n in points:
            key_points.append([i,m,n])
    return key_points
def parse_paf_and_heat(model_heatmap,model_pafmap,img_path):
    limbSeq = [[13, 14], [14, 1], [14, 4], [1, 2], [2, 3], [4, 5], [5, 6], [1, 7], [7, 8],
            [8, 9], [4, 10], [10, 11], [11, 12],[1,4],[4,1],[7,10],[10,7],[1,10],[4,7]]
    limbSeq = [[13, 14], [14, 1], [14, 4], [1, 2], [2, 3], [4, 5], [5, 6], [1, 7], [7, 8],
            [8, 9], [4, 10], [10, 11], [11, 12],[13,14],[14,1],[7,10],[10,7],[1,10],[4,7]]

    limbSeq = [[x-1,y-1] for x,y in limbSeq]
    numoflinks = len(limbSeq)
    numofparts = 14
    img_path,img,heatmaps,offset_scores,pafmaps = getHeatAndPAF(model_heatmap,model_pafmap,img_path)
    key_points = parse_keypoints(heatmaps, offset_scores)
    visual.show_pafmap(img_path, pafmaps)
#    visual.plot_points(img, key_points)
    key_points_partid = [[] for _ in range( numofparts)]
    for i,[partid,m,n] in enumerate(key_points):
        key_points_partid[partid].append((m,n,i))
    pprint(key_points_partid)
    pprint(key_points)
    G = nx.DiGraph()
    for limb_id in range(len(limbSeq)):
        partA,partB = limbSeq[limb_id]
        for pA in key_points_partid[partA]:
            for pB in key_points_partid[partB]:
                mid_num = 50
                startend = [ [int(x) for x in np.linspace(pA[0], pB[0], num=mid_num)],
                             [int(x) for x in np.linspace(pA[1], pB[1], num=mid_num)] ]
                vec_n = pafmaps[:,:,limb_id*2][startend]
                vec_m = pafmaps[:,:,limb_id*2 + 1][startend]
                norm = np.sqrt((pA[0]-pB[0]) **2 +(pA[1]-pB[1]) **2 ) + 1e-6
                vec_normlized= np.array([pB[0]-pA[0],pB[1]-pA[1]]) / norm                 
                mid_inte = vec_normlized[0] * vec_m + vec_normlized[1] * vec_n
#                 mid_inte =  vec_m* vec_m + vec_n* vec_n                
                non_zero_count  = np.sum(np.abs(mid_inte) > 0.001)
                if non_zero_count >= int(0.8 * len(mid_inte)):
                    edge_score = np.sum(mid_inte)
                    if edge_score/len(mid_inte) > 0.15:
                        G.add_edge(pA[2], pB[2],weight = edge_score)
                        print(pA[2], pB[2])
    visual.plot_network(img,G, key_points)
    
if __name__ == "__main__":
    model_heatmap,model_pafmap = get_module()
    for x,y,z in os.walk(test_images_path):
        for name in z:
            parse_paf_and_heat(model_heatmap,model_pafmap,os.path.join(x,name))
