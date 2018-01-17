'''
Created on Jan 16, 2018

@author: kohill
'''


from pycocotools.coco import COCO
## install coco library https://github.com/pdollar/coco
from collections import namedtuple
import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
import pylab,json,time,cv2,os
from numpy import linalg as LA
def coco_convert():
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)
    
    ## load keypoints json
    annFile = '/home/kohill/hszc/data/coco/annotations/person_keypoints_train2014.json' # keypoint file
    trainimagepath = '/home/kohill/hszc/data/coco/train2014'             # train image path
    
    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds )
    jointall=[]
    
    count = 0
    joint_all = dict()
    numimgs = len(imgIds)
    
    for i in range(0, numimgs):
        #print('----image: '+str(imgIds[i])+' ---------------')
        img = coco.loadImgs(imgIds[i])[0]
        img_path =  os.path.join(trainimagepath , img['file_name'])
        img_ori = cv2.imread(img_path)
        img_canvas = img_ori.copy()        
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        
        height = img['height']
        width = img['width']
        
        prev_center = list(list())
        segmentations = list()
        
        for j in range(len(anns)):
#             if anns[j]['num_keypoints'] < 5 or anns[j]['area'] < 32*32:
            segmentations.append(anns[j]['segmentation'])
        try:
            con = [np.array(x).astype(np.int32).reshape(-1,1,2) for x in segmentations]
            
            cv2.drawContours(img_canvas,con,-1,(0,255,255),-1)
        except Exception as e:
            print(e)
            print(len(segmentations))
            print(segmentations)
        for j in range(len(anns)):
            
            ## remove those persons whose keypoints are too small
            if anns[j]['num_keypoints'] < 5 or anns[j]['area'] < 32*32:
                continue
            
            person_center = [anns[j]['bbox'][0] + anns[j]['bbox'][2]/2,
                             anns[j]['bbox'][1] + anns[j]['bbox'][3]/2]
            flag = 0
            isValidation = 0 
            
            for k in range(len(prev_center)):
                dist1 = prev_center[k][0] - person_center[0]
                dist2 = prev_center[k][1] - person_center[1]
                #print dist1, dist2
                if dist1*dist1+dist2*dist2 < prev_center[k][2]*0.3:
                    flag = 1
                    continue
            

            currentjoin={'isValidation': isValidation,
                         'img_paths': trainimagepath + img['file_name'],
                         'objpos': person_center,
                         'image_id': img['id'],
                         'bbox': anns[j]['bbox'],
                         'img_width': width,
                         'img_height': height,
                         'segment_area': anns[j]['area'],
                         'num_keypoints': anns[j]['num_keypoints'],
                         'joint_self': np.zeros((17,3)).tolist(),
                         'scale_provided': anns[j]['bbox'][3]/368.0,
                         'segmentations': segmentations,
                         'joint_others': {},
                         'annolist_index': i ,
                         'people_index': j,
                         'numOtherPeople':0,
                         'scale_provided_other':{},
                         'objpos_other':{},
                         'bbox_other':{},
                         'segment_area_other':{},
                         'num_keypoints_other':{}
                        }    
            
            for part in range(17):
                currentjoin['joint_self'][part][0] = anns[j]['keypoints'][part*3]
                currentjoin['joint_self'][part][1] = anns[j]['keypoints'][part*3+1]
                # 2 means cropped, 0 means occluded by still on image
                if(anns[j]['keypoints'][part*3+2] == 2):
                    currentjoin['joint_self'][part][2] = 1
                elif(anns[j]['keypoints'][part*3+2] == 1):
                    currentjoin['joint_self'][part][2] = 0                
                else:
                    currentjoin['joint_self'][part][2] = 2
                
            count_other = 1     
            currentjoin['joint_others'] ={}
           
            for k in range(len(anns)):
                
                if k==j or anns[k]['num_keypoints']==0:
                    continue
                    
                annop = anns[k]
                currentjoin['scale_provided_other'][count_other] = annop['bbox'][3]/368
                currentjoin['objpos_other'][count_other] = [annop['bbox'][0]+annop['bbox'][2]/2, 
                                            annop['bbox'][1]+annop['bbox'][3]/2]
                currentjoin['bbox_other'][count_other] = annop['bbox']
                currentjoin['segment_area_other'][count_other] = annop['area']
                currentjoin['num_keypoints_other'][count_other] = annop['num_keypoints']
                currentjoin['joint_others'][count_other] = np.zeros((17,3)).tolist()
                
                for part in range(17):
                    currentjoin['joint_others'][count_other][part][0] = annop['keypoints'][part*3]
                    currentjoin['joint_others'][count_other][part][1] = annop['keypoints'][part*3+1]
                    
                    if(annop['keypoints'][part*3+2] == 2):
                        currentjoin['joint_others'][count_other][part][2] = 1
                    elif(annop['keypoints'][part*3+2] == 1):
                        currentjoin['joint_others'][count_other][part][2] = 0
                    else:
                        currentjoin['joint_others'][count_other][part][2] = 2
                    
                  
                currentjoin['numOtherPeople'] = len(currentjoin['joint_others']) 
                count_other = count_other + 1
                            
            prev_center.append([person_center[0], person_center[1],
                                max(anns[j]['bbox'][2], anns[j]['bbox'][3])])
            
            joint_all[count] = currentjoin
    
            count = count + 1    
        plt.imshow(img_canvas[:,:,(2,1,0)])
        plt.show()
if __name__ == '__main__':
    coco_convert()
    pass