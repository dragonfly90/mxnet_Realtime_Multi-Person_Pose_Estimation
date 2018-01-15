'''
Created on Jan 13, 2018

@author: kohill
'''
from __future__ import print_function
from torch.utils.data import Dataset, DataLoader
import torch
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import json,cv2,logging 
from generateLabelCPM import getImageandLabel
numberofparts = 19
numberoflinks = 19
class DataIter(Dataset):
    def __init__(self, 
                 datajson):

        with open(datajson, 'r') as f:
            data = json.load(f)
        self.data = data
        self.keys = data.keys()
    def __getitem__(self, index):
        image, mask, heatmap, pagmap = getImageandLabel(self.data[self.keys[index]],change_dir = True)
        maskscale = mask[0:368:8, 0:368:8, 0]
        heatweight = np.repeat(maskscale[np.newaxis, :, :], 19, axis=0)
        vecweight  = np.repeat(maskscale[np.newaxis, :, :], 38, axis=0)        
        transposeImage = np.transpose(np.float32(image), (2,0,1))
        return list(map(lambda x:torch.from_numpy(x.astype(np.float32)),
                        [np.array(transposeImage),
                         np.array(heatmap),
                         np.array(pagmap),
                         np.array(heatweight),
                         np.array(vecweight),
                         ] ))
    def __len__(self):
        return len(self.keys)
def collate_fn(batch):
    imgs = []
    labels = []
    for sample in batch:
        img = np.array(sample[0].numpy()[np.newaxis,:])
        label = list(map(lambda x:np.array(x.numpy()[np.newaxis,:]),sample[1:]))
        label = np.concatenate(label, axis = 1)
        imgs.append(img)
        labels.append(label)
    data = np.concatenate(imgs,axis = 0)
    label = np.concatenate(labels,axis = 0)
    return [data,label]
def getDataLoader(batch_size = 16):
    test_iter = DataIter("../pose_io/data_v1.json")
    r = DataLoader(test_iter, batch_size=batch_size, shuffle=True, num_workers=5, collate_fn=collate_fn, pin_memory=False,drop_last = True)
    return r
if __name__ == "__main__":
    for data_batch in getDataLoader():
        data = data_batch[0]
        label = data_batch[1]
        for i in range(data.shape[0]):
            img = data[i,:]
            l = label[i,:]
            print(l.shape)
            x = [img,l[0:18],l[19:(19+19*2)],l[19*3:19*4],l[19*4:]]
            for i in range(len(x)):
                print(x[i].shape)
                x[i] = np.transpose(x[i],(1,2,0))
            x[0] = x[0][:,:,(2,1,0)]
            x[0] = x[0].astype(np.uint8)
            fig, axes = plt.subplots(2, len(x)//2 + len(x)%2, figsize=(45, 45),
                                 subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(hspace=0.3, wspace=0.05) 
     
            count = 0
            for j in range(len(axes)):
                for i in range(len(axes[0])):
                    try:
                        img = x[count]
                        count += 1
                    except IndexError:
                        break
                    print(count,len(x))
                    if len(img.shape)>=2 and img.shape[2] > 3:
                        axes[j][i].imshow(np.max(img,axis = 2)) 
                    else:
                        axes[j][i].imshow(img) 
                         
            plt.show()
    pass