from featurePyramidCPM import *

sym = fpn_pose()

class DataBatch(object):
    def __init__(self, data, heatmap_label4, part_affinity_label4, heat_weight4, vec_weight4,
                 heatmap_label8, part_affinity_label8, heat_weight8, vec_weight8,
                 heatmap_label16, part_affinity_label16, heat_weight16, vec_weight16, pad=0):
        self.data = [data]
        self.label = [heatmap_label4, part_affinity_label4, heat_weight4, vec_weight4,
                      heatmap_label8, part_affinity_label8, heat_weight8, vec_weight8,
                      heatmap_label16, part_affinity_label16, heat_weight16, vec_weight16]
        self.pad = pad


class cocoIterweightBatchFPN:
    def __init__(self, datajson,
                 data_names, 
                 data_shapes, 
                 label_names,
                 label_shapes, 
                 batch_size = 1):
        
        print 'initialize begin'
        self._data_shapes = data_shapes
        self._label_shapes = label_shapes
        self._provide_data = zip([data_names], [data_shapes])
        self._provide_label = zip(label_names, label_shapes)
        self._batch_size = batch_size

        with open(datajson, 'r') as f:
            data = json.load(f)

        self.num_batches = len(data)/self._batch_size*self._batch_size
        self.data = data
        self.cur_batch = 0
        self.keys = data.keys()
        
        
    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label
    
    def resizeImages(self, images, scale):
        resizeimages = [cv.resize(image, (0, 0), fx=scale, fy=scale)for image in images]
        return resizeimages
    
    def next(self):
        if self.cur_batch < 5: #self.num_batches:
            
            # load training data 
            
            transposeImage_batch = []
            #heatmap_batch = []
            #pagmap_batch = []
            #heatweight_batch = []
            #vecweight_batch = []
            heatmap_on_imgs = dict()
            pagmap_on_imgs = dict()
            heatweight_on_imgs = dict()
            vecweight_on_imgs = dict()
            
            for s in [4,8,16]:
                heatmap_on_imgs.update({'stride%s' % s: list()})
                pagmap_on_imgs.update({'stride%s' % s: list()})
                heatweight_on_imgs.update({'stride%s' % s: list()})
                vecweight_on_imgs.update({'stride%s' % s: list()})
               
            for i in range(self._batch_size):
                if self.cur_batch >= self.num_batches:
                    break
                    
                
                image, mask, heatmap_8, pagmap_8= getImageandLabel(self.data[self.keys[self.cur_batch]])
                heatmap_4 = self.resizeImages(heatmap_8, 2)
                heatmap_16 = self.resizeImages(heatmap_8, 0.5)
                pagmap_4 = self.resizeImages(pagmap_8, 2)
                pagmap_16 = self.resizeImages(pagmap_8, 0.5)
                
 
                print 'heatmap len: ', len(heatmap_4)
                print heatmap_4[0].shape
                print 'heatmap len: ', len(heatmap_8)
                print heatmap_8[0].shape
                print 'heatmap len: ', len(heatmap_16)
                print heatmap_16[0].shape
                
                print 'pagmap len: ', len(pagmap_4)
                print pagmap_4[0].shape
                print 'pagmap len: ', len(pagmap_8)
                print pagmap_8[0].shape
                print 'pagmap len: ', len(pagmap_16)
                print pagmap_16[0].shape
                
                maskscale_4 = mask[0:368:4, 0:368:4, 0]
                heatweight_4 = np.repeat(maskscale_4[np.newaxis, :, :], 19, axis=0)
                vecweight_4  = np.repeat(maskscale_4[np.newaxis, :, :], 38, axis=0)
                
                maskscale_8 = mask[0:368:8, 0:368:8, 0]
                heatweight_8 = np.repeat(maskscale_8[np.newaxis, :, :], 19, axis=0)
                vecweight_8  = np.repeat(maskscale_8[np.newaxis, :, :], 38, axis=0)
               
                maskscale_16 = mask[0:368:16, 0:368:16, 0]
                heatweight_16 = np.repeat(maskscale_16[np.newaxis, :, :], 19, axis=0)
                vecweight_16  = np.repeat(maskscale_16[np.newaxis, :, :], 38, axis=0)
                
                transposeImage = np.transpose(np.float32(image), (2,0,1))/256 - 0.5
 
                self.cur_batch += 1
                '''                   
                heatmap_on_imgs.update({'stride%s' % s: list()})
                pagmap_on_imgs.update({'stride%s' % s: list()})
                heatweight_on_imgs.update({'stride%s' % s: list()})
                vecweight_on_imgs.update({'stride%s' % s: list()})
                '''     
                transposeImage_batch.append(transposeImage)
                heatmap_on_imgs['stride4'].append(heatmap_4) 
                pagmap_on_imgs['stride4'].append(pagmap_4)
                heatweight_on_imgs['stride4'].append(heatweight_4)
                vecweight_on_imgs['stride4'].append(vecweight_4)
                
                heatmap_on_imgs['stride8'].append(heatmap_8) 
                pagmap_on_imgs['stride8'].append(pagmap_8)
                heatweight_on_imgs['stride8'].append(heatweight_8)
                vecweight_on_imgs['stride8'].append(vecweight_8)
                
                heatmap_on_imgs['stride16'].append(heatmap_16) 
                pagmap_on_imgs['stride16'].append(pagmap_16)
                heatweight_on_imgs['stride16'].append(heatweight_16)
                vecweight_on_imgs['stride16'].append(vecweight_16)
                
            return DataBatch(
                mx.nd.array(transposeImage_batch),
                mx.nd.array(heatmap_on_imgs['stride4']),
                mx.nd.array(pagmap_on_imgs['stride4']),
                mx.nd.array(heatweight_on_imgs['stride4']),
                mx.nd.array(vecweight_on_imgs['stride4']),
                mx.nd.array(heatmap_on_imgs['stride8']),
                mx.nd.array(pagmap_on_imgs['stride8']),
                mx.nd.array(heatweight_on_imgs['stride8']),
                mx.nd.array(vecweight_on_imgs['stride8']),
                mx.nd.array(heatmap_on_imgs['stride16']),
                mx.nd.array(pagmap_on_imgs['stride16']),
                mx.nd.array(heatweight_on_imgs['stride16']),
                mx.nd.array(vecweight_on_imgs['stride16']))
        else:
            raise StopIteration  
            

class poseFPNModule(mx.mod.Module):

    def fit(self, train_data, num_epoch, batch_size, prefix, carg_params=None, begin_epoch=0):
        
        assert num_epoch is not None, 'please specify number of epochs'

        
        self.bind(data_shapes=[('data', (batch_size, 3, 368, 368))], 
                  label_shapes=[
                    ('heatmap_label4', (batch_size, 19, 92, 92)),
                    ('part_affinity_label4', (batch_size, 38, 92, 92)),
                    ('heat_weight4', (batch_size, 19, 92, 92)),
                    ('vec_weight4', (batch_size, 38, 92, 92)),
        
                    ('heatmap_label8', (batch_size, 19, 46, 46)),
                    ('part_affinity_label8', (batch_size, 38, 46, 46)),
                    ('heat_weight8', (batch_size, 19, 46, 46)),
                    ('vec_weight8', (batch_size, 38, 46, 46)),
        
                    ('heatmap_label16', (batch_size, 19, 23, 23)),
                    ('part_affinity_label16', (batch_size, 38, 23, 23)),
                    ('heat_weight16', (batch_size, 19, 23, 23)),
                    ('vec_weight16', (batch_size, 38, 23, 23)),
                ])
        
        self.init_params(arg_params = carg_params, aux_params={},
                         allow_missing=True)
        self.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.00004), ))
        
        losserror_list_paf1 = []
        losserror_list_heat1 = []
        
        
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            i=0
            sumerror_paflevel1 = 0
            sumerror_heatmaplevel1 = 0
            
            
            while not end_of_batch:
                data_batch = next_data_batch
                cmodel.forward(data_batch, is_train=True)       # compute predictions  
                prediction=cmodel.get_outputs()
                i=i+1
                sumloss=0
                numpixel=0
                print 'iteration: ', i
                
                '''
                print 'length of prediction:', len(prediction)
                for j in range(len(prediction)):
                    
                    lossiter = prediction[j].asnumpy()
                    cls_loss = np.sum(lossiter)
                    print 'loss: ', cls_loss
                    sumloss += cls_loss
                    numpixel +=lossiter.shape[0]
                    
                '''
                lossiter = prediction[0].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                print 'start paf level1: ', cls_loss
                sumerror_paflevel1 = sumerror_paflevel1 + cls_loss
            
                lossiter = prediction[1].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                print 'start heat level1: ', cls_loss
                sumerror_heatmaplevel1 = sumerror_heatmaplevel1 + cls_loss
                
                if i%100==0:
                    print i     
                    
                cmodel.backward()   
                self.update()           
                
                ## remov this line if you want to train all images
                                  
                try:
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch)
                except StopIteration:
                    end_of_batch = True
                nbatch += 1
            
                    
            losserror_list_paf1.append(sumerror_paflevel1/i)
            losserror_list_heat1.append(sumerror_heatmaplevel1/i)
          
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)
            if epoch%1 == 0:
                self.save_checkpoint(prefix, epoch +1)
            
            train_data.reset()
            
        
        text_file = open("OutputLossErrorFPN.txt", "w")
        
        text_file.write('paf level 1\n')
        text_file.write(' '.join([str(i) for i in losserror_list_paf1]))
        
        text_file.write('\nheat map level 1\n')
        text_file.write(' '.join([str(i) for i in losserror_list_heat1]))
        
        text_file.close() 

cmodel = poseFPNModule(symbol=sym, 
                       context=mx.gpu(3),
                       label_names=['heatmap_label4', 'part_affinity_label4', 'heat_weight4','vec_weight4',
                                   'heatmap_label8', 'part_affinity_label8', 'heat_weight8', 'vec_weight8',
                                   'heatmap_label16', 'part_affinity_label16', 'heat_weight16', 'vec_weight16']
                      )

batch_size=10

cocodata = cocoIterweightBatchFPN('pose_io/data.json',
                                  'data', (batch_size, 3, 368,368),
                                  ['heatmap_label4', 'part_affinity_label4', 'heat_weight4','vec_weight4',
                                   'heatmap_label8', 'part_affinity_label8', 'heat_weight8', 'vec_weight8',
                                   'heatmap_label16', 'part_affinity_label16', 'heat_weight16', 'vec_weight16'],
                                  [(batch_size, 19, 92, 92), (batch_size, 38, 92, 92), (batch_size, 19, 92, 92), 
                                   (batch_size, 38, 92, 92), (batch_size, 19, 46, 46), (batch_size, 38, 46, 46), 
                                   (batch_size, 19, 46, 46), (batch_size, 38, 46, 46), (batch_size, 19, 23, 23), 
                                   (batch_size, 38, 23, 23), (batch_size, 19, 23, 23), (batch_size, 38, 23, 23)],
                                    batch_size)

import time
warmupModel = '../mxnet_CPM/model/vgg19'
testsym, arg_params, aux_params = mx.model.load_checkpoint(warmupModel, 0)
newargs = {}
for ikey in config.TRAIN.vggparams:
        newargs[ikey] = arg_params[ikey]

arg_shapes, output_shapes, aux_shapes = sym.infer_shape(data=(1,3,368,368),
                                                        heatmap_label4=(1, 19, 92, 92),
                                                        part_affinity_label4=(1, 38, 92, 92),
                                                        heat_weight4=(1, 19, 92, 92),
                                                        vec_weight4=(1, 38, 92, 92),
                                                        heatmap_label8= (1, 19, 46, 46), 
                                                        part_affinity_label8=(1, 38, 46, 46),
                                                        heat_weight8=(1, 19, 46, 46), 
                                                        vec_weight8=(1, 38, 46, 46), 
                                                        heatmap_label16=(1, 19, 23, 23), 
                                                        part_affinity_label16=(1, 38, 23, 23), 
                                                        heat_weight16=(1, 19, 23, 23), 
                                                        vec_weight16=(1, 38, 23, 23))

prefix = 'vggFPNpose'
starttime = time.time()
cmodel.fit(cocodata, num_epoch = 1, batch_size = batch_size, prefix = prefix, carg_params = newargs)
cmodel.save_checkpoint(prefix, 1)


import argparse

import mxnet as mx

from rcnn.config import config, default, generate_config
from rcnn.tools.demo_maskrcnn import demo_maskrcnn

def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    # general
    parser.add_argument('--network', help='network name', default=default.network, type=str)
    parser.add_argument('--dataset', help='dataset name', default=default.dataset, type=str)
    args, rest = parser.parse_known_args()
    generate_config(args.network, args.dataset)
    parser.add_argument('--image_set', help='image_set name', default=default.test_image_set, type=str)
    parser.add_argument('--root_path', help='output data folder', default=default.root_path, type=str)
    parser.add_argument('--dataset_path', help='dataset path', default=default.dataset_path, type=str)
    parser.add_argument('--result_path', help='result path', type=str)
    # testing
    parser.add_argument('--prefix', help='model to test with', default=default.rcnn_prefix, type=str)
    parser.add_argument('--epoch', help='model to test with', default=default.rcnn_epoch, type=int)
    parser.add_argument('--gpu', help='GPU device to test with', default=0, type=int)
    # rcnn
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    parser.add_argument('--has_rpn', help='generate proposals on the fly', action='store_true')
    parser.add_argument('--proposal', help='can be ss for selective search or rpn', default='rpn', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    ctx = mx.gpu(args.gpu)
    print args
    demo_maskrcnn(args.network, args.dataset, args.image_set, args.root_path, args.dataset_path, args.result_path,
                ctx, args.prefix, args.epoch,
                args.vis, args.shuffle, args.has_rpn, args.proposal, args.thresh)

if __name__ == '__main__':
    main()