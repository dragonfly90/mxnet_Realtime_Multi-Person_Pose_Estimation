import sys
sys.path.append('../../practice_demo')
from modelCPM import *
from config.config import config

class poseModule(mx.mod.Module):

    def fit(self, train_data, num_epoch, batch_size, prefix, carg_params=None, begin_epoch=0):
        
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=[('data', (batch_size, 3, 368, 368))], label_shapes=[
        ('heatmaplabel', (batch_size, 19, 46, 46)),
        ('partaffinityglabel', (batch_size, 38, 46, 46)),
        ('heatweight', (batch_size, 19, 46, 46)),
        ('vecweight', (batch_size, 38, 46, 46))])
   
        self.init_params(arg_params = carg_params, aux_params={},
                         allow_missing=True)
        self.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.00004), ))
        
        losserror_list_paf1 = []
        losserror_list_heat1 = []
        losserror_list_paf2 = []
        losserror_list_heat2 = []
        losserror_list_paf3 = []
        losserror_list_heat3 = []
        losserror_list_paf4 = []
        losserror_list_heat4 = []
        losserror_list_paf5 = []
        losserror_list_heat5 = []
        losserror_list_paf6 = []
        losserror_list_heat6 = []
        
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            i=0
            sumerror_paflevel1 = 0
            sumerror_heatmaplevel1 = 0
            sumerror_paflevel2 = 0
            sumerror_heatmaplevel2 = 0
            sumerror_paflevel3 = 0
            sumerror_heatmaplevel3 = 0
            sumerror_paflevel4 = 0
            sumerror_heatmaplevel4 = 0
            sumerror_paflevel5 = 0
            sumerror_heatmaplevel5 = 0
            sumerror_paflevel6 = 0
            sumerror_heatmaplevel6 = 0
            
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
                
                lossiter = prediction[2].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                print 'start paf level2: ', cls_loss
                sumerror_paflevel2 = sumerror_paflevel2 + cls_loss
                
                lossiter = prediction[4].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                print 'start paf level3: ', cls_loss
                sumerror_paflevel3 = sumerror_paflevel3 + cls_loss
                
                lossiter = prediction[6].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                print 'start paf level4: ', cls_loss
                sumerror_paflevel4 = sumerror_paflevel4 + cls_loss
                
                lossiter = prediction[8].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                print 'start paf level5: ', cls_loss
                sumerror_paflevel5 = sumerror_paflevel5 + cls_loss
                
                lossiter = prediction[10].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                sumerror_paflevel6 = sumerror_paflevel6 + cls_loss
                print 'end paf level6: ', cls_loss
                
                lossiter = prediction[1].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                print 'start heat level1: ', cls_loss
                sumerror_heatmaplevel1 = sumerror_heatmaplevel1 + cls_loss
                
                lossiter = prediction[3].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                print 'start heat level2: ', cls_loss
                sumerror_heatmaplevel2 = sumerror_heatmaplevel2 + cls_loss
                
                lossiter = prediction[5].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                print 'start heat level3: ', cls_loss
                sumerror_heatmaplevel3 = sumerror_heatmaplevel3 + cls_loss
                
                lossiter = prediction[7].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                print 'start heat level4: ', cls_loss
                sumerror_heatmaplevel4 = sumerror_heatmaplevel4 + cls_loss
                
                lossiter = prediction[9].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                print 'start heat level5: ', cls_loss
                sumerror_heatmaplevel5 = sumerror_heatmaplevel5 + cls_loss
                  
                lossiter = prediction[11].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                print 'end heat level6: ', cls_loss
                sumerror_heatmaplevel6 = sumerror_heatmaplevel6 + cls_loss
  
                '''
                lossiter = prediction[-1].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                print 'paf: ', cls_loss
                sumerror = sumerror + cls_loss
                lossiter = prediction[-2].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                print 'heat: ', cls_loss
                '''   
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
            losserror_list_paf2.append(sumerror_paflevel2/i)
            losserror_list_heat2.append(sumerror_heatmaplevel2/i)
            losserror_list_paf3.append(sumerror_paflevel3/i)
            losserror_list_heat3.append(sumerror_heatmaplevel3/i)
            losserror_list_paf4.append(sumerror_paflevel4/i)
            losserror_list_heat4.append(sumerror_heatmaplevel4/i)
            losserror_list_paf5.append(sumerror_paflevel5/i)
            losserror_list_heat5.append(sumerror_heatmaplevel5/i)
            losserror_list_paf6.append(sumerror_paflevel6/i)
            losserror_list_heat6.append(sumerror_heatmaplevel6/i)
            
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)
            if epoch%1 == 0:
                self.save_checkpoint(prefix, epoch +1)
            
            train_data.reset()
            
        
        text_file = open("OutputLossError.txt", "w")
        
        text_file.write('paf level 1\n')
        text_file.write(' '.join([str(i) for i in losserror_list_paf1]))
        text_file.write('\npaf level 2')
        text_file.write(' '.join([str(i) for i in losserror_list_paf2]))
        text_file.write('\npaf level 3\n')
        text_file.write(' '.join([str(i) for i in losserror_list_paf3]))
        text_file.write('\npaf level 4\n')
        text_file.write(' '.join([str(i) for i in losserror_list_paf4]))
        text_file.write('\npaf level 5\n')
        text_file.write(' '.join([str(i) for i in losserror_list_paf5]))
        text_file.write('\npaf level 6\n')
        text_file.write(' '.join([str(i) for i in losserror_list_paf6]))
        
        text_file.write('\nheat map level 1\n')
        text_file.write(' '.join([str(i) for i in losserror_list_heat1]))
        text_file.write('\nheat map level 2\n')
        text_file.write(' '.join([str(i) for i in losserror_list_heat2]))
        text_file.write('\nheat map level 3\n')
        text_file.write(' '.join([str(i) for i in losserror_list_heat3]))
        text_file.write('\nheat map level 4\n')
        text_file.write(' '.join([str(i) for i in losserror_list_heat4]))
        text_file.write('\nheat map level 5\n')
        text_file.write(' '.join([str(i) for i in losserror_list_heat5]))
        text_file.write('\nheat map level 6\n')
        text_file.write(' '.join([str(i) for i in losserror_list_heat6]))
        
        text_file.close() 
        
batch_size = 10
cocodata = cocoIterweightBatch('pose_io/data.json',
                               'data', (batch_size, 3, 368,368),
                               ['heatmaplabel','partaffinityglabel','heatweight','vecweight'],
                               [(batch_size, 19, 46, 46), (batch_size, 38, 46, 46),
                                (batch_size, 19, 46, 46), (batch_size, 38, 46, 46)],
                               batch_size
                             )

sym = poseSymbol()
cmodel = poseModule(symbol=sym, context=mx.cpu(),
                    label_names=['heatmaplabel',
                                 'partaffinityglabel',
                                 'heatweight',
                                 'vecweight'])
## Load parameters from vgg
warmupModel = '/data/guest_users/liangdong/liangdong/practice_demo/mxnet_CPM/model/vgg19'
testsym, arg_params, aux_params = mx.model.load_checkpoint(warmupModel, 0)
newargs = {}
for ikey in config.TRAIN.vggparams:
    newargs[ikey] = arg_params[ikey]

prefix = 'vggpose'
starttime = time.time()
cmodel.fit(cocodata, num_epoch = 3, batch_size = batch_size, prefix = prefix, carg_params = newargs)
cmodel.save_checkpoint(prefix, config.TRAIN.num_epoch)
endtime = time.time()

print 'cost time: ', (endtime-starttime)/60
