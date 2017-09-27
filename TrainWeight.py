import sys
sys.path.append('../../practice_demo')
from modelCPM import *
from config.config import config

class poseModule(mx.mod.Module):

    def fit(self, train_data, num_epoch, batch_size, prefix, carg_params=None, begin_epoch=0):
        
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=[('data', (batch_size, 3, 368, 368))], label_shapes=[
        ('heatmaplabel', (batch_size, 19, 46, 46)),
        ('partaffinityglabel', (batch_size, 38,46,46)),
        ('heatweight', (batch_size,19,46,46)),
        ('vecweight', (batch_size,38,46,46))])
   
        self.init_params(arg_params = carg_params, aux_params={},
                         allow_missing=True)
        self.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.00004), ))
        losserror_list = []
        
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            nbatch = 0
            data_iter = iter(train_data)
            end_of_batch = False
            next_data_batch = next(data_iter)
            i=0
            sumerror=0
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
                print 'start paf: ', cls_loss
                sumerror = sumerror + cls_loss
                
                lossiter = prediction[1].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                print 'start heat: ', cls_loss
                sumerror = sumerror + cls_loss
                                  
                lossiter = prediction[10].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                sumerror = sumerror + cls_loss
                print 'end paf: ', cls_loss
                
                lossiter = prediction[11].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                sumerror = sumerror + cls_loss
                print 'end heat: ', cls_loss
                
                '''
                lossiter = prediction[-1].asnumpy()              
                cls_loss = np.sum(lossiter)/batch_size
                print 'paf: ', cls_loss
                sumerror = sumerror + cls_loss
                lossiter = prediction[-2].asnumpy()
                cls_loss = np.sum(lossiter)/batch_size
                print 'heat: ', cls_loss
                '''
                
                sumerror = sumerror + cls_loss
                    
               
                if i > 20:
                    break
                
                #sumerror=sumerror+(math.sqrt(sumloss/numpixel))    
                if i%100==0:
                    print i
                
                cmodel.backward()   
                self.update()           
                    
                try:
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch)
                except StopIteration:
                    end_of_batch = True
                nbatch += 1
            
                    
            print '------Error-------'
            print sumerror/i
            losserror_list.append(sumerror/i)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)
            
            self.save_checkpoint(prefix, epoch +1)
            
            train_data.reset()
            
        print losserror_list
        text_file = open("OutputLossError.txt", "w")
        text_file.write(' '.join([str(i) for i in losserror_list]))
        text_file.close() 
        
batch_size = 2
cocodata = cocoIterweightBatch('pose_io/data.json',
                               'data', (batch_size, 3, 368,368),
                               ['heatmaplabel','partaffinityglabel','heatweight','vecweight'],
                               [(batch_size, 19, 46, 46), (batch_size, 38, 46, 46),
                                (batch_size, 19, 46, 46), (batch_size, 38, 46, 46)],
                               batch_size
                             )

sym = poseSymbol()
cmodel = poseModule(symbol=sym, context=mx.gpu(1),
                    label_names=['heatmaplabel',
                                 'partaffinityglabel',
                                 'heatweight',
                                 'vecweight'])

testsym, newargs, aux_params = mx.model.load_checkpoint(config.TRAIN.initial_model, 0)

prefix = 'vggpose'
starttime = time.time()
cmodel.fit(cocodata, num_epoch = config.TRAIN.num_epoch, batch_size = batch_size, prefix = prefix, carg_params = newargs)
cmodel.save_checkpoint(prefix, config.TRAIN.num_epoch)
endtime = time.time()

print 'cost time: ', (endtime-starttime)/60
