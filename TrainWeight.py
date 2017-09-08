import sys
sys.path.append('../../practice_demo')
from modelCPMWeight import *
from config.config import config

sym = CPMModel()
testsym, arg_params, aux_params = mx.model.load_checkpoint(config.TRAIN.initial_model, 0)

class NewModule(mx.mod.Module):

    def fit(self, train_data, num_epoch, arg_params=arg_params, aux_params=aux_params, begin_epoch=0):
             
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=[('data', (1, 3, 368, 368))], label_shapes=[
        ('heatmaplabel',(1, 19, 46, 46)),
        ('partaffinityglabel',(1,38,46,46)),
        ('heatweight',(1,19,46,46)),
        ('vecweight',(1,38,46,46))])
   
        self.init_params(arg_params=arg_params, aux_params=aux_params)

        self.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.00000001), ))
       
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
                for j in range(len(prediction)):
                    lossiter = prediction[j].asnumpy()
                    cls_loss = np.sum(lossiter)
                    sumloss += cls_loss
                    numpixel +=lossiter.shape[0]

                sumerror=sumerror+(math.sqrt(sumloss/numpixel))
                print sumerror
                if i%100==0:
                    print i
                
                cmodel.backward()   
                self.update()
                
                '''
                if i > 40:
                    break
                '''
                try:
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch)
                except StopIteration:
                    end_of_batch = True
          
                nbatch += 1
             
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)
            train_data.reset()
            
cocodata = cocoIterweight('pose_io/data.json',
                          'data', (1, 3, 368, 368),
                          ['heatmaplabel','partaffinityglabel','heatweight','vecweight'],
                          [(1, 19, 46, 46),(1,38,46,46),(1,19,46,46),(1,38,46,46)])

cmodel = NewModule(symbol=sym, context=mx.gpu(3),
                   label_names=['heatmaplabel',
                                'partaffinityglabel',
                                'heatweight',
                                'vecweight'])

cmodel.fit(cocodata, num_epoch = config.TRAIN.num_epoch, arg_params=arg_params, aux_params=aux_params)

cmodel.save_checkpoint(config.TRAIN.output_model, 0)
