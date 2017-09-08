import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

config.boxsize = 368

config.TEST = edict()

config.TEST.scale_search = [0.5, 1, 1.5, 2, 2.3]

config.TEST.imgIds_num = -1

config.TEST.model_path = 'testConfigModel'

#config.TEST.model_path = '../realtimePose'

config.TEST.epoch = 0

config.TEST.imgIds_num = 5

config.TRAIN = edict()

config.TRAIN.num_epoch = 1

config.TRAIN.initial_model = '../realtimePose'

config.TRAIN.output_model = 'testConfigModel'



