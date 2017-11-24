from easydict import EasyDict as edict
import mxnet as mx
import json
from config.config import config
from generateLabelCPM import *

config.HEATMAP_FEAT_STRIDE = [16, 8, 4]

eps = 2e-5
use_global_stats = True
workspace = 512
res_deps = {'50': (3, 4, 6, 3), '101': (3, 4, 23, 3), '152': (3, 8, 36, 3), '200': (3, 24, 36, 3)}
units = res_deps['50']
filter_list = [256, 512, 1024, 2048]

def residual_unit(data, num_filter, stride, dim_match, name):
    bn1   = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn1')
    act1  = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               no_bias=True, workspace=workspace, name=name + '_conv1')
    bn2   = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn2')
    act2  = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
    conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
                               no_bias=True, workspace=workspace, name=name + '_conv2')
    bn3   = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name=name + '_bn3')
    act3  = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
    conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), no_bias=True,
                               workspace=workspace, name=name + '_conv3')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True,
                                      workspace=workspace, name=name + '_sc')
    sum = mx.sym.ElementWiseSum(*[conv3, shortcut], name=name + '_plus')
    return sum


def get_resnet_conv(data):
    # res1
    data_bn = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=eps, use_global_stats=use_global_stats, name='bn_data')
    conv0   = mx.sym.Convolution(data=data_bn, num_filter=64, kernel=(7, 7), stride=(2, 2), pad=(3, 3),
                               no_bias=True, name="conv0", workspace=workspace)
    bn0   = mx.sym.BatchNorm(data=conv0, fix_gamma=False, eps=eps, use_global_stats=use_global_stats, name='bn0')
    relu0 = mx.sym.Activation(data=bn0, act_type='relu', name='relu0')
    pool0 = mx.symbol.Pooling(data=relu0, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='pool0')

    # res2
    unit = residual_unit(data=pool0, num_filter=filter_list[0], stride=(1, 1), dim_match=False, name='stage1_unit1')
    for i in range(2, units[0] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[0], stride=(1, 1), dim_match=True,
                             name='stage1_unit%s' % i)
    conv_C2 = unit

    # res3
    unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='stage2_unit1')
    for i in range(2, units[1] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[1], stride=(1, 1), dim_match=True,
                             name='stage2_unit%s' % i)
    conv_C3 = unit

    # res4
    unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='stage3_unit1')
    for i in range(2, units[2] + 1):
        unit = residual_unit(data=unit, num_filter=filter_list[2], stride=(1, 1), dim_match=True,
                             name='stage3_unit%s' % i)
    conv_C4 = unit

    conv_feat = [conv_C4, conv_C3, conv_C2]
    return conv_feat

def get_resnet_conv_down(conv_feat):
    # C4 to P4, 1x1 dimension reduction to 256
    P4 = mx.symbol.Convolution(data=conv_feat[0], kernel=(1, 1), num_filter=256, name="P4_lateral")

    # P4 2x upsampling + C3 = P3
    P4_up   = mx.symbol.UpSampling(P4, scale=2, sample_type='nearest', workspace=512, name='P4_upsampling', num_args=1)
    P3_la   = mx.symbol.Convolution(data=conv_feat[1], kernel=(1, 1), num_filter=256, name="P3_lateral")
    P4_clip = mx.symbol.Crop(*[P4_up, P3_la], name="P3_clip")
    P3      = mx.sym.ElementWiseSum(*[P4_clip, P3_la], name="P3_sum")
    P3      = mx.symbol.Convolution(data=P3, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P3_aggregate")

    # P3 2x upsampling + C2 = P2
    P3_up   = mx.symbol.UpSampling(P3, scale=2, sample_type='nearest', workspace=512, name='P3_upsampling', num_args=1)
    P2_la   = mx.symbol.Convolution(data=conv_feat[2], kernel=(1, 1), num_filter=256, name="P2_lateral")
    P3_clip = mx.symbol.Crop(*[P3_up, P2_la], name="P2_clip")
    P2      = mx.sym.ElementWiseSum(*[P3_clip, P2_la], name="P2_sum")
    P2      = mx.symbol.Convolution(data=P2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="P2_aggregate")

    conv_fpn_feat = dict()
    conv_fpn_feat.update({"stride16":P4, "stride8":P3, "stride4":P2})

    return conv_fpn_feat, [P2, P3, P4]

def pafphead(backend, paf_level):
    if type(paf_level)!=str:
        paf_level=str(paf_level)
        
    conv_1_CPM_L1 = mx.symbol.Convolution(name='conv'+paf_level+'_1_CPM_L1', 
        data=backend , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu_1_CPM_L1 = mx.symbol.Activation(name='relu'+paf_level+'_1_CPM_L1', 
        data=conv_1_CPM_L1 , act_type='relu')
   
    conv_2_CPM_L1 = mx.symbol.Convolution(name='conv'+paf_level+'_2_CPM_L1', 
        data=relu_1_CPM_L1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu_2_CPM_L1 = mx.symbol.Activation(name='relu'+paf_level+'_2_CPM_L1',
        data=conv_2_CPM_L1 , act_type='relu')
    
    conv_3_CPM_L1 = mx.symbol.Convolution(name='conv'+paf_level+'_3_CPM_L1', 
        data=relu_2_CPM_L1 , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu_3_CPM_L1 = mx.symbol.Activation(name='relu'+paf_level+'_3_CPM_L1', 
        data=conv_3_CPM_L1 , act_type='relu')
    
    conv_4_CPM_L1 = mx.symbol.Convolution(name='conv'+paf_level+'_4_CPM_L1', 
        data=relu_3_CPM_L1 , num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    relu_4_CPM_L1 = mx.symbol.Activation(name='relu'+paf_level+'_4_CPM_L1', 
        data=conv_4_CPM_L1 , act_type='relu')
    
    conv_5_CPM_L1 = mx.symbol.Convolution(name='conv'+paf_level+'_5_CPM_L1', 
        data=relu_4_CPM_L1 , num_filter=38, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    return conv_5_CPM_L1

def heatmaphead(backend, paf_level):
    if type(paf_level)!=str:
        paf_level=str(paf_level)
        
    conv_1_CPM_L2 = mx.symbol.Convolution(name='conv'+paf_level+'_1_CPM_L2',
        data=backend , num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu_1_CPM_L2 = mx.symbol.Activation(name='relu'+paf_level+'_1_CPM_L2', 
        data=conv_1_CPM_L2, act_type='relu')
   
    conv_2_CPM_L2 = mx.symbol.Convolution(name='conv'+paf_level+'_2_CPM_L2', 
        data=relu_1_CPM_L2, num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu_2_CPM_L2 = mx.symbol.Activation(name='relu'+paf_level+'_2_CPM_L2', 
        data=conv_2_CPM_L2, act_type='relu')
    
    conv_3_CPM_L2 = mx.symbol.Convolution(name='conv'+paf_level+'_3_CPM_L2', 
        data=relu_2_CPM_L2, num_filter=128, pad=(1,1), kernel=(3,3), stride=(1,1), no_bias=False)
    relu_3_CPM_L2 = mx.symbol.Activation(name='relu'+paf_level+'_3_CPM_L2', 
        data=conv_3_CPM_L2, act_type='relu')
    
    conv_4_CPM_L2 = mx.symbol.Convolution(name='conv'+paf_level+'_4_CPM_L2',
        data=relu_3_CPM_L2, num_filter=512, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    relu_4_CPM_L2 = mx.symbol.Activation(name='relu'+paf_level+'_4_CPM_L2',
        data=conv_4_CPM_L2, act_type='relu')
    
    conv_5_CPM_L2 = mx.symbol.Convolution(name='conv'+paf_level+'_5_CPM_L2',
        data=relu_4_CPM_L2, num_filter=19, pad=(0,0), kernel=(1,1), stride=(1,1), no_bias=False)
    
    return conv_5_CPM_L2

def fpn_pose():
    heatmap_feat_stride = config.HEATMAP_FEAT_STRIDE
    
    data = mx.symbol.Variable(name='data')
    '''
    heatmaplabel = mx.sym.Variable("heatmaplabel")
    partaffinityglabel = mx.sym.Variable('partaffinityglabel')
    heatweight = mx.sym.Variable('heatweight')    
    vecweight = mx.sym.Variable('vecweight')
    '''
    heatmaplabel = dict()
    partaffinitylabel = dict()
    heatweight = dict()
    vecweight = dict()
    print 'stride'
    print heatmap_feat_stride
     
    for s in [4, 8, 16]:
        heatmaplabel['heatmap_label%s' %s] = mx.symbol.Variable(name='heatmap_label%s' %s)
        partaffinitylabel['part_affinity_label%s' % s] = mx.symbol.Variable(name='part_affinity_label%s' %s)
        heatweight['heat_weight%s' % s] = mx.symbol.Variable(name='heat_weight%s' %s)
        vecweight['vec_weight%s' % s] = mx.symbol.Variable(name='vec_weight%s' %s)
         
    '''
    for s in heatmap_feat_stride:
        heatmaplabel['heatmap_label%s' % s] = mx.symbol.Variable(name='heatmap_label%s' % s)
        partaffinitylabel['part_affinity_label%s' % s] = mx.symbol.Variable(name='part_affinity_label%s' % s)
        heatweight['heat_weight%s' % s] = mx.symbol.Variable(name='heat_weight%s' % s)
        vecweight['vec_weight%s' % s] = mx.symbol.Variable(name='vec_weight%s' % s)
    '''
    # reshape input
   
    for s in [4, 8, 16]:
        heatmaplabel['heatmap_label%s' % s]  = mx.symbol.Reshape(data = heatmaplabel['heatmap_label%s' %s],
                                                                 shape = (-1,),
                                                                 name = 'heatmap_label%s' %s)
        partaffinitylabel['part_affinity_label%s' % s] = mx.symbol.Reshape(data=partaffinitylabel['part_affinity_label%s' %s],
                                                                           shape = (-1,),
                                                                           name = 'part_affinity_label%s' %s)
        heatweight['heat_weight%s' % s] = mx.symbol.Reshape(data=heatweight['heat_weight%s' %s],
                                                            shape = (-1,),
                                                            name = 'heat_weight%s' %s)
        vecweight['vec_weight%s' % s] = mx.symbol.Reshape(data=vecweight['vec_weight%s' %s],
                                                          shape = (-1,),
                                                          name = 'vec_weight%s' %s)

    #heat_map = mx.symbol.concat(*heatmaplabel_list, dim=0)
    #part_affinitylabel = mx.symbol.concat(*partaffinitylabel_list, dim=0)
    #heat_weight = mx.symbol.concat(*heatweight_list, dim=0)
    #vec_weight = mx.symbol.concat(*vecweight_list, dim=0)
    
    resnet_backend = get_resnet_conv(data)
    resnet_backend_pool = get_resnet_conv_down(resnet_backend)
    
    heat_map_predict_list = []
    part_affinitylabel_predict_list = []
    
    for i in range(len(resnet_backend_pool[1])):
        pafphead_level = pafphead(resnet_backend_pool[1][i], i+1)
        heatmaphead_level = heatmaphead(resnet_backend_pool[1][i], i+1)
        part_affinitylabel_predict_list.append(pafphead_level)
        heat_map_predict_list.append(heatmaphead_level)
      
    #print len(heat_map_predict_list)
    #print len(part_affinitylabel_predict_list)
    '''
    heat_map_concat = mx.symbol.concat(*heat_map_predict_list, dim=0)  
    part_affinitylabel_concat = mx.symbol.concat(*part_affinitylabel_predict_list, dim=0)
    '''
    heatmap_label4r = mx.symbol.Reshape(data=heat_map_predict_list[0], shape=(-1,), name='heatmap_label4r')
    heat_map_square4 = mx.symbol.square(heatmaplabel['heatmap_label4'] - heatmap_label4r)
    heat_map_w4 = heat_map_square4*heatweight['heat_weight4']
    heat_map_loss4  = mx.symbol.MakeLoss(heat_map_w4)
    
    part_affinity_label4r = mx.symbol.Reshape(data=part_affinitylabel_predict_list[0], shape=(-1,), name='part_affinity_label4r')
    part_affinity_square4 = mx.symbol.square(partaffinitylabel['part_affinity_label4'] - part_affinity_label4r)
    part_affinity_w4 = part_affinity_square4*vecweight['vec_weight4']
    part_affinity_loss4  = mx.symbol.MakeLoss(part_affinity_w4)
    
    heatmap_label8r = mx.symbol.Reshape(data=heat_map_predict_list[1], shape=(-1,), name='heatmap_label8r')
    heat_map_square8 = mx.symbol.square(heatmaplabel['heatmap_label8'] - heatmap_label8r)
    heat_map_w8 = heat_map_square8*heatweight['heat_weight8']
    heat_map_loss8  = mx.symbol.MakeLoss(heat_map_w8)
    
    part_affinity_label8r = mx.symbol.Reshape(data=part_affinitylabel_predict_list[1], shape=(-1,), name='part_affinity_label8r')
    part_affinity_square8 = mx.symbol.square(partaffinitylabel['part_affinity_label8'] - part_affinity_label8r)
    part_affinity_w8 = part_affinity_square8*vecweight['vec_weight8']
    part_affinity_loss8  = mx.symbol.MakeLoss(part_affinity_w8)
    
    heatmap_label16r = mx.symbol.Reshape(data=heat_map_predict_list[2], shape=(-1,), name='heatmap_label16r')
    heat_map_square16 = mx.symbol.square(heatmaplabel['heatmap_label16'] - heatmap_label16r)
    heat_map_w16 = heat_map_square16*heatweight['heat_weight16']
    heat_map_loss16  = mx.symbol.MakeLoss(heat_map_w16)
    
    part_affinity_label16r = mx.symbol.Reshape(data=part_affinitylabel_predict_list[2], shape=(-1,), name='part_affinity_label16r')
    part_affinity_square16 = mx.symbol.square(partaffinitylabel['part_affinity_label16'] - part_affinity_label16r)
    part_affinity_w16 = part_affinity_square16*vecweight['vec_weight16']
    part_affinity_loss16  = mx.symbol.MakeLoss(part_affinity_w16)
    
    group = mx.symbol.Group([heat_map_loss4, part_affinity_loss4, heat_map_loss8, part_affinity_loss8,
                            heat_map_loss16, part_affinity_loss16])
    return group
    
    
