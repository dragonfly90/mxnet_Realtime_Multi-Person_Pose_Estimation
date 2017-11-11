### Reimplementation of human keypoint detection in mxnet

1. You can download mxnet model and parameters(coco and MPII) from google drive:

   https://drive.google.com/drive/folders/0BzffphMuhDDMV0RZVGhtQWlmS1U

   or check caffe_to_mxnet folder to download original caffe model and transfer it to mxnet model.
   
   install heatmap and pafmap cython:  cython/rebuild.sh
   
2. Test demo based on model of coco dataset: testModel.ipynb
3. Test demo based on model of MPII dataset: testModel_mpi.ipynb
4. Train with batch_size: TrainWeight.py 
5. Check if heat map, part affinity graph map, mask are generated correctly in training: test_generateLabel.ipynb
6. Evaluation on coco validation dataset : evaluation_coco.py

### Cite paper Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields

```
@article{cao2016realtime,
  title={Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  author={Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
  journal={arXiv preprint arXiv:1611.08050},
  year={2016}
  }
```

original caffe training https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose



## TODO:
- [x] Test demo
- [x] Train demo
- [x] Add image augmentation: rotation, flip
- [x] Add weight vector
- [x] Train all images
- [x] Train from vgg model
- [x] Evaluation code
- [x] Generate heat map and part affinity graph map in C++
- [ ] image read and augmentation in C++

## Train result
We tested the code using two K60 GPUS on COCO dataset, with batch size set to 10 and learning rate set to 0.00004. and using vgg pretrained model on <data.mxnet.io> to initialize our parameters. After 20 epochs, we tested our model on COCO validation dataset(only 50 images) and we got a score of 0.10.  

```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.048
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.183
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.019
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.078
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.035
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.066
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.224
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.022
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.075
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.054

```

The traning process is not so easy, I found this model even can't converge if all layers are initialized randomly, I guess one reason is that this model uses many convolution layers with a large kernel, whose big pad may introduce much noise, and another reason may be the fact that this model uses MSE as loss function, and maybe it's better to use sigmoid as the avtivation function of the last layer and use entropy loss function instead. 


## Other implementations 

[Original caffe training model](https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose)

[Original data preparation and demo](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

[Pytorch](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)

[keras](https://github.com/raymon-tian/keras_Realtime_Multi-Person_Pose_Estimation)
