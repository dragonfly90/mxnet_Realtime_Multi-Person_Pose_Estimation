### Reimplementation of human keypoint detection in mxnet

1. You can download mxnet model and parameters(coco and MPII) from google drive:

   https://drive.google.com/drive/folders/0BzffphMuhDDMV0RZVGhtQWlmS1U

   or check caffe_to_mxnet folder to download original caffe model and transfer it to mxnet model.
   
   install heatmap and pafmap cython:  cython/rebuild.sh
   
2. Test demo based on model of coco dataset: testModel.ipynb

3. Test demo based on model of MPII dataset: testModel_mpi.ipynb

4. Train with vgg model warm up. You can download mxnet model and parameters for vgg19 from [here](http://data.mxnet.io/models/imagenet/vgg/)
   ```bash
   python TrainWeightOnVgg.py
   ```
   Train from CMU's converted model
   ```bash
   python TrainWeight.py 
   ```
5. Check if heat map, part affinity graph map, mask are generated correctly in training: test_generateLabel.ipynb
6. Evaluation on coco validation dataset with transfered mxnet model: evaluation_coco.py

The result is as following, the mean average precision (AP) over 10 OKS threshold  on the first 2644 images in the val set is 0.550, which is 0.577 in original implementation.

```bash
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets= 20 ] = 0.550
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets= 20 ] = 0.800
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets= 20 ] = 0.610
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.541
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.576
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 20 ] = 0.591
Average Recall (AR) @[ IoU=0.50 | area= all | maxDets= 20 ] = 0.812
Average Recall (AR) @[ IoU=0.75 | area= all | maxDets= 20 ] = 0.644
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.549
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.651
```

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
- [ ] Enhancement: feature pyramid backend in training, symbol and iterator in featurePyramidCPM.py

## Training with vgg warm up
```bash
python TrainWeightOnVgg.py
```
(1) Before
We tested the code using two K80 GPUS on COCO dataset, with batch size set to 10 and learning rate set to 0.00004. and using vgg pretrained vgg model to initialize our parameters. After 20 epochs, we tested our model on COCO validation dataset(only 50 images) and we got only 0.048 as mAP, very low compared to original implementation. Please reach us if you have some ideas about this issue.  

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

(2) Fix the iterator bug, no data augmentation

We tested the code using one TITAN X (Pascal) on COCO dataset, with batch size set to 10 and learning rate set to 0.00004. and using pretrained vgg model to initialize our parameters. After 4 epochs, we tested our model on COCO validation dataset(only first 50 images) and we got only 0.115 as mAP, the original transfered model gots 0.530 . 
```bash
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.115
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.350
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.030
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.168
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.091
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.141
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.373
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.067
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.164
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.117
```
After 18 epochs, we tested our model on COCO validation dataset(only first 50 images) and we got only 0.226 as mAP. 
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.226
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.434
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.201
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.254
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.226
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.250
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.440
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.239
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.261
```
After 23 epochs, we tested our model on COCO validation dataset(only first 50 images) and we got only 0.231 as mAP. 

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.231
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.466
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.230
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.245
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.249
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.251
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.470
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.261
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.243
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.278
```
After 36 epochs, we tested our model on COCO validation dataset(only first 50 images) and we got only 0.229 as mAP. 
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.229
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.442
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.218
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.233
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.260
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.257
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.455
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.269
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.232
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.302
```

The traning process is not so easy, I found this model even can't converge if all layers are initialized randomly, I guess one reason is that this model uses many convolution layers with a large kernel, whose big pad may introduce much noise, and another reason may be the fact that this model uses MSE as loss function, and maybe it's better to use sigmoid as the avtivation function of the last layer and use entropy loss function instead. 


## Other implementations 

[Original caffe training model](https://github.com/CMU-Perceptual-Computing-Lab/caffe_rtpose)

[Original data preparation and demo](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

[Pytorch](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation)

[keras](https://github.com/raymon-tian/keras_Realtime_Multi-Person_Pose_Estimation)
