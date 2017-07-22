# Reimplementation of human keypoint detection in mxnet

1. You can download model and parameters from google drive:

https://drive.google.com/drive/folders/0BzffphMuhDDMV0RZVGhtQWlmS1U

or check caffe_to_mxnet folder to download original caffe model and transfer it to mxnet model.

2. Test demo: test_model.ipynb
3. Train demo: train_model.ipynb


### Paper Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields

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
- [ ] Add weight vector
- [ ] Train all images
- [ ] Train from vgg model
- [ ] image read and augmentation in C++

