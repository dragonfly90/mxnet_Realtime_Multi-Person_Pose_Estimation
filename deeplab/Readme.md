### result on deeplab-resnet-101
In the beginning, we set batch size to 8 and learning rate to 0.0001

one epoch(only fist 50 images):

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.225
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.484
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.163
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.224
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.275
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.257
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.507
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.224
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.221
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.312
```

5 epochs(only fist 50 images)

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.268
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.525
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.219
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.265
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.300
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.545
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.276
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.265
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.361
```

Then, we change learning rate to 0.00001

6 epochs(only fist 50 images):

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.308
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.562
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.292
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.389
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.590
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.351
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.292
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.424
```

11 epochs(only fist 50 images):

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.321
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.576
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.335
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.292
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.351
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.597
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.373
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.293
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.444
 
```

Then, we change learning rate to 0.000001,and change optimizer to SGD

