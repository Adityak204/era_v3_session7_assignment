# MNIST Digit Classification with CNN

A deep learning project implementing a Convolutional Neural Network (CNN) for classifying handwritten digits using the MNIST dataset. This projects aim to reach 99.4% accuracy on the test set with less than 8k parameters and within 15 epochs.

## Experiments

### 1. Base Model
**Target:** Aim in this experiment is to create a based model first which adhere to the constraints of less than 8k parametes. And monitor the train and test accuracy

**Model Architecture:**
#### Layerwise Details

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
              ReLU-2            [-1, 8, 28, 28]               0
            Conv2d-3            [-1, 8, 28, 28]             576
              ReLU-4            [-1, 8, 28, 28]               0
            Conv2d-5            [-1, 8, 28, 28]             576
              ReLU-6            [-1, 8, 28, 28]               0
         MaxPool2d-7            [-1, 8, 14, 14]               0
            Conv2d-8            [-1, 4, 14, 14]              32
            Conv2d-9            [-1, 8, 14, 14]             288
             ReLU-10            [-1, 8, 14, 14]               0
           Conv2d-11           [-1, 16, 14, 14]           1,152
             ReLU-12           [-1, 16, 14, 14]               0
           Conv2d-13           [-1, 16, 14, 14]           2,304
             ReLU-14           [-1, 16, 14, 14]               0
        MaxPool2d-15             [-1, 16, 7, 7]               0
           Conv2d-16              [-1, 4, 7, 7]              64
           Linear-17                   [-1, 10]           1,970
================================================================
Total params: 7,034
Trainable params: 7,034
Non-trainable params: 0
----------------------------------------------------------------
```

**Results:**
- Max Train Accuracy: 99.92%
- Max Test Accuracy: 99.17%

**Observation:** Model has reached max accuracy of 99.92% on training data. It shows that model is able to learn the training data very well. But it is severly over-fitting as there is significant gap between train and test accuracy.

### 2. Model with Dropout (Addressing Overfitting)
**Target:** We will try to address the overfitting by introducing dropouts

**Model Architecture:**
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
              ReLU-2            [-1, 8, 28, 28]               0
            Conv2d-3            [-1, 8, 28, 28]             576
              ReLU-4            [-1, 8, 28, 28]               0
            Conv2d-5            [-1, 8, 28, 28]             576
              ReLU-6            [-1, 8, 28, 28]               0
           Dropout-7            [-1, 8, 28, 28]               0
         MaxPool2d-8            [-1, 8, 14, 14]               0
            Conv2d-9            [-1, 4, 14, 14]              32
           Conv2d-10            [-1, 8, 14, 14]             288
             ReLU-11            [-1, 8, 14, 14]               0
           Conv2d-12           [-1, 16, 14, 14]           1,152
             ReLU-13           [-1, 16, 14, 14]               0
           Conv2d-14           [-1, 16, 14, 14]           2,304
             ReLU-15           [-1, 16, 14, 14]               0
          Dropout-16           [-1, 16, 14, 14]               0
        MaxPool2d-17             [-1, 16, 7, 7]               0
           Conv2d-18              [-1, 4, 7, 7]              64
           Linear-19                   [-1, 10]           1,970
================================================================
Total params: 7,034
Trainable params: 7,034
Non-trainable params: 0
----------------------------------------------------------------
```

**Results:**
- Max Train Accuracy: 98.43%
- Max Test Accuracy: 99.44%

**Observation:** Image transformation really helped in increasing Test accuracy. Train accuracy has been impacted but it is not overfitting anymore.

### 3. Final Model with Data Transformations
**Target:** Based on random test samples review, exp with data transformation to improve test accuracy

**Model Architecture:**
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              72
       BatchNorm2d-2            [-1, 8, 28, 28]              16
              ReLU-3            [-1, 8, 28, 28]               0
            Conv2d-4            [-1, 8, 28, 28]             576
       BatchNorm2d-5            [-1, 8, 28, 28]              16
              ReLU-6            [-1, 8, 28, 28]               0
            Conv2d-7           [-1, 16, 28, 28]           1,152
       BatchNorm2d-8           [-1, 16, 28, 28]              32
              ReLU-9           [-1, 16, 28, 28]               0
          Dropout-10           [-1, 16, 28, 28]               0
        MaxPool2d-11           [-1, 16, 14, 14]               0
           Conv2d-12            [-1, 4, 14, 14]              64
           Conv2d-13            [-1, 8, 14, 14]             288
      BatchNorm2d-14            [-1, 8, 14, 14]              16
             ReLU-15            [-1, 8, 14, 14]               0
           Conv2d-16           [-1, 16, 14, 14]           1,152
      BatchNorm2d-17           [-1, 16, 14, 14]              32
             ReLU-18           [-1, 16, 14, 14]               0
           Conv2d-19           [-1, 16, 14, 14]           2,304
      BatchNorm2d-20           [-1, 16, 14, 14]              32
             ReLU-21           [-1, 16, 14, 14]               0
          Dropout-22           [-1, 16, 14, 14]               0
        MaxPool2d-23             [-1, 16, 7, 7]               0
           Conv2d-24              [-1, 4, 7, 7]              64
           Linear-25                   [-1, 10]           1,970
================================================================
Total params: 7,786
Trainable params: 7,786
Non-trainable params: 0
----------------------------------------------------------------
```

**Results:**
- Max Train Accuracy: 99.43%
- Max Test Accuracy: 99.23%

**Observation:** Dropouts have decreased the gap between train and test accuracy significantly. But test accuracy is still far away from 99.4% threshold.

## Training and Test Logs
```
********* Epoch = 1 *********
loss=0.1862 batch_id=58: 100%|██████████| 59/59 [00:17<00:00,  3.36it/s]

Epoch 1: Train set: Average loss: 0.3854, Accuracy: 52814/60000 (88.02%)



Test set: Average loss: 0.0753, Accuracy: 9756/10000 (97.56%)

LR =  [0.01]
********* Epoch = 2 *********
loss=0.0934 batch_id=58: 100%|██████████| 59/59 [00:16<00:00,  3.49it/s]

Epoch 2: Train set: Average loss: 0.1268, Accuracy: 57606/60000 (96.01%)



Test set: Average loss: 0.0544, Accuracy: 9830/10000 (98.30%)

LR =  [0.01]
********* Epoch = 3 *********
loss=0.0948 batch_id=58: 100%|██████████| 59/59 [00:17<00:00,  3.39it/s]

Epoch 3: Train set: Average loss: 0.1018, Accuracy: 58037/60000 (96.73%)



Test set: Average loss: 0.0517, Accuracy: 9844/10000 (98.44%)

LR =  [0.01]
********* Epoch = 4 *********
loss=0.1110 batch_id=58: 100%|██████████| 59/59 [00:17<00:00,  3.39it/s]

Epoch 4: Train set: Average loss: 0.0926, Accuracy: 58188/60000 (96.98%)



Test set: Average loss: 0.0468, Accuracy: 9861/10000 (98.61%)

LR =  [0.01]
********* Epoch = 5 *********
loss=0.0652 batch_id=58: 100%|██████████| 59/59 [00:16<00:00,  3.54it/s]

Epoch 5: Train set: Average loss: 0.0835, Accuracy: 58382/60000 (97.30%)



Test set: Average loss: 0.0392, Accuracy: 9868/10000 (98.68%)

LR =  [0.01]
********* Epoch = 6 *********
loss=0.0784 batch_id=58: 100%|██████████| 59/59 [00:17<00:00,  3.47it/s]

Epoch 6: Train set: Average loss: 0.0789, Accuracy: 58431/60000 (97.39%)



Test set: Average loss: 0.0401, Accuracy: 9871/10000 (98.71%)

LR =  [0.01]
********* Epoch = 7 *********
loss=0.0717 batch_id=58: 100%|██████████| 59/59 [00:16<00:00,  3.53it/s]

Epoch 7: Train set: Average loss: 0.0755, Accuracy: 58513/60000 (97.52%)



Test set: Average loss: 0.0304, Accuracy: 9892/10000 (98.92%)

LR =  [0.01]
********* Epoch = 8 *********
loss=0.0469 batch_id=58: 100%|██████████| 59/59 [00:17<00:00,  3.39it/s]

Epoch 8: Train set: Average loss: 0.0712, Accuracy: 58598/60000 (97.66%)



Test set: Average loss: 0.0292, Accuracy: 9905/10000 (99.05%)

LR =  [0.01]
********* Epoch = 9 *********
loss=0.0634 batch_id=58: 100%|██████████| 59/59 [00:16<00:00,  3.52it/s]

Epoch 9: Train set: Average loss: 0.0682, Accuracy: 58631/60000 (97.72%)



Test set: Average loss: 0.0252, Accuracy: 9917/10000 (99.17%)

LR =  [0.01]
********* Epoch = 10 *********
loss=0.0770 batch_id=58: 100%|██████████| 59/59 [00:17<00:00,  3.44it/s]

Epoch 10: Train set: Average loss: 0.0672, Accuracy: 58675/60000 (97.79%)



Test set: Average loss: 0.0254, Accuracy: 9913/10000 (99.13%)

LR =  [0.01]
********* Epoch = 11 *********
loss=0.0579 batch_id=58: 100%|██████████| 59/59 [00:16<00:00,  3.48it/s]

Epoch 11: Train set: Average loss: 0.0628, Accuracy: 58757/60000 (97.93%)



Test set: Average loss: 0.0262, Accuracy: 9915/10000 (99.15%)

LR =  [0.001]
********* Epoch = 12 *********
loss=0.0592 batch_id=58: 100%|██████████| 59/59 [00:16<00:00,  3.53it/s]

Epoch 12: Train set: Average loss: 0.0527, Accuracy: 58990/60000 (98.32%)



Test set: Average loss: 0.0210, Accuracy: 9937/10000 (99.37%)

LR =  [0.001]
********* Epoch = 13 *********
loss=0.0425 batch_id=58: 100%|██████████| 59/59 [00:17<00:00,  3.45it/s]

Epoch 13: Train set: Average loss: 0.0503, Accuracy: 58993/60000 (98.32%)



Test set: Average loss: 0.0195, Accuracy: 9940/10000 (99.40%)

LR =  [0.001]
********* Epoch = 14 *********
loss=0.0426 batch_id=58: 100%|██████████| 59/59 [00:16<00:00,  3.57it/s]

Epoch 14: Train set: Average loss: 0.0477, Accuracy: 59060/60000 (98.43%)



Test set: Average loss: 0.0196, Accuracy: 9943/10000 (99.43%)

LR =  [0.001]
********* Epoch = 15 *********
loss=0.0560 batch_id=58: 100%|██████████| 59/59 [00:17<00:00,  3.44it/s]

Epoch 15: Train set: Average loss: 0.0471, Accuracy: 59043/60000 (98.41%)



Test set: Average loss: 0.0189, Accuracy: 9944/10000 (99.44%)

LR =  [0.001]
```
