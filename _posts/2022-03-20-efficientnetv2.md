# EfficientNet vs. EfficientNetV2

1. TOC 
{:toc}

## Why EfficientNet?

Note: EfficientNet and EfficientNetV1 refer to the same thing

EfficientNet paper [here](https://arxiv.org/abs/1905.11946).

In the above fig-1, EfficientNets significantly outperform other ConvNets (Convolutional Neural Networks). EfficientNet-B7, which is the
EfficientNet with the highest number of Parameters, achieved new state of the art with 84.4% top-1 accuracy outperforming the previous SOTA
(State of the art), but being 8.4 times smaller and 6.1 times faster.

From the paper:

> EfficientNet-B7 achieves state- of-the-art 84.4% top-1 / 97.1% top-5 accuracy on ImageNet, while being 8.4x smaller and 6.1x
faster on inference than the best existing ConvNet. Our EfficientNets also transfer well and achieve state-of-the-art accuracy on
CIFAR-100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters.

## How does EfficientNet pull this off?

The main idea introduced in the research paper is **Compound Scaling**.

### Compound Scaling:

Keeping the network architecture constant, we can scale a DNN (Deep Neural Network) in one of 3 ways:


- Increasing the width (more channels)
- Increasing the depth (more convolutions)
- Increasing the resolution

In fig-2, EfficientNet V1 proposes Compound Scaling - combining the 3 different ways of scaling. Compound Scaling is basically to scale all
three dimensions while maintaining a balance between all dimensions of the network, i.e. maintaining a fixed ratio.

From the paper:

> In this paper, we want to study and rethink the process of scaling up ConvNets. In particular, we investigate the central
question: is there a principled method to scale up ConvNets that can achieve better accuracy and efficiency? Our empirical
study shows that it is critical to balance all dimensions of network width/depth/resolution, and surprisingly such balance can be
achieved by simply scaling each of them with constant ratio. Based on this observation, we propose a simple yet effective
compound scaling method. Unlike conventional practice that arbitrary scales these factors, our method uniformly scales network
width, depth, and resolution with a set of fixed scaling coefficients.

This main idea of Compound Scaling really set apart EfficientNets from its predecessors. And intuitively, this idea of compound scaling makes
sense too because if the input image is bigger (input resolution), then the network needs more layers (depth) and more channels (width) to
capture more fine-grained patterns on the bigger image.


From fig-3, we can see that the versions of MobileNet and ResNet architectures scaled using the Compound Scaling approach perform better
than their baselines.

The authors of EfficientNet architecture ran a lot of experiments scaling depth, width and image resolution and made two main observations:

> Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger
models. In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution
during ConvNet scaling.

Depth
```
Scaling network depth (number of layers), is the most common way used by many ConvNets.

With the advancements in deep learning (particularly thanks to Residual Connections, BatchNorm), it has now been possible to train deeper
neural networks that generally have higher accuracy than their shallower counterparts. The intuition is that deeper ConvNet can capture richer


and more complex features, and generalize well on new tasks. However, deeper networks are also more difficult to train due to the vanishing
gradient problem. Although residual connections and batchnorm help alleviate this problem, the accuracy gain of very deep networks diminishes.
For example, ResNet-1000 has similar accuracy as ResNet-101 even though it has much more layers.

In fig-4 (middle), we can also see that ImageNet Top-1 Accuracy saturates at d=6.0 and no further improvement can be seen after.

```
Width
```
Scaling network width - that is, increasing the number of channels in Convolution layers - is most commonly used for smaller sized models.

While wider networks tend to be able to capture more fine-grained features and are easier to train, extremely wide but shallow networks tend to
have difficulties in capturing higher level features.

Also, as can be seen in fig-4 (left), accuracy quickly saturates when networks become much wider with larger w.

```
Resolution
```
From the paper:

```
With higher resolution input images, ConvNets can potentially capture more fine-grained patterns. Starting from 224x224 in
early ConvNets, modern ConvNets tend to use 299x299 (Szegedy et al., 2016) or 331x331 (Zoph et al., 2018) for better
accuracy. Recently, GPipe (Huang et al., 2018) achieves state-of-the-art ImageNet accuracy with 480x480 resolution. Higher
resolutions, such as 600x600, are also widely used in object detection ConvNets (He et al., 2017; Lin et al., 2017).
```
fig-4 (right), we can see that accuracy increases with an increase in input image size.

By studying the individual effects of scaling depth, width and resolution, this brings us to the first observation which I post here again for
reference:

```
Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger
models.
```
Each dot in a line in fig-5 above denotes a model with different width(w). We can see that the best accuracy gains can be observed by
increasing depth, resolution and width.

r=1.0represents 224x224 resolution whereas r=1.3 represents 299x299 resolution.

Therefore, with deeper (d=2.0) and higher resolution (r=2.0), width scaling achieves much better accuracy under the same FLOPS cost.


This brings to the second observation:

```
In order to pursue better accuracy and efficiency, it is critical to balance all dimensions of network width, depth, and resolution
during ConvNet scaling.
```
## Developing a strong Baseline Model using Neural Architecture Search

The paper also demonstrated a strong mobile-size baseline using Compound Scaling and a search technique called Neural Architecture Search
or NAS.

### Neural Architecture Search (NAS):

This is a reinforcement learning based approach where the authors developed a baseline neural architecture Efficient-B0 by leveraging a multi-
objective search that optimizes for both Accuracy and FLOPS. From the paper:

```
Specifically, we use the same search space as (Tan et al., 2019), and use ACC(m)×[FLOPS(m)/T]w as the optimization goal,
where ACC(m) and FLOPS(m) denote the accuracy and FLOPS of model m, T is the target FLOPS and w=-0.07 is a
hyperparameter for controlling the trade-off between accuracy and FLOPS. Unlike (Tan et al., 2019; Cai et al., 2019), here we
optimize FLOPS rather than latency since we are not targeting any specific hardware device.
```
The EfficientNet-B0 architecture has been summarized in fig-6 below:

The MBConv layer will be discussed next.

Starting from this baseline architecture, the authors scaled the EfficientNet-B0 using Compound Scaling to obtain EfficientNet B1-B7.

### MBConv Layer (Inverted Bottleneck):


The key idea in a Bottleneck layer is to first use a 1x1 convolution to bring down the number of channels and apply the 3x3 or 5x5 convolution
operation to the reduced number of channels to get output features. Finally, use another 1x1 convolution operation to again increase the number
of channels to the initial value.

The inverted bottleneck as in MBConv does the reverse - instead of reducing the number of channels, the first 1x1 Conv layer increases the
number of channels to 3 times the initial.

Using a standard convolution operation here would be computationally expensive, so a Depthwise Convolution is used to get the output feature
map. Finally, the second 1x1conv layer downsamples the number of channels to the initial value. This has been illustrated in fig-8 below.


#### 1.

#### 2.

#### 3.

#### 1.

#### 2.

#### 3.

So to summarize, the EfficientNet-B0 architecture uses this inverted bottleneck with Depthwise Convolution operation. But, to this, they also add
squeeze and excitation operation.

From the paper:

```
The main building block of EfficientNet-B0 is mobile inverted bottleneck MBConv (Sandler et al., 2018; Tan et al., 2019), to
which we also add squeeze-and-excitation optimization (Hu et al., 2018).
```
### Scaling EfficientNet-B0 to get B1-B7:

From the paper:

```
Starting from the baseline EfficientNet-B0, we apply our compound scaling method to scale it up with two steps:
```
```
STEP 1: we first fix = 1, assuming twice more resources available, and do a small grid search of , ,. In particular, we find the
best values for EfficientNet-B0 are = 1.2, = 1.1, = 1.15, under constraint of *^2 *^2 2.
```
```
STEP 2: we then fix , , as constants and scale up baseline network with different , to obtain EfficientNet-B1 to B7.
```
### Conclusion of EfficientNet:

From the paper:

```
In this paper, we systematically study ConvNet scaling and identify that carefully balancing network width, depth, and resolution
is an important but missing piece, preventing us from better accuracy and efficiency. To address this issue, we propose a simple
and highly effective compound scaling method, which enables us to easily scale up a baseline Con- vNet to any target resource
constraints in a more principled way, while maintaining model efficiency. Powered by this compound scaling method, we
demonstrate that a mobile- size EfficientNet model can be scaled up very effectively, surpassing state-of-the-art accuracy with
an order of magnitude fewer parameters and FLOPS, on both ImageNet and five commonly used transfer learning datasets
```
### Drawbacks of EfficientNet:

```
Training with large images is slow. Need to decrease batch size to accommodate large images. This process is not optimal.
Depthwise convolutions are still expensive operations and slow in early layers.
Compound scaling equally scales up everything - doubling the depth doubles width and resolution equally. Plus, doubling depth
doubles depth in all stages of the network which leads to less precise control.
```
## EfficientNetV2:

EfficientNet V2 paper here.

EfficientNetV2 architecture was developed in exactly the same way as EfficientNet and the main contributions of the paper are:

```
Introduce new EfficientNetV2 architecture.
Proposed an improved method of progressive learning, which adjusts the regularization with image size.
Upto 11x faster training speed and 6.8x better parameter efficiency for EfficientNetV2 architecture as shown in fig-9 below.
```

### Overcoming with the Drawbacks of EfficientNet V1:

1. Training with very large images is slow

EfficientNet's large image size results in significant memory usage. Since the total memory on hardware devices such as GPU/TPU is fixed,
therefore, these models with larger image sizes need to be trained with smaller batch size. One simple improvement that can be used to fix this is
to use the methodology mentioned in Fixing the train-test resolution discrepancy (FixRes) paper where the main idea is to train on smaller on
image sizes and test on larger image sizes.

Using the FixRes technique mentioned above, we can observe in the fig-10 below, that using smaller image sizes leads to slightly better
accuracy.


2. Depthwise convolutions are slow in early layers

Fused-MBConv was proposed which replaces the depthwise 3x3 convolution and expansion 1x1 convolution in MBConv with a regular 3x
convolution as shown in Figure-2 above.

To compare these two building blocks and performance improvement, the authors of the EfficientNetV2 architecture gradually replaced the
original MBConv in EfficientNetV1 with Fused-MBConv and the results are shown in fig-11 below.

As can be seen in fig-11 above, it was observed that when Fused-MBConv is applied in the early layers (stages 1-3), this can lead to an
improvement in the training speed with a small overhead on parameters and FLOPs. However, if the authors replace all blocks with Fused-
MBConv, then it significantly increases parameters and FLOPs while also slowing down training speed.

Since, there is no simple answer as to when to utilize Fused-MBConv or MBConv, the authors leveraged Neural Architecture Search to search
for the best combination.

3. Equally scaling up every stage is sub-optimal

EfficientNetV1 architecture equally scales up all the stages using a simple compound scaling rule. However, these stages are not equally
contributed to the training speed and parameter efficiency. Therefore, for EfficientNetV2, the authors use a non-uniform scaling strategy to
gradually add more layers.

### Architecture:

The architecture for EfficientNetV2-S which was found using Neural Architecture Search (NAS) and the changes described above.


#### 1.

#### 2.

#### 3.

It can be observed that:

```
The EfficientNetV2 architecture extensively utilizes both MBConv and the newly added Fused-MBConv in the early layers.
EfficientNetV2 prefers small 3x3 kernel sizes as opposed to 5x5 in EfficientNetV1.
EfficientNetV2 completely removes the last stride-1 stage as compared to EfficientNetV1 (fig-6).
```
### Progressive Learning:

Progressive resizing was first introduced by Jeremy Howard. Progressive resizing’s main idea is to gradually increase the image size as the
training progresses. For example, if we are to train for a total of 10 epochs, then start out with image size 224x224 for the first 6 epochs and
then fine-tune using image size 256x256 for the last 4 epochs.

However, going from a smaller image size to a larger image size leads to an accuracy drop. The authors of EfficientNetV2 paper hypothesize that
this drop in accuracy comes from unbalanced regularization. When training with different image sizes, they mention that we should also
change the regularization strength accordingly.

To validate their hypothesis, they train a model with different image sizes and data augmentations as shown in fig-13 below.

It can be observed from fig-13 above, when the image size is small, it has the best accuracy with weak augmentation; but for larger image
size, it performs better with stronger augmentation. This observation motivated the authors to adaptively adjust regularization along with
image size during training - leading to this different method of Progressive Learning.

### Comparison of EfficientNetV2 to EfficientNet and other models:


As seen above in fig-14, the medium version of EfficientNetV2 performs better than the large version of EfficientNet V1-B7(85.1% vs.
84.7% accuracy on ImageNet). It is not only smaller (54M parameters vs. 66M parameters), but trains much much faster (13 hours vs. 139
hours). The inference time is also faster (57ms vs. 170ms).

### Final Summary

```
Criteria Which is better? EfficientNet EfficientNetV
```
```
Best accuracy score for medium-sized model higher is better 84.7% 85.1%
```
```
Number of Parameters (million) lower is better 66 54
```
```
FLOPs (billion) lower is better 38 24
```
```
Training time (hours) lower is better 139 13
```
```
Inference time (milli-seconds) lower is better 170 57
```
```
Feature/Technique EfficientNet EfficientNetV
```
```
Compound Scaling Introduced Improved using a non-uniform scaling
strategy
```
```
MBConv Implemented Improved using Fused-MBConv
```

#### 1.

#### 1.

```
Neural Architecture Search (NAS) Implemented, used to find the optimal
number of MBConv units
```
```
Used to improve the optimal number of
Fused-MBConv units
```
```
Progressive Learning - Introduced, inspired by Progressive Resizing
```
### Code Implementation

EfficientNetV2 can be implemented in a few lines of code:

## import tensorflow as tf

## import tensorflow_hub as hub

## def get_hub_url_and_isize(model_name, ckpt_type, hub_type):

## if ckpt_type == '1k':

## ckpt_type = ''

## else:

## ckpt_type = '-' + ckpt_type

## model_name = 'efficientnetv2-b0'

## ckpt_type = '1k'

## hub_type = 'feature-vector'

## batch_size = 32

## hub_url, image_size = get_hub_url_and_isize(model_name, ckpt_type,

## hub_type)

## # Define train_generator and valid_generator here

## tf.keras.backend.clear_session()

## # change input image size if necessary

## image_size = 300

## model = tf.keras.Sequential([

## tf.keras.layers.InputLayer(input_shape=[image_size, image_size, 3]),

## hub.KerasLayer(hub_url, trainable=True),

## tf.keras.layers.Dropout(rate=0.2),

## tf.keras.layers.Dense(train_generator.num_classes,

## kernel_regularizer=tf.keras.regularizers.l

## (0.0001))

## ])

## model.build((None, image_size, image_size, 3))

## model.summary()

### Useful Links:

```
Google Colab notebook on implementing fine-tuning of EfficientNetV2 on flowers dataset
```
```
References:
```

#### 1. https://amaarora.github.io/2020/08/13/efficientnet.html

#### 2. https://wandb.ai/wandb_fc/pytorch-image-models/reports/EfficientNetV2--Vmlldzo2NTkwNTQ

#### 3. https://arxiv.org/abs/1905.

#### 4. https://arxiv.org/abs/2104.

#### 5. https://colab.research.google.com/github/google/automl/blob/master/efficientnetv2/tfhub.ipynb#scrollTo=mmaHHH7Pvmth

#### 6. https://www.tensorflow.org/api_docs/python/tf/keras/applications/efficientnet_v2/EfficientNetV2M