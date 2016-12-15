# factors-variation
Code for the paper `Disentangling factors of variation in deep representations using adversarial training` by Mathieu, Zhao, Ramesh, Sprechmann and LeCun, NIPS 2016.

This code is a cleaned up version of our research code, and does not cover everything that is in the paper (yet). In particular, the code for the NORB dataset not there yet.

## Requirements

This code runs using torch. It is only tested on GPU. The following torch libraries are required:
nn, cunn, cudnn, optim, image, nngraph.

The following libraries are optional (but strongly recommended:
nnx, display.

We are using preprocessed versions of the dataset. You can download the mnist files [here](http://cs.nyu.edu/~mathieu/mnist.tgz) .
You need to create a folder data/mnist containing these files.

## Usage

We provide two model architectures: small and big. By default, the big architecture is used, it should work in all cases. The small architecture is faster to train and should work for MNIST and Sprites.

### MNIST

```
th train.lua --optimDisc adam
```
or
```
th train.lua --modelSize small --optimDisc adam
```

### Sprites

```
th train.lua --dataset sprites --nFeatures 14 --optimDisc adam
```

### yaleB

```
th train.lua --dataset yaleBExtended --swap2weight 1 --learningRateGen 0.0002 --learningRateDisc 0.0002
```
