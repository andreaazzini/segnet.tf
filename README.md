# SegNet

SegNet is a TensorFlow implementation of the [segmentation network proposed by Kendall et al.](http://mi.eng.cam.ac.uk/projects/segnet/).

## Configuration

Before running, download the [VGG16 weights file](https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz)
and save it as `input/vgg16_weights.npz` if you want to initialize the encoder weights with the VGG16 ones trained on ImageNet classification dataset.

In `config.py`, choose your working dataset. The dataset name needs to match the data directories you create in your `input` folder.
You can use `segnet-32` and `segnet-13` to replicate the aforementioned Kendall et al. experiments.

## Train and test

Train SegNet with `python -m src/train.py`. Analogously, test it with `python -m src/test.py`. 
