# SegNet
SegNet is a TensorFlow implementation of the [segmentation network proposed by Kendall et al.](http://mi.eng.cam.ac.uk/projects/segnet/), with cool features
like strided deconvolution, a minified architecture and more.

## Configuration
Create a `config.py` file, containing color maps, working dataset and other options.

```
autoencoder = 'segnet'
colors = {
  'segnet-32': [
    [64, 128, 64],   # Animal
    [192, 0, 128],   # Archway
    [0, 128, 192],   # Bicyclist
    [0, 128, 64],    # Bridge
    [128, 0, 0],     # Building
    [64, 0, 128],    # Car
    [64, 0, 192],    # CartLuggagePram
    [192, 128, 64],  # Child
    [192, 192, 128], # Column_Pole
    [64, 64, 128],   # Fence
    [128, 0, 192],   # LaneMkgsDriv
    [192, 0, 64],    # LaneMkgsNonDriv
    [128, 128, 64],  # Misc_Text
    [192, 0, 192],   # MotorcycleScooter
    [128, 64, 64],   # OtherMoving
    [64, 192, 128],  # ParkingBlock
    [64, 64, 0],     # Pedestrian
    [128, 64, 128],  # Road
    [128, 128, 192], # RoadShoulder
    [0, 0, 192],     # Sidewalk
    [192, 128, 128], # SignSymbol
    [128, 128, 128], # Sky
    [64, 128, 192],  # SUVPickupTruck
    [0, 0, 64],      # TrafficCone
    [0, 64, 64],     # TrafficLight
    [192, 64, 128],  # Train
    [128, 128, 0],   # Tree
    [192, 128, 192], # Truck_Bus
    [64, 0, 64],     # Tunnel
    [192, 192, 0],   # VegetationMisc
    [0, 0, 0],       # Void
    [64, 192, 0]     # Wall
  ]
}
gpu_memory_fraction = 0.7
strided = True
working_dataset = 'segnet-32'
```

Two kinds of architectures are supported at the moment: the original SegNet
Encoder-Decoder (`segnet`), and a smaller version of the same (`mini`), which
can be used for simpler segmentation problems. I suggest to use `strided = True`
for faster and more reliable results.

The `dataset_name` needs to match the data directories you create in your `input` folder. You can use `segnet-32` and `segnet-13` to replicate the aforementioned Kendall et al. experiments.

## Train and test
Generate your TFRecords using `tfrecorder.py`. In order to do so, put your PNG
images in a `raw` folder, as follows:

```
input/
    raw/
        train/
        train-labels/
        test/
        test-labels/

```

Once you have your TFRecords, train SegNet with `python src/train.py`. Analogously, test it with `python src/test.py`.
