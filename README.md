# NN vs CNN on IoT Firmware Image Classification Dataset

This is a repository about a comparison of neural network and convolutional neural network on IoT Firmware Image Classification dataset. In this repository, I have implemented the two experiments for the given dataset. The first experiment was implement as describe in the instruction document and for the second experiment I randomly choose a model for implementation. For both experiment, I showed the loss vs. epoch graph and the accuracy.


## Dataset

The given dataset is “IoT Firmware Image Classification”. You can download the dataset using

```http
  kaggle datasets download -d datamunge/iot-firmware-image-classification
```

Or here is link of the dataset: https://www.kaggle.com/datamunge/iot-firmware-image-classification . In the given dataset, the files and directory was distributed as:

- Firmware
    - Firmware.csv
- Imagery
    - Benignware
    - Gray
    - Hackware
    - Malware

Here I only work with the image dataset that was in imagery directory. It contain about 4482 images where benignware has 2999 images, gray has 669 images, hackware has 103 images and malware has 711 images. In the code you can see full visualization of the of the dataset.


## Pre-processing Steps
In the pre-processing part I transform and resize into 32 x 32 so that all of the images in 
the dataset are same size. I used “torchvision” “ImageFolder” and “DataLoader” to load 
the dataset with batch size of 20. DataLoader pre-processes the dataset for the model in 
pytorch.


## Neural Network

The model that is used is given below:

![a](https://user-images.githubusercontent.com/43477718/153032968-9f682966-edb3-4014-b447-228c0c0a9022.png)

## Convolutional Neural Network

1. nn.Conv2d(in_channels=3, out_channels=256,kernel_size=3, stride=1, padding=1)
2. nn.Conv2d(in_channels=256, out_channels=128,kernel_size=3, stride=1, padding=1)
3. nn.Conv2d(in_channels=128, out_channels=64,kernel_size=3, stride=1, padding=1)
4. nn.Conv2d(in_channels=64, out_channels=32,kernel_size=3, stride=1, padding=1)
5. flatten()
6. self.linear_layer_1 = nn.Linear(128, 512)
7. self.linear_layer_2 = nn.Linear(512, 128) 
8. self.linear_layer_3 = nn.Linear(128, 64) 
9. self.linear_layer_4 = nn.Linear(64, 32)
10. self.linear_layer_5 = nn.Linear(32, 4)


## Experiment
For the given dataset I conducted two experiment.

### Experiment 1:
- Batch size: 20
- Loss function: torch.nn.functional.F.nll_loss
- Optimizer: Adam Optimizer
- Learning Rate: 0.001
- Epoch: 20

### Experiment 2:
- Batch size: 20
- Loss function: torch.nn.functional.F.nll_loss
- Optimizer: Adam Optimizer
- Learning Rate: 0.001
- Epoch: 20


## Author

- [@RahatKaderKhan](https://github.com/rahatkader)
