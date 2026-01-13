# README
This folder contains the geting started examples from Walrus. Here are some
notes on how to get things up and running.

## Files
`walrus_example_0_ConvertingDataIntoWellFormat.py`:
This script will download the walrus checkpoint and config. It also downloads
one of the bubbleML datasets to so that we can run on a non-Well dataset later. However, I had to alter the config file so I've included that in this git repository. 

`walrus_example_1_RunningWalrus.py`:
This script will run Walrus on the 2D radiative layer example from the well. See notes on Data to get that setup. You will also need to alter the `configs/extended_config.yaml` file

## Environment
`ffmpeg` is required in order to create the videos. Install (on ubuntu/devian):
```
sudo apt update
sudo apt install ffmpeg
```

## Data
Walrus is setup to use data formatted following The Well. I did not want to 
download the full 15TB dataset so I selected one of the smaller ones (2D turbulent radiative layer, 6.7GB):
```
the-well-download --dataset turbulent_radiative_layer_2D
```
