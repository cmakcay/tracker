# Face Tracker using FAN<sup>[1]</sup> Landmarks and FLAME<sup>[2]</sup> model

This repository contains the face tracker which uses the approach in DECA, but simplified. The
face tracker predicts pose, shape and expression parameters of FLAME model. A trained model can be downloaded 
[here](https://drive.google.com/file/d/1xyIDFreHW_f_uBmAydBxmJFzeO4WdVfh/view?usp=sharing) and moved into "data/chkpts/" but the model can also be trained using this repo.


## Datasets
VGG-Face2 dataset is used to train the face tracker. <br />
Raw data is provided with face crop rectangles. 'dataset/pre_process.py' uses this face crops but modifies them according to
data cleaning step in DECA. The preprocessing script gets the FAN landmarks, and creates a csv file with coordinates of these 68 landmarks. The images are discarded if:<br />
1) The distance between the landmarks in the original face crop and modified face crop is larger than some threshold, as in DECA.
2) The image is either too large (makes CUDA run out of memory), or too small (not enough dimensions survive until the end of ResNet).

After the csv files with landmarks are created, they can be filtered based on their size in the same script. In DECA, they
use a mixture of larger than 224x224 and smaller than 224x224 images. In this implementation, only the larger ones are used.<br />
Also:
1) For training, train split of VGG-Face2 is used with 1800 individuals (n000002 to n001913).
2) For testing and validation, all test split of VGG-Face2 is used. For validation, 256 individuals (n000001
to n004679) and for testing, 244 individuals (n004709 to n009294) from test split are used. They can be separated manually
after using 'pre_process.py'.

Finally after the preprocessing step, csv files should be moved to data/landmarks as "train_list.csv" and "val_list.csv".

The preprocessed and separated landmarks can be directly downloaded from [here](https://drive.google.com/drive/folders/1nYYN1ZMLZnuSHhTLUS9BVwJfgQ42x6ut?usp=sharing) and moved into "data/landmarks" directory and there is no need to preprocess if these are used.

## Requirements
1. First, FLAME generic model should be downloaded from [FLAME website](https://flame.is.tue.mpg.de).
After signing in, download FLAME 2020 from 'Downloads' section and unzip it. Place 'generic_model.pkl' into 'data/flame_files'. Download "landmark_embedding.npy" from [DECA](https://github.com/YadiraF/DECA/tree/master/data) and move it to the "flame_files" as well.


2. In order to install the requirements, simply create a virtual environment, e.g. using venv. After activating the virtual environment, just run:
```
pip install -r requirements.txt
```
inside the main folder. The code is tested with Python 3.7.

## Train
Configuration settings for the training are in 'training options' section of 'config.py'. After arranging configs, training simply starts by:
```bash
$ python train.py
```

## Demo
For the demonstration of model, configuration settings are listed in 'demo options' section of 'config.py'. After setting
the configurations, demo starts by running:
```bash
$ python demo.py
```
The script can work with a video or webcam video real-time (with some delay). It uses MediaPipe's Face Detection to detect the face in the video, then approximates the model on this face crop. Note that it is designed to run with only one face in a video and the face should not be located at the edges of frames.

The model used for the demo is specified in, '--checkpoint_model_path'. 

1. For a live webcam demo without saving the output, the demo configs should be as follows: <br />
```bash
"--use_webcam": True
"--save_mesh_video": False
"--visualize_demo": True
```
2. To perform the demo on a recorded video and save the results, input and output video paths should be specified: <br />
```bash
"--demo_video_path": "demo/vid/demo_video.mp4"
"--output_video_path": "demo/vid/demo_mesh_result.mp4"
"--use_webcam": False
"--save_mesh_video": True
"--visualize_demo": False
```
If it is desired to visualize the results of each frame simutaneously, '--visualize_demo' can be set as True but it 
makes it much slower.

