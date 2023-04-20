# ScaleNet
### Setting up
Create a data directory and place the ground truth depth data folder and ground truth input from AdaBins it:
```
mkdir data
cd data
mkdir depth_gt
mkdir depth_input
```

Store the ground truth depth images to the `depth_gt` folder and the depth output from AdaBins to `depth_input` folder.

### Training
```
python3 ScaleNet.py
```
