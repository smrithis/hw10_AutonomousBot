# Self Driving Turtlebot 
This project implements a vision-based steering angle prediction model for autonomous navigation using the TurtleBot platform. <br>
It involves data collection via teleoperation, preprocessing of camera images, and training a PyTorch based regression model to predict steering commands from RGB images. The project has complicated procedures, please follow the read me and watch the workflow videos kept in the resource folder.

The model learns to predict the angular velocity (ω) command directly from the RGB image captured by the front camera of the TurtleBot.
It serves as a foundation for end-to-end autonomous steering control in a ROS2 environment.

## Software package

The workspace contains a folder named model_generation and a ROS package called self_driving.

Model_generation folder structure:

├── ckpt_best.pt &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# trained weight (initially might not be present) <br>
├── data		<br>
│   ├── processed <br>
│   │   └── merged_dataset &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# dataset name <br>
│   │       ├── images &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # folder contains images - gets populated <br>
│   │       ├── index_smooth.json &nbsp;&nbsp;# smoothed cmd_vel - gets populated <br>
│   │       └── index_split.json &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# raw cmd_vel - gets populated <br>
│   └── raw &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# contains the ROS2 bag file(s) <br>
├── dataset.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Class that loads dataset <br>
├── eval.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# script to test model <br>
├── model.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# neural net model <br>
├── runs &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# stores log for tensorboard <br>
├── train.ipynb &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# script to train the model <br>
└── util <br>
    ├── data_split.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# split the data randomly into train, test, val  <br>
    ├── extract_ros_bag.py	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# script to extract ros bag and dump images (*.png), cmd_vel (*.json) <br>
    ├── merge_dataset.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# merge dataset from different bags <br>
    ├── plot_data.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# plot cmd_vel for debugging <br>
    └── smooth_omega.py	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# smooth cmd_vel for better training and generalization <br>

Self_driving package structure:

├── Launch &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# trained weight (initially might not be present) <br>
├── resource		<br>
└── self_driving <br>
    ├──_init.py	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# split the data randomly into train, test, val  <br>
    ├── model.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# plot cmd_vel for debugging <br>
    └── steering_inference_image.py &nbsp;&nbsp;&nbsp;# smooth cmd_vel for better training and generalization <br>
├── package.xml &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# trained weight (initially might not be present) <br>
├── setup.config		<br>
└── setup.py <br>
 
       
## Software dependencies and Setup(in sequence):

```shell
$ sudo apt install python3-pip
$ pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
$ pip install tensorboard
$ pip install tqdm
$ sudo apt install ros-humble-cv-bridge
```

You will train the model using google colab with accelerated GPU enabled. The file for it is train.ipynb. You will have to upload it to the drive and link it to the colab notebook inorder for the script to be able to access your dataset. The instructions for this is provided in train.ipynb file, please follow it accordingly.

## Files to modify

**In the robot computer** <br>
We need to increase the framerate of the topic /cmd_vel. <br>
Open the following file:
```shell
$ nano turtlebot3_ws/src/turtlebot3/turtlebot3_teleop/turtlebot3_teleop/script/teleop_keyboard.py
```
**Modify line #89 <br>
rlist, _, _ = select.select([sys.stdin], [], [], 0.03) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # 0.03 = 33Hz originally 0.1 for 10 hz**


Compile and source subsequently

## Important tip

We assume that the the images (/image_raw/compressed) are arriving at 5Hz and cmd_vel at 33 Hz. without this rate, the regression training and testing will not work. 
You may also have to alter the camera position to get a good frame.

## Data collection:

The first step of the problem is to collect data for training. Data collection is an important step while training neural networks. 
Data is collected using ROS 2 bag recording while manually teleoperating the TurtleBot along the track.
Place the turtlebot on the track (blue lines) at the center. Launch the bringup and teleoperation node. 
Move the turtlebot along the track completing atleast 2 CCW laps and 2 CW laps. 
Record ros2 bag: inside the data/raw folder


- bag file 1 should have two laps for Counter Clockwise drive
- bag file 2 should have two laps for Clock Wises drive

Quality Guidelines:
- The blue track must be clearly visible in every frame.
- Minor occlusions at corners are acceptable, but ensure most of the track is always within the camera’s field of view.
- Refer to the provided “good data” reference video for visual examples of good recording quality.
 
Navigate to the folder /data/raw and run the below command while teleoperating.

```shell
$ ros2 bag record /image_raw/compressed /cmd_vel
```

## Dataset generation
Use the functions provided in the Util folder for dataset generation. 
Be sure to replace rosbag2_xxx and rosbag2_yyy with the correct name of your rosbags.
assuming we have two ros bags: rosbag2_xxx and rosbag2_yyy



1. Extract Images and Commands. Each ROS bag is processed to extract:
Camera frames from /image_raw/compressed
Velocity commands from /cmd_vel

```shell
$ python3 util/extract_ros_bag.py --bag data/raw/rosbag2_xxx --out data/processed/bag_1 --image-topic /image_raw/compressed --cmd-topic /cmd_vel
$ python3 util/extract_ros_bag.py --bag data/raw/rosbag2_yyy --out data/processed/bag_2 --image-topic /image_raw/compressed --cmd-topic /cmd_vel
```

2. Split Each Dataset. Split each bag into training and validation sets for reproducibility

```shell
$ python3 util/data_split.py --index data/processed/bag_1/index.json  --seed 123
$ python3 util/data_split.py --index data/processed/bag_2/index.json  --seed 123
```

3. Merge Datasets. Combine both directions (CW + CCW) into one unified dataset:

```shell
$ python3 util/merge_dataset.py --runs data/processed/bag_1 data/processed/bag_2 --out data/processed/merged_dataset --copy
```

4. Smooth Steering Values. Apply temporal smoothing to reduce noise in angular velocity (ω) commands:

```shell
$ python3 util/smooth_omega.py --index data/processed/merged_dataset/index_split.json
```

5. Visualize the Dataset. To inspect command distribution and verify data integrity:

```shell
$ python3 util/plot_data.py data/processed/merged_dataset/index_split.json
$ python3 util/plot_data.py data/processed/merged_dataset/index_smooth.json
```

You’ll see:
Histograms of ω command values
Sample image–command pairs
Smoothing effect comparisons	

## Training

The model is trained to learn a direct mapping from front camera images to steering commands (ω) using the processed and smoothed dataset.
It uses a CNN-based regression network optimized with MAE loss to minimize steering prediction error.

Model training is done using the provided Google Colab notebook, which includes all setup, dataset loading, and logging steps.
Navigate to train.ipynb file (>model_generation>train.ipynb) and follow the instructions in the notebook.

Once training completes, the best model checkpoint (ckpt_best.pt) will be saved automatically in your Drive. 

```
python3 -m tensorboard.main --logdir runs     
```
Execute this from the root directory of the project, ctrl + click on the **http://localhost:port_no** link to visualize the tensorboard

## Evaluate
After training, evaluate the model’s performance on the test split using the saved checkpoint.
Understand what the graphs represent and answer the corresponding questions.

You could also either run it through terminal as below:

```shell
python3 eval.py --index data/processed/merged_dataset/index_smooth.json --root  data/processed/merged_dataset --ckpt  ckpt_best.pt --split test --outdir eval_out --save-overlays --flip-sign --short-side 224 --top-crop 0.2
```

This will dump important metrics, plots and test images inside the eval_out folder 

## Testing
Before starting, place the TurtleBot centered on the track and ensure it faces the forward driving direction.
Once positioned, follow the steps below to deploy the trained model for autonomous steering.

## ROS2 Nodes

**Robot:**
- Robot bring up
```shell
$ ros2 launch turtlebot3_bringup robot.launch.py
```	
- Launch the camera node with QoS
```shell
$ ros2 run v4l2_camera v4l2_camera_node  --ros-args -p image_size:="[320, 240]" -p qos_overrides.image_raw.publisher.reliability:=best_effort -p qos_overrides.image_raw.publisher.history:=keep_last -p qos_overrides.image_raw.publisher.depth:=10 -p qos_overrides.image_raw.publisher.durability:=volatile
```

**Host computer:**
    
If any subscriber latches and reduces the frame rate, the inference loop will slow down — leading to unstable control.

```shell
$ ros2 topic hz /image_raw/compressed
$ ros2 run rqt_image_view rqt_image_view image:=/image_raw/compressed            ======> make sure the /image_raw/compressed frame rate does not drop
$ ros2 topic  hz /cmd_vel
$ ros2 topic echo /cmd_vel
$ v4l2-ctl -d /dev/video0 -p 5       # set the camera Hz to 5 Hz
$ ros2 launch self_driving steering_inference.launch.xml
```
✅ Expected Behavior at the end of the task -- The robot autonomously follows the blue track.



