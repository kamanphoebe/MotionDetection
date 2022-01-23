# Motion detection using optical flow

This project aims to investigate the feasibility of motion detection in the way of supervised learning, together with optical flow. Overall, optical flow of target objects are fed into neural network as inputs, while the outputs demonstrate the possibility of the objects' status, i.e., moving or still.</br>

The project is based on several open resources as listed below:</br>

- Dataset: [Nuscenes](https://github.com/nutonomy/nuscenes-devkit)
- Optical flow algorithms: [FastFlowNet](https://github.com/ltkong218/FastFlowNet), [Raft](https://github.com/princeton-vl/RAFT)
- Model: [ResNet18](https://arxiv.org/abs/1512.03385)

## Performance

Giving a target object, the model will output its possibility of status within [0, 1], where 0.5 is a threshold value. Objects with possibility greater than 0.5 will be predicted as moving and vice versa. Here are our classification scores of different optical flow algorithms:

|                     |   F1 (%)  |  Precision (%) |  Recall (%) | 
|:-------------------:|:---------:|:--------------:|:-----------:|
| FastFlowNet (Kitti) |    92.2   |      92.4      |     92.3    |
|    Raft (Kitti)     |    89.5   |      90.2      |     89.4    |

Video of visualization can be found [here](). The predictions are shown in different colors:
|                  Format           |   Prediction  |  Ground Truth |
|:---------------------------------:|:-------------:|:-------------:|
|    Red solid lines, red number    |     Moving    |     Moving    |
|  Blue dashed lines, blue number   |     Still     |     Still     |
|   Purple solid lines, red number  |     Moving    |     Still     |
|  Purple solid lines, blue number  |     Still     |     Moving    |

=====SOME IMAGES======

## Usage

The code has been tested with Python 3.8, PyTorch 1.6 and Cuda 10.2.

### Demo

If you would like to have a quick look about the viualization, you can simply run `source demo/demo.sh` to generate a `demo_visual.mp4` under the `demo` directory.

### Implement the complete pipeline

Some preperations are needed before starting.
- Run `source mkdir.sh` to create directories that will be needed.
- Download nuScenes dataset and install nuscenes-devkit. Fill in the installation path in `config.yml` (the key is `nuscnenes_path`).
- Clone the repository of FastFlowNet and/or RAFT and set them up. To create a conda environment needed for FastFlowNet, run `source env_fastflow.sh` to help.
- Move `flowgraph_fastflow.py` and `flowgraph_raft.py` to the root directory of FastFlowNet or RAFT, like `{FILE_PATH}/FastFlowNet/`.
- Modify the path of `config.yml` in `flowgraph_fastflow.py` and `flowgraph_raft.py`:

```python
# Please modify the path of config.yml
with open("{FILE_PATH}/config.yml") as f:
config = yaml.load(f, Loader=yaml.FullLoader)
```

Now you are ready to go!

1. `python scene_filter.py`</br>
Select target scenes in Nuscenes and save their tokens into `use_scene.json`.

2. `python select_dataset.py`</br>
Iterate over selected scenes to find frame pairs that contain target obj(s) and save their paths and tokens into `rawlst_train.json` or `rawlst_valid.json`.

3. `python flowgraph_fastflow.py` or `python flowgraph_raft.py --model=models/raft-kitti.pth` </br>
Generate optical flow graphs of selected raw images. </br>

4. `python generate_label.py` </br>
Estimate the velocity of an object using two frames and label it as 1(moving) or 0(still). All labels information are then saved into `labels_train.csv` and `labels_valid.csv`.

5. `python model_train.py` </br>
Find objects in frames and cut them off, followed by some preproccessing. Then fed the cutting pieces into the network to train.

6. `python model_visual.py` </br>
Visualize the prediction with the format mentioned above.

7. `ffmpeg -r 5 -pattern_type glob -i '{}/*.png' -b 8000k visual.mp4 -pix_fmt yuv420p` </br>
Generate a video.

8. Enjoy! ;)