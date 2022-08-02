# Optical Flow Based Motion Detection for Autonomous Driving

This project aims to investigate the feasibility of motion detection (especially for distant objects) in the way of supervised learning, together with optical flow. Overall, optical flow of target objects are fed into neural network as inputs, while the outputs demonstrate the confidence of the objects' status, i.e., moving or still. For details, please refer to the [associated paper](https://arxiv.org/abs/2203.11693) posted on arXiv.</br>

The project is based on several open resources as listed below:</br>

- Dataset: [nuScenes](https://github.com/nutonomy/nuscenes-devkit)
- Optical flow algorithms: [FastFlowNet](https://github.com/ltkong218/FastFlowNet), [Raft](https://github.com/princeton-vl/RAFT)
- Model: [ResNet18](https://arxiv.org/abs/1512.03385)

## Performance

Here are our classification scores of different optical flow algorithms:

|                     |   F1 (%)  |  Precision (%) |  Recall (%) | 
|:-------------------:|:---------:|:--------------:|:-----------:|
| FastFlowNet (Kitti) |    92.9   |      94.3      |     91.7    |
|    Raft (Kitti)     |    89.5   |      89.7      |     89.9    |

Videos of visualization can be found [here](https://www.youtube.com/playlist?list=PLVVrWgq4OrlBnRebmkGZO1iDHEksMHKGk). The predictions are shown in different colors: red boxes represents moving objects while blue boxes represents still objects.

![example1](/examples/img1.png)
![example2](/examples/img2.png)
![example3](/examples/img3.png)

## Usage

The code has been tested with Python 3.8, PyTorch 1.6 and Cuda 10.2.

### Demo

If you would like to have a quick try about the inference and visualization with nuScenes dataset, you can simply run the following commands to generate a `demo_visual.mp4`:
```bash
mkdir demo_infer
python demo_model_visual.py 
ffmpeg -r 2 -pattern_type glob -i './demo_infer/*.png' -pix_fmt yuv420p -b 8000k demo_visual.mp4
```

### Customized demo

The procedure below helps you apply our model on your **own** images of a video:
1. `sh custom_mkdir.sh`
2. Put all of your images under `custom_demo/custom_raw` directory. The images should be named in order according to time, e.g. `001.png`, `002.png`.
3. Generate the corresponding labels by yourself and save them into `custom_demo/custom_label.csv`. Please refer to `demo_label.csv` for the way of organizing labels. Note that `motionFlag` is actually not needed here.
4. Clone the repository of FastFlowNet and/or RAFT and set them up. To create a conda environment needed for FastFlowNet, run `source ./dev/env_fastflow.sh` to help.
5. Move `custom_fastflow.py` and `custom_raft.py` to the root directory of FastFlowNet or RAFT, like `{FILE_PATH}/FastFlowNet/`.
6. Run `python custom_fastflow.py --path REPO_PATH` or `python custom_raft.py --repo_path REPO_PATH  --model=models/raft-kitti.pth`, where `REPO_PATH` means the path of MotionDectection repository, to generate the optical flow data.
7. Run `python custom_model_visual.py` for inference and visualization.
8. Run `ffmpeg -r 2 -pattern_type glob -i './custom_demo/custom_infer/*.png' -pix_fmt yuv420p -b 8000k custom_visual.mp4` and you will get your own video!


### Implement the complete pipeline

Some preperations are needed before starting.
- `cd dev ; sh mkdir.sh`
- Download nuScenes dataset and install nuscenes-devkit.
- Clone the repository of FastFlowNet and/or RAFT and set them up. To create a conda environment needed for FastFlowNet, run `source env_fastflow.sh` to help.
- Move `flow_fastflow.py` and `flow_raft.py` to the root directory of FastFlowNet or RAFT, like `{FILE_PATH}/FastFlowNet/`.
- Follow the instructions to complete `config.yaml`. Also, modify the path of `config.yaml` in `flowgraph_fastflow.py` and `flowgraph_raft.py`:
```python
# Please modify the path of config.yaml
with open("{FILE_PATH}/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
```

Now you are ready to go!

1. `python scene_filter.py`</br>
Select target scenes from nuScenes and save their tokens into `use_scene.json`.

2. `python select_dataset.py`</br>
Iterate over selected scenes to find frame pairs that contain target obj(s) and save their paths and tokens into `rawlst_train.json` or `rawlst_valid.json`.

3. `python flow_fastflow.py` or `python flow_raft.py --model=models/raft-kitti.pth` </br>
Generate optical flow graphs of selected raw images. Note that the scripts are by default set to use FastFlowNet, so please replace the word "fastflow" with "raft" for the scripts in the following steps if you would like to use RAFT instead.

4. `python generate_label.py` </br>
Estimate the velocity of an object using two frames and label it as 1(moving) or 0(still). All labels information are then saved into `label_train.csv` and `label_valid.csv`.

5. `python model_train.py` </br>
Find objects in frames and cut them out, followed by some preproccessing. Then fed the cutting pieces into the network to train.

6. `python model_visual.py` </br>
Visualize the prediction with the format mentioned above.

7. `ffmpeg -r 2 -pattern_type glob -i 'visual/seq_20/*.png' -pix_fmt yuv420p -b 8000k visual.mp4`
Generate a video. Now you successfully obtain a video, but in a simpler form which only contains keyframes. To generate a complete video that includes non-keyframes and also near objects, there are several more steps:
```shell
mv flow_fastflow_expand.py {FILE_PATH}/FastFlowNet/
python {FILE_PATH}/FastFlowNet/flow_fastflow_expand.py
python generate_label_expand.py
python model_visual_expand.py
ffmpeg -r 12 -pattern_type glob -i 'visual_expand/scene_6/*.png' -pix_fmt yuv420p -b 8000k visual_expand.mp4
```

8. Enjoy! ;)

## Acknowledgement

Some of the scripts, namely `flow_fastflow.py`, `flow_fastflow_expand.py` and `flow_raft.py`, are based on the code of their original projects, FastFlowNet and RAFT.
