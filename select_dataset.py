import numpy as np
import json
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
import yaml

with open("./config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Settings
sensor = config["sensor"]
use_category = config["use_category"]
use_visibility = config["use_visibility"]
scene_num_train = config["scene_num_train"]
min_dist = config["min_dist"]
max_dist = config["max_dist"]

use_scene_path = config["use_scene_path"]
nuscenes_path = config["nuscenes_path"]
rawlst_paths = config["rawlst_path"]


# Compute distance between box and ego.
def box_distance(sensor_data, ann_data):
    ego_location = nusc.get("ego_pose", sensor_data["ego_pose_token"])[
        "translation"]
    box_world_location = ann_data["translation"]
    distance = np.linalg.norm(
        np.array(box_world_location) - np.array(ego_location))
    return distance


with open(use_scene_path, "r") as f:
    use_scene = json.load(f)
img_seq = []
scene_lsts = []
scene_lsts.append(use_scene[:scene_num_train])  # [0, scene_num_train) for training set
scene_lsts.append(use_scene[scene_num_train:])  # [scene_num_train, last] for validation set
seq_head = True
found_box = False
nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_path, verbose=False)

# Iterate over selected scenes to record frame pairs
# that contain obj(s) which meet requirements.
for i, scene_lst in enumerate(scene_lsts):

    for scene in nusc.scene:

        if scene["token"] in scene_lst:

            sample = nusc.get("sample", scene["first_sample_token"])
            sensor_data = nusc.get("sample_data", sample['data'][sensor])
            curr_sensor_token = sensor_data["token"]

            # Iterate over all CAM_FRONT images in current scene (all frames)
            while curr_sensor_token:

                sensor_data = nusc.get("sample_data", curr_sensor_token)
                # Last frame of current scene.
                if not sensor_data["next"]:
                    found_box = False
                    seq_head = True
                    break
                # Ignore non-keyframe.
                if not sensor_data["is_key_frame"]:
                    curr_sensor_token = sensor_data["next"]
                    continue

                # Get boxes in CAM_FRONT of corresponding frame
                data_path, curr_boxlst, _ = nusc.get_sample_data(
                    curr_sensor_token, box_vis_level=BoxVisibility.ALL)

                for cbox in curr_boxlst:

                    if cbox.name not in use_category:
                        continue
                    annotation = nusc.get("sample_annotation", cbox.token)
                    visibility_token = annotation["visibility_token"]
                    if nusc.get("visibility", visibility_token)["level"] != use_visibility:
                        continue
                    distance = box_distance(sensor_data, annotation)
                    if distance >= min_dist and distance <= max_dist:
                        curr_instance_token = annotation["instance_token"]
                        next_sensor_token = sensor_data["next"]
                        next_sensor_data = nusc.get(
                            "sample_data", next_sensor_token)
                        while not next_sensor_data["is_key_frame"]:
                            next_sensor_token = next_sensor_data["next"]
                            if not next_sensor_token:
                                break
                            next_sensor_data = nusc.get(
                                "sample_data", next_sensor_token)

                        if not next_sensor_data:
                            break

                        next_data_path, next_boxlst, _ = nusc.get_sample_data(
                            next_sensor_data["token"], box_vis_level=BoxVisibility.ALL)

                        for nbox in next_boxlst:

                            next_annotation = nusc.get(
                                "sample_annotation", nbox.token)
                            next_instance_token = next_annotation["instance_token"]

                            # Same obj in two frames.
                            if next_instance_token == curr_instance_token:
                                next_visibility_token = next_annotation["visibility_token"]
                                if nusc.get("visibility", next_visibility_token)["level"] != use_visibility:
                                    break
                                next_sensor_data = nusc.get(
                                    "sample_data", next_sensor_token)
                                next_distance = box_distance(
                                    next_sensor_data, next_annotation)
                                if next_distance >= min_dist and next_distance <= max_dist:
                                    # If beginning of a new image sequence, record current frame also.
                                    if seq_head:
                                        img_seq.append(list())
                                        curr_lst = img_seq[-1]
                                        curr_lst.append(dict())
                                        curr_lst[-1]["imgpath"] = nusc.get_sample_data_path(
                                            curr_sensor_token)
                                        curr_lst[-1]["name"] = curr_sensor_token
                                        seq_head = False
                                    curr_lst = img_seq[-1]
                                    curr_lst.append(dict())
                                    curr_lst[-1]["imgpath"] = nusc.get_sample_data_path(
                                        next_sensor_token)
                                    curr_lst[-1]["name"] = next_sensor_token
                                    found_box = True
                                    break

                        if found_box:
                            break

                if not found_box:
                    seq_head = True

                found_box = False
                curr_sensor_token = sensor_data["next"]


    with open(rawlst_paths[i], 'w') as f:
        json.dump(img_seq, f, sort_keys=True, indent=4)
