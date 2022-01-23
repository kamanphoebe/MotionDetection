import json
import random
import yaml

with open("./config.yml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

exclusion = config["exclusion"]  # Excluded scenes
allscene_json_path = config["nuscenes_path"] + "/v1.0-trainval/scene.json"
use_scene_path = config["use_scene_path"]

# Choose scenes that are not night and have normal weather.
f = open(allscene_json_path, mode='r')
scenes = json.load(f)

use_scene = []    # tokens of suitable scenes
descriptions = set()

for scene in scenes:
    flag = 0
    words = scene["description"].split(", ")
    words = [word.lower() for word in words]
    descriptions.update(words)
    for ex in exclusion:
        if ex in words:
            flag = 1
            break
    if not flag:
        use_scene.append(scene["token"])

f.close()

random.shuffle(use_scene)

with open(use_scene_path, "w") as f:
    json.dump(use_scene, f, sort_keys=True, indent=4)