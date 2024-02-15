import logging

import bpy
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np
parser = kb.ArgumentParser()
parser.add_argument("--objects_split", choices=["train", "test"],
                    default="train")
# Configuration for the objects of the scene
parser.add_argument("--min_num_objects", type=int, default=3,
                    help="minimum number of objects")
parser.add_argument("--max_num_objects", type=int, default=5,
                    help="maximum number of objects")
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--backgrounds_split", choices=["train", "test"],
                    default="train")

parser.add_argument("--camera", choices=["fixed_random", "linear_movement", "static"],
                    default="static")
parser.add_argument("--max_camera_movement", type=float, default=4.0)


# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str,
                    default="gs://kubric-public/assets/GSO/GSO.json")
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.add_argument("--num_scene", type=int, default=1)
parser.add_argument("--output_prefix", type=str, default="sample")
parser.add_argument("--z_elevation", type=float, default= np.pi / 9)
parser.add_argument("--num_camera", type=int, default=8)
parser.set_defaults(save_state=False, frame_end=24, frame_rate=12,
                    resolution=256)

FLAGS = parser.parse_args()
kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)
train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
# Add random objects
train_split, test_split = gso.get_test_split(fraction=0.1)
if FLAGS.objects_split == "train":
    logging.info("Choosing one of the %d training objects...", len(train_split))
    active_split = train_split
else:
    logging.info("Choosing one of the %d held-out objects...", len(test_split))
    active_split = test_split
    
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
for i in range(len(active_split)):
    obj = gso.create(asset_id=active_split[i])
    print(np.max(obj.bounds[1]-obj.bounds[0]))