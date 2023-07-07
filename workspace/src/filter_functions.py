import sys
import pyrootutils
root = pyrootutils.setup_root(__file__, pythonpath=True)
sys.path.append(root)

import os
import numpy as np

import hydra
from omegaconf import DictConfig

from workspace.src.filters.filter_images import *
from workspace.src.filters.filter_videos import *

filter_image_cfg = None
filter_video_cfg = None

config_path = os.path.join(str(root), 'workspace/configs')
@hydra.main(version_base = '1.1', config_path = config_path, config_name = 'filter_configs.yaml')
def filter_configs(cfg: DictConfig):
    # print(image)
    # print(filter_name)
    # if filter_name == 'glasses':
    #     filter_glasses_image(input, cfg.filters.filter_glasses_image)
    # if filter_name == '':
    #     pass
    # print(cfg)
    global filter_image_cfg
    global filter_video_cfg
    filter_image_cfg = cfg.filters.filter_image
    filter_video_cfg = cfg.filters.filter_video
    # print('filter_image_cfg', filter_image_cfg)
    # print('filter_video_cfg', filter_video_cfg)
    return list((filter_image_cfg, filter_video_cfg))

# filter_image_cfg, filter_video_cfg = filter_configs()
filter_configs()
print(filter_image_cfg)
print(filter_video_cfg)

glasses_image = hydra.utils.instantiate(filter_image_cfg.filter_glasses_image)
face_swapping_image = hydra.utils.instantiate(filter_image_cfg.filter_face_swapping_image)

glasses_video = hydra.utils.instantiate(filter_video_cfg.filter_glasses_video)
face_swapping_video = hydra.utils.instantiate(filter_video_cfg.filter_face_swapping_video)

def filter_glasses_image(image: np.array):
    # image = Image.open(image)
    # image = np.array(image)

    # config_path = os.path.join(str(root), 'workspace/configs')
    # @hydra.main(version_base = '1.1', config_path = config_path, config_name = 'filter_glasses_image.yaml')
    # def main(cfg: DictConfig):
    #     pass
    return glasses_image.filter_glasses_image(image)


def filter_face_swapping_image(image: np.array):
    return face_swapping_image.filter_face_swapping_image(image)

def filter_glasses_video(input_path):
    glasses_video.filter_glasses_video(input_path = input_path)

def filter_face_swapping_video(input_path):
    face_swapping_video.filter_face_swapping_video(input_path = input_path)

def filter_image(image, filter_name: str):
    # image = Image.open(image)
    image = np.array(image)
    print(image)
    if filter_name == 'eye_glasses':
        return filter_glasses_image(image)
    if filter_name == 'face_swapping':
        return filter_face_swapping_image(image)
    return None

def filter_video(input_path, filter_name: str):
    print(input_path)
    if filter_name == 'eye_glasses':
        filter_glasses_video(input_path)
    if filter_name == 'face_swapping':
        filter_face_swapping_video(input_path)

