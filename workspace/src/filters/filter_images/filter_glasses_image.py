import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import albumentations
import albumentations.pytorch
import torchvision.transforms as T

import sys
import pyrootutils
root = pyrootutils.setup_root(__file__, pythonpath=True)
sys.path.append(root)

from workspace.src.filters.utils.utils_filter import *

class FilterGlassesImage():
    def __init__(self, model: torch.nn.Module, image_folder: str, filter_name: str, transform = None, device: str = 'cpu'):
        self.model = model
        self.image_folder = image_folder
        self.filter_name = filter_name
        self.device = get_device(device)
        self.transform = transform

        # self.glasses = cv2.imread(os.path.join(self.image_folder, self.filter_name), cv2.IMREAD_UNCHANGED)
        # self.glasses = cv2.cvtColor(self.glasses, cv2.COLOR_BGRA2RGBA)
        self.glasses = Image.open(os.path.join(self.image_folder, self.filter_name))
        self.glasses = np.array(self.glasses)
        self.glasses_tensor = albumentations.pytorch.ToTensorV2()(image = self.glasses)['image'].to(self.device)

    def filter_glasses_image(self, image: np.array) -> np.array:
        image = image.copy()

        faces = detect_faces(image, self.device)
        for face in faces:
            top, right, bottom, left = face
            face_location = image[top: bottom + 1, left: right + 1, :]
            keypoints = get_keypoints(
                image = face_location, model = self.model, transform = self.transform, normalize = False, device = self.device
            )
            keypoints += torch.Tensor([left, top]).to(self.device)

            image = self.wear_glasses(image, self.glasses_tensor, keypoints)
        
        return image

    def wear_glasses(self, image: np.array, glasses_tensor: torch.Tensor, keypoints: torch.Tensor) -> np.array:
        image = image.copy()
        left_eye_end = keypoints[36]
        left_eye_start = keypoints[39]

        right_eye_end = keypoints[45]
        right_eye_start = keypoints[42]

        top_nose = keypoints[27]

        top_eye = keypoints[44]
        bottom_eye = keypoints[41]

        glasses_ratio = glasses_tensor.shape[1] / glasses_tensor.shape[2]

        width_glasses = int(torch.sqrt((right_eye_end[0] - left_eye_end[0]) * (right_eye_end[0] - left_eye_end[0]) + (right_eye_end[1] - left_eye_end[1]) * (right_eye_end[1] - left_eye_end[1])) * 1.5)
        height_glasses = int(width_glasses * glasses_ratio)

        glasses_tensor_resize = T.Resize(size = (height_glasses, width_glasses))(glasses_tensor)

        angle = float(torch.rad2deg(torch.atan((left_eye_end[1] - right_eye_end[1]) / (right_eye_end[0] - left_eye_end[0]))))
        glasses_tensor_resize = T.functional.rotate(glasses_tensor_resize, angle, expand = True)

        _, height_glasses, width_glasses = glasses_tensor_resize.shape

        glasses_x = int(top_nose[0]) - int((width_glasses - 1) / 2)
        glasses_y = int(top_nose[1]) - int((height_glasses - 1) / 2)
        if width_glasses % 2 == 0:
            if right_eye_end[0] - left_eye_end[0] >= 2 * top_nose[0]:
                glasses_x = int(top_nose[0]) - int((width_glasses - 2) / 2)
            else:
                glasses_x = int(top_nose[0]) - int(width_glasses / 2)
        if height_glasses % 2 == 0:
            if bottom_eye[1] - top_eye[1] >= 2 * top_nose[1]:
                glasses_y = int(top_nose[1]) - int((height_glasses - 2) / 2)
            else:
                glasses_y = int(top_nose[1]) - int(height_glasses / 2)

        if glasses_x < 0:
            glasses_tensor_resize = glasses_tensor_resize[:, :, -glasses_x:]
            width_glasses += glasses_x
            glasses_x = 0
        if glasses_y < 0:
            glasses_tensor_resize = glasses_tensor_resize[:, -glasses_y:, :]
            height_glasses += glasses_y
            glasses_y = 0
        if glasses_x + width_glasses > image.shape[1]:
            width_glasses -= (glasses_x + width_glasses - image.shape[1])
            glasses_tensor_resize = glasses_tensor_resize[:, :, 0: width_glasses]
        if glasses_y + height_glasses > image.shape[0]:
            height_glasses -= (glasses_y + height_glasses - image.shape[0])
            glasses_tensor_resize = glasses_tensor_resize[:, 0: height_glasses, :]

        local = image[glasses_y: glasses_y + height_glasses, glasses_x: glasses_x + width_glasses, :].copy()
        local = albumentations.pytorch.ToTensorV2()(image = local)['image'].to(self.device)

        weight = glasses_tensor_resize[3, :, :] / 255

        for channel in range(3):
            local[channel, :, :] = (1 - weight) * local[channel, :, :] + weight * glasses_tensor_resize[channel, :, :]
            
        image[glasses_y: glasses_y + height_glasses, glasses_x: glasses_x + width_glasses, :] = local.permute(1, 2, 0).cpu().numpy()
    
        return image
    
if __name__ == '__main__':
    import os
    import hydra
    from omegaconf import DictConfig
    from workspace.src.model.filter_resnet import FilterResnet

    config_path = os.path.join(str(root), 'workspace/configs/filters/filter_image/filter_glasses_image')
    image_folder = os.path.join(str(root), 'workspace/inputs/images/filters')
    pretrained_weight_path = os.path.join(str(root), 'workspace/inputs/pretrained_weights/weight_resnet50_1.pt')
    model = FilterResnet(model_name= 'resnet50', pretrained_weight_path= pretrained_weight_path)
    transform = MyTransform()
    image = Image.open('./workspace/inputs/images/filters/leonardo_dicarpio.jpg')
    image = np.array(image)
    print(image.shape)

    @hydra.main(config_path=config_path, config_name = 'filter_glasses_image.yaml')
    def main(cfg: DictConfig):
        cfg.image_folder = image_folder
        cfg.model = model
        cfg.transform = transform
        # print(type(cfg.model), cfg.model)
        # cfg.model.pretrained_weight_path = pretrained_weight_path
        # print(cfg.model.pretrained_weight_path)

        filter_image = hydra.utils.instantiate(cfg)
        print(os.getcwd())

        image_with_glasses = filter_image.filter_glasses_image(image)
        # cv2.imshow('new_face', image_with_new_face[:, :, ::-1])
        # cv2.waitKey(0)
        plt.imshow(image_with_glasses)
        plt.show()

    main()