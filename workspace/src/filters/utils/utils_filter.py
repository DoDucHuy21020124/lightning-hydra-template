import cv2
import numpy as np
import torch
import torch_directml
import face_recognition as fc
from PIL import Image, ImageDraw

import sys
import pyrootutils
root = pyrootutils.setup_root(__file__, pythonpath=True)
sys.path.append(root)

from workspace.src.filters.utils.transform_workspace import MyTransform

def detect_faces(image: np.array, device) -> list:
    faces = []
    if device.type == 'cuda':
        faces = fc.face_locations(image, model = 'cnn')
    else:
        faces = fc.face_locations(image, model = 'hog')
    return faces

def get_keypoints(image: np.array,
                model: torch.nn.Module,
                transform = None,
                normalize: bool = True,
                device = None) -> torch.Tensor:
    if device is None:
        device = get_device()
    model = model.to(device)
    if transform is None:
        transform = MyTransform()
    
    keypoints = torch.squeeze(model(transform(image)[None, :, :, :].to(device)))
    if not normalize:
        keypoints = (keypoints + 0.5) * torch.Tensor([image.shape[1], image.shape[0]]).to(device)
    return keypoints

def get_device(device: str = 'cpu'):
    assert device in ['cpu', 'gpu', 'dml']
    if device == 'cpu':
        return torch.device('cpu')
    if device == 'gpu':
        assert torch.cuda.is_available()
        return torch.device('cuda')
    if device == 'dml':
        return torch_directml.device(torch_directml.default_device())

def find_rect(keypoints: np.array):
    convexhull = cv2.convexHull(keypoints.reshape(-1, 1, 2))
    rect = cv2.boundingRect(convexhull)
    return rect

def extract_index_nparray(nparray: np.array):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def draw_image_with_keypoints(
    image: torch.Tensor,
    keypoints: torch.Tensor,
    width: float = None,
    height: float = None,
    normalize: bool = False
) -> None:
    assert width is not None and height is not None
    if normalize:
        keypoints = (keypoints + 0.5) * torch.Tensor([width, height])
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for i in range(keypoints.shape[0]):
        if keypoints[i][0] is not None and keypoints[i][1] is not None: # and keypoints[i][0] >= 0 and keypoints[i][0] <= width and keypoints[i][1] >= 0 and keypoints[i][1] <= height:
            draw.ellipse((keypoints[i][0] - 2, keypoints[i][1] - 2, keypoints[i][0] + 2, keypoints[i][1] + 2), fill = (255, 255, 0))
    return image

def draw_image_with_keypoints_and_boundingbox(
    image: np.ndarray,
    box: list,
    keypoints: list,
    width: float = None,
    height: float = None,
    normalize: bool = False
) -> None:
    left_box = box[0]
    top_box = box[1]
    width_box = box[2]
    height_box = box[3]
    
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.rectangle((left_box, top_box, left_box + width_box, top_box + height_box), width = 2, outline = (0, 255, 0))
    print('image', type(image))
    image = FilterDataset.draw_image_with_keypoints(image, keypoints, width, height, normalize)
    return image

if __name__ == '__main__':
    device = get_device('cpu')
    print(device.type)