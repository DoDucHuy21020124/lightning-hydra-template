from typing import Dict, List
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy import ndarray
import math
import torch
import omegaconf
import hydra
import pyrootutils

def init_dataframe(part: int) -> dict[str, List]:
    data = {
        'file_path': list(),
        'box': list()
    }
    for i in range(0, part):
        data[str(i)] = list()
    return data

def get_imagexml(file_path: str) -> List:
    data_xml = None
    with open(file_path, 'r') as f:
        data_xml = f.read()
    data_xml = BeautifulSoup(data_xml, 'xml')
    image_xml = data_xml.find_all('image')
    return image_xml

def get_data(image_xml: List, data: dict) -> dict[str, List]:
    for i in range(0, len(image_xml)):
        file_path = image_xml[i].get('file')
        data['file_path'].append(file_path)

        box = image_xml[i].find('box')
        box_features = list()
        box_features.append(float(box.get('left')))
        box_features.append(float(box.get('top')))
        box_features.append(float(box.get('width')))
        box_features.append(float(box.get('height')))
        data['box'].append(box_features)

        key_points = box.find_all('part')
        j = 0
        for k in range(len(key_points)):
            coordinates = list()
            name = key_points[k].get('name')
            if int(name) == j:
                coordinates.append(float(key_points[k].get('x')))
                coordinates.append(float(key_points[k].get('y')))
                j = j + 1
            else:
                coordinates = None
            data[str(k)].append(coordinates)
    return data

# def change_label_to_keypoints(label: torch.Tensor) -> List:
#     return label.reshape(-1, 2)

# def change_keypoints_to_label(keypoints: List) -> List:
#     label = list()
#     for i in range(len(keypoints)):
#         label.append(keypoints[i][0])
#         label.append(keypoints[i][1])
#     return label

# def draw_image_with_keypoints(
#         image: ndarray,
#         keypoints: torch.Tensor,
#         width: float = None,
#         height: float = None,
#         normalize: bool = False
#     ) -> None:
#     if normalize:
#         assert width is not None and height is not None
#         for i in range(keypoints.size()[0]):
#             keypoints[i][0] = (keypoints[i][0] + 0.5) * width
#             keypoints[i][1] = (keypoints[i][1] + 0.5) * height
#         #print(keypoints)
#     plt.imshow(image)
#     for i in range(keypoints.size()[0]):
#         if keypoints[i][0] is not None and keypoints[i][1] is not None and keypoints[i][0] >= 0 and keypoints[i][0] <= width and keypoints[i][1] >= 0 and keypoints[i][1] <= height:
#             plt.plot(float(keypoints[i][0]), float(keypoints[i][1]), marker = '.', markersize = 1, color = 'red')
#     plt.axis('off')

# def draw_image_with_keypoints_and_boundingbox(
#         image: ndarray,
#         box: List,
#         keypoints: List,
#         width: float = None,
#         height: float = None,
#         normalize: bool = False
#     ) -> None:
#     left_box = box[0]
#     top_box = box[1]
#     width_box = box[2]
#     height_box = box[3]

#     plt.gca().add_patch(Rectangle(
#         xy = (left_box, top_box),
#         width = width_box,
#         height = height_box,
#         fill = False,
#         edgecolor = 'green',
#         lw = 2
#     ))
#     draw_image_with_keypoints(image, keypoints, width, height, normalize)

# def draw_batch_image(
#         images: torch.Tensor,
#         labels: torch.Tensor,
#         width: float = None,
#         height: float = None,
#         normalize: bool = False
#     ) -> None:
#     fig = plt.figure(figsize = (4, 4))
#     plt.subplots_adjust(left=0,bottom=0,right=1,top=1, wspace=0, hspace=0)
#     number_image = images.size()[0]
#     row = int(math.sqrt(number_image))
#     col = None
#     if number_image % row == 0:
#         col = int(number_image / row)
#     else:
#         col = int(number_image / row) + 1
#     for i in range(images.size()[0]):
#         keypoints = labels[i].reshape(-1, 2)
#         image = images[i].squeeze().numpy().transpose(1, 2, 0)
#         fig.add_subplot(row, col, i + 1)
#         draw_image_with_keypoints(image, keypoints, width, height, normalize)
#     plt.show()

# def draw_whole_batch_image(batch_iteration):
#     while True:
#         images, labels = next(batch_iteration, "end")
#         if images == "end":
#             break
#         else:
#             draw_batch_image(images, labels)
        
# def instantiate_data(file_path: str):
#     root = pyrootutils.setup_root(__file__, pythonpath= True)
#     cfg = omegaconf.OmegaConf.load(root / "configs"/ "data"/ file_path)
#     cfg.data_dir = str(root / "data")
#     data = hydra.utils.instantiate(cfg)
#     return data

# def instantiate_transform(file_path: str):
#     root = pyrootutils.setup_root(__file__, pythonpath= True)
#     cfg = omegaconf.OmegaConf.load(root / "configs"/ "data"/ file_path)
#     data = hydra.utils.instantiate(cfg)
#     return data