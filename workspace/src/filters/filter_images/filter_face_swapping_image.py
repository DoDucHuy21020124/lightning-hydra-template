import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os

import sys
import pyrootutils
root = pyrootutils.setup_root(__file__, pythonpath=True)
sys.path.append(root)

from workspace.src.filters.utils.utils_filter import *
from workspace.src.filters.utils.delauney_triangulation import DelauneyTriangulation

class FilterFaceSwappingImage():
    def __init__(self, model: torch.nn.Module, image_folder: str, csv_folder: str, face1_csv: str, transform = None, device: str = 'cpu'):
        self.model = model
        self.csv_folder = csv_folder
        self.face1_csv = face1_csv
        self.image_folder = image_folder
        self.transform = transform
        self.device = get_device(device)

        self.triangles, self.index_triangles, self.filename = DelauneyTriangulation.get_triangle_from_csv(
            os.path.join(self.csv_folder, self.face1_csv)
        )

        # self.filter = cv2.imread(os.path.join(self.image_folder, self.filename), cv2.IMREAD_UNCHANGED)
        self.filter = Image.open(os.path.join(self.image_folder, self.filename))
        # plt.imshow(self.filter)
        # plt.show()
        # self.filter = cv2.cvtColor(self.filter, cv2.COLOR_BGR2RGB)
        # cv2.imshow('filter', self.filter)
        self.filter = np.array(self.filter)
        # cv2.imshow('cv2', self.filter)
        # cv2.waitKey(0)

    # def get_points_and_cropped_triangle(self, image: np.array, triangle: np.array):
    #     rect = cv2.boundingRect(triangle)
    #     (x, y, w, h) = rect
    #     cropped_triangle = image[y: y + h, x: x + w]
    #     cropped_triangle_mask = np.zeros((h, w), dtype = np.uint8)

    #     points = np.array([[triangle[0][0] - x, triangle[0][1] - y],
    #                         [triangle[1][0] - x, triangle[1][1] - y],
    #                         [triangle[2][0] - x, triangle[2][1] - y]], dtype = np.int32)
    #     cv2.fillConvexPoly(cropped_triangle_mask, points, 255)

    #     cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask = cropped_triangle_mask) 
    #     return points, cropped_triangle

    def filter_face_swapping_image(self, image: np.array) -> np.array:
        image = image.copy()
        faces = detect_faces(image, self.device)

        for face in faces:
            top, right, bottom, left = face
            face_location = image[top: bottom + 1, left: right + 1, :]
            keypoints = get_keypoints(
                image = face_location, model = self.model, transform = self.transform, normalize = False, device = self.device
            )
            keypoints += torch.Tensor([left, top]).to(self.device)
            keypoints = keypoints.detach().numpy().astype(np.int32)

            image = self.face_swapping(image, self.filter, keypoints)
        return image


    def face_swapping(self, image: np.array, filter: np.array, keypoints: np.array) -> np.array:
        image = image.copy()
        image_mask = np.zeros_like(image)
        for triangle, index_triangle in zip(self.triangles, self.index_triangles):
            # first face is filter, second face is real face
            # triangulation of the first face
            triangle1_point1 = [triangle[0], triangle[1]]
            triangle1_point2 = [triangle[2], triangle[3]]
            triangle1_point3 = [triangle[4], triangle[5]]
            triangle1 = np.array([triangle1_point1, triangle1_point2, triangle1_point3], dtype = np.int32)
            rect1 = cv2.boundingRect(triangle1)
            (x1, y1, w1, h1) = rect1
            cropped_triangle1 = filter[y1: y1 + h1, x1: x1 + w1]
            cropped_triangle1_mask = np.zeros((h1, w1), dtype = np.uint8)

            points1 = np.array([[triangle1_point1[0] - x1, triangle1_point1[1] - y1],
                                [triangle1_point2[0] - x1, triangle1_point2[1] - y1],
                                [triangle1_point3[0] - x1, triangle1_point3[1] - y1]], dtype = np.int32)
            cv2.fillConvexPoly(cropped_triangle1_mask, points1, 255)

            cropped_triangle1 = cv2.bitwise_and(cropped_triangle1, cropped_triangle1, mask = cropped_triangle1_mask) 
            # points1, cropped_triangle1 = self.get_points_and_cropped_triangle(filter, triangle1)

            #triangulation of the second face
            triangle2_point1 = keypoints[index_triangle[0]]
            triangle2_point2 = keypoints[index_triangle[1]]
            triangle2_point3 = keypoints[index_triangle[2]]
            triangle2 = np.array([triangle2_point1, triangle2_point2, triangle2_point3], dtype = np.int32)

            rect2 = cv2.boundingRect(triangle2)
            (x2, y2, w2, h2) = rect2
            cropped_triangle2 = image[y2: y2 + h2, x2: x2 + w2]
            cropped_triangle2_mask = np.zeros((h2, w2), np.uint8)

            points2 = np.array([[triangle2_point1[0] - x2, triangle2_point1[1] - y2],
                                [triangle2_point2[0] - x2, triangle2_point2[1] - y2],
                                [triangle2_point3[0] - x2, triangle2_point3[1] - y2]], np.int32)
            cv2.fillConvexPoly(cropped_triangle2_mask, points2, 255)
            # print(cropped_triangle2.shape, cropped_triangle2_mask.shape)
            cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2, mask = cropped_triangle2_mask)
            # points2, cropped_triangle2 = self.get_points_and_cropped_triangle(image, triangle2)

            # warp triangles
            points1 = np.float32(points1)
            points2 = np.float32(points2)

            M = cv2.getAffineTransform(points1, points2)
            warped_triangle = cv2.warpAffine(cropped_triangle1, M, (w2, h2), flags = cv2.INTER_CUBIC)

            # reconstruct destination face
            triangle_area = image_mask[y2: y2 + h2, x2: x2 + w2]
            triangle_area = cv2.add(triangle_area, warped_triangle)
            image_mask[y2: y2 + h2, x2: x2 + w2] = triangle_area

        face2_rect = find_rect(keypoints = keypoints)
        (x, y, w, h) = face2_rect

        image_mask_face = image_mask[y: y + h, x: x + w, :]
        # image_mask_face = cv2.medianBlur(image_mask_face, 5)
        # image_mask_face = cv2.GaussianBlur(image_mask_face, (5, 5), 0)
        image_mask_face_gray = cv2.cvtColor(image_mask_face, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('image_mask_gray', image_mask_face_gray)
        _, noise_mask = cv2.threshold(image_mask_face_gray, 254, 255, cv2.THRESH_BINARY)
        # print('noise_mask', noise_mask.shape)
        kernel = np.ones(shape = (3, 3), dtype = np.uint8)
        noise_mask = cv2.dilate(noise_mask, kernel, iterations=1)
        # cv2.imshow('noise_mask', noise_mask)
        image_mask_face = cv2.inpaint(image_mask_face, noise_mask, 5, cv2.INPAINT_TELEA)
        # cv2.imshow('inpaint', image_mask_face)
        image_mask_face = cv2.medianBlur(image_mask_face, 5)
        # image_mask_face = cv2.GaussianBlur(image_mask_face, (5, 5), 0)
        # cv2.imshow('image_mask_face', image_mask_face)
        image_mask[y: y + h, x: x + w, :] = image_mask_face
            
        image_mask_gray = cv2.cvtColor(image_mask, cv2.COLOR_RGB2GRAY)
        _, background = cv2.threshold(image_mask_gray, 1, 255, cv2.THRESH_BINARY)
        inverse_background = 255 - background

        image_no_face = cv2.bitwise_and(image, image, mask = inverse_background)
        image_with_new_face = cv2.add(image_no_face, image_mask)

        # seamless clone
        face2_center = (int(x + w/2), int(y + h/2))
        image = cv2.seamlessClone(image_with_new_face, image, background, face2_center, cv2.MIXED_CLONE)

        return image

if __name__ == '__main__':
    import os
    import hydra
    from omegaconf import DictConfig
    from workspace.src.model.filter_resnet import FilterResnet

    config_path = os.path.join(str(root), 'workspace/configs/filters/filter_image')
    image_folder = os.path.join(str(root), 'workspace/inputs/images/filters')
    csv_folder = os.path.join(str(root), 'workspace/inputs/csv_files')
    pretrained_weight_path = os.path.join(str(root), 'workspace/inputs/pretrained_weights/weight_resnet50_1.pt')
    image = Image.open('./workspace/inputs/images/filters/leonardo_dicarpio.jpg')
    image = np.array(image)
    print(image.shape)

    @hydra.main(version = '1.1', config_path=config_path, config_name = 'filter_face_swapping_image.yaml')
    def main(cfg: DictConfig):
        cfg.image_folder = image_folder
        print(type(cfg.model), cfg.model)
        cfg.csv_folder = csv_folder
        cfg.model.pretrained_weight_path = pretrained_weight_path
        print(cfg.model.pretrained_weight_path)

        filter_image = hydra.utils.instantiate(cfg)
        print(os.getcwd())

        image_face_swapping = filter_image.filter_face_swapping_image(image)
        # cv2.imshow('new_face', image_with_new_face[:, :, ::-1])
        # cv2.waitKey(0)
        plt.imshow(image_face_swapping)
        plt.show()

    main()
