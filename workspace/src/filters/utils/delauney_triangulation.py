import cv2
import face_recognition as fc
import torch
import csv
import os
import pandas as pd

import sys
import pyrootutils
root = pyrootutils.setup_root(__file__, pythonpath=True)
sys.path.append(root)

from workspace.src.filters.utils.utils_filter import *

class DelauneyTriangulation():
    def __init__(self, model: torch.nn.Module, folder_path: str):
        self.model = model
        self.folder_path = folder_path

    def save_triangle_csv(self, keypoints: np.array, filename: str): # keypoints.shape = [68, 2]
        rows = []
        keypoints = np.array(keypoints, dtype = np.int32)
        triangles = DelauneyTriangulation.get_triangles_list(keypoints)

        output_path = os.path.join(self.folder_path, filename + '.csv')
        with open(output_path, 'w', encoding='UTF8', newline= '') as f:
            writer = csv.writer(f)
            header = ['point1_x', 'point1_y', 'point2_x', 'point2_y', 'point3_x', 'point3_y',
                    'index_point1', 'index_point2', 'index_point3', 'filename']
            writer.writerow(header)
            for triangle in triangles:
                point1 = (triangle[0], triangle[1])
                point2 = (triangle[2], triangle[3])
                point3 = (triangle[4], triangle[5])

                index_point1 = np.where((keypoints == point1).all(axis = 1))
                index_point1 = extract_index_nparray(index_point1)

                index_point2 = np.where((keypoints == point2).all(axis = 1))
                index_point2 = extract_index_nparray(index_point2)

                index_point3 = np.where((keypoints == point3).all(axis = 1))
                index_point3 = extract_index_nparray(index_point3)

                if index_point1 is not None and index_point2 is not None and index_point3 is not None:
                    row = list(triangle) + [index_point1, index_point2, index_point3] + [filename + '.png']
                    rows.append(row)
                    writer.writerow(row)
        return rows

    def get_delauney_triangulation(self, image: np.array, filename: str):
        faces = fc.face_locations(image, model = 'hog')
        count = 0
        for face in faces:
            top, right, bottom, left = face
            face_location = image[top: bottom + 1, left: right + 1, :]

            keypoints = get_keypoints(image = face_location, model = self.model, normalize = False)
            keypoints += torch.Tensor([left, top])
            keypoints = keypoints.detach().numpy().astype(np.int32)
            filename_dup = filename
            if count > 0:
                filename_dup += str(count)
            _ = self.save_triangle_csv(keypoints = keypoints, filename = filename_dup)
            count += 1

    @staticmethod
    def get_triangles_list(keypoints: np.array):
        list_points = []
        for keypoint in keypoints:
            list_points.append((int(keypoint[0]), int(keypoint[1])))

        rect = find_rect(keypoints)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(list_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype = np.int32)
        return triangles

    @staticmethod
    def get_triangle_from_csv(input_path: str):
        df = pd.read_csv(input_path)
        
        triangles = df[['point1_x', 'point1_y', 'point2_x', 'point2_y', 'point3_x', 'point3_y']]
        triangles = np.array(triangles, dtype = np.int32)

        index_triangles = df[['index_point1', 'index_point2', 'index_point3']]
        index_triangles = np.array(index_triangles)

        filename = df['filename'].iloc[0]

        return triangles, index_triangles, filename

if __name__ == '__main__':
    import os
    from PIL import Image
    from src.models.components.filter_resnet import FilterResnet

    def main():
        root = pyrootutils.setup_root(__file__, pythonpath=True)
        folder_path = os.path.join(str(root), 'workspace/triangle_csv/')
        filename = 'le_hai_lam'
        pretrained_weight_path = os.path.join(str(root), 'outputs/weight_resnet50_1.pt')
        
        print(folder_path)
        print(filename)
        print(pretrained_weight_path)

        model = FilterResnet(model_name='resnet50')
        model.load_state_dict(pretrained_weight_path)

        image = Image.open('./ibug_300W_large_face_landmark_dataset/le_hai_lam.png')
        image = np.array(image)
        print(image.shape)
        delauney = DelauneyTriangulation(model, folder_path)
        delauney.get_delauney_triangulation(image = image, filename = filename)
    
    main()