import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import albumentations.pytorch

import sys
import pyrootutils
root = pyrootutils.setup_root(__file__, pythonpath=True)
sys.path.append(root)

from workspace.src.filters.utils.utils_filter import *
from workspace.src.filters.filter_images.filter_glasses_image import FilterGlassesImage

class FilterGlassesVideo():
    def __init__(
        self,
        output_folder: str,
        filter_image: FilterGlassesImage
    ):
        self.output_folder = output_folder
        self.filter_image = filter_image

    def filter_glasses_video(
        self,
        input_path: str = None,
        output_name: str = None,
        new_width: int = None,
        new_height: int = None,
    ):
        cap = None
        if input_path is None:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(input_path)
        assert cap is not None

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if new_height is None and new_width is None:
            new_height = height
            new_width = width
        elif new_height is not None and new_width is None:
            ratio = width / height
            new_width = int(ratio * new_height)
        elif new_height is None and new_width is not None:
            ratio = height / width
            new_height = int(ratio * new_width)
        
        # total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # count = 0

        out = None
        if input_path is not None and output_name is not None:
            output_path = os.path.join(self.output_folder, output_name)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (new_width, new_height))

        prev_frame_gray = None
        prev_keypoints = None

        kalman = cv2.KalmanFilter(136, 68, 0)

        # Define state transition matrix
        kalman.transitionMatrix = np.eye(136, dtype=np.float32)

        # Define measurement matrix
        kalman.measurementMatrix = np.ones((68, 136), np.float32)
        # for i in range(68):
        #     kalman.measurementMatrix[i, i*2] = 1
        #     kalman.measurementMatrix[i, i*2+1] = 1

        # Define process noise covariance matrix
        # kalman.processNoiseCov = np.eye(136, dtype=np.float32) * 1e-3

        # Define measurement noise covariance matrix
        # kalman.measurementNoiseCov = np.ones((68, 68), dtype=np.float32) * 1e-1

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                # count += 1
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                faces = detect_faces(frame, self.filter_image.device)
                for face in faces:
                    top, right, bottom, left = face
                    width_face = right - left + 1
                    height_face = bottom - top + 1
                                
                    local_frame = frame[top: bottom + 1, left: right + 1].copy()
                    keypoints = get_keypoints(
                        image = local_frame,
                        model = self.filter_image.model,
                        transform = self.filter_image.transform,
                        normalize = False,
                        device = self.filter_image.device
                    )
                    keypoints += torch.Tensor([left, top]).to(self.filter_image.device)
                    keypoints_array = keypoints.detach().cpu().numpy()

                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_frame_gray is None and prev_keypoints is None:
                        prev_frame_gray = frame_gray.copy()
                        prev_keypoints = keypoints_array.copy()
                    
                    lk_params = dict(winSize=(101, 101), maxLevel=15,
                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
                    next_keypoints, status, error = cv2.calcOpticalFlowPyrLK(
                        prev_frame_gray, frame_gray, prev_keypoints, keypoints_array.copy(), **lk_params 
                    )
                    for k in range(keypoints_array.shape[0]):
                        d = cv2.norm(keypoints_array[k] - next_keypoints[k])
                        alpha = np.exp(-d * d / 50)
                        keypoints_array[k] = (1 - alpha) * keypoints_array[k] + alpha * next_keypoints[k]
                    # # keypoints_array = (keypoints_array + next_keypoints) / 2

                    # prediction = kalman.predict()
                    # print('prediction', prediction.shape)
                    # # measurement = kalman.measurementNoiseCov * np.random.randn(68, 68)
                    # # print('measurement1', measurement.shape)
                    # # mt = np.dot(kalman.measurementMatrix, keypoints_array.reshape(-1, 1))
                    # # print('mt', mt.shape)
                    # # measurement = np.dot(kalman.measurementMatrix, keypoints_array.reshape(-1, 1)) + measurement
                    measurement = np.dot(kalman.measurementMatrix, keypoints_array.reshape(-1, 1))
                    # print(measurement)
                    # print('measurement2', measurement.shape)
                    
                    
                    # # measurement = np.random.rand(68, 68).astype(dtype = np.float32)
                    # print(measurement.dtype)
                    kalman.correct(measurement)
                    # # process_noise = np.sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(136, 1)
                    # # print('process_noise', process_noise.shape)
                    # # keypoints_array = np.dot(kalman.transitionMatrix, keypoints_array.reshape(-1, 1)) + process_noise
                    keypoints_array = np.dot(kalman.transitionMatrix, keypoints_array.reshape(-1, 1))
                    # print('keypoints_array1', keypoints_array.shape)
                    keypoints_array = keypoints_array.reshape(-1, 2)
                    # print('keypoints_array2', keypoints_array.shape)

                    keypoints = torch.from_numpy(keypoints_array).to(self.filter_image.device)
                    
                    frame = self.filter_image.wear_glasses(
                        image = frame,
                        glasses_tensor = self.filter_image.glasses_tensor,
                        keypoints = keypoints
                    )

                    convexhull = cv2.convexHull(keypoints_array.reshape(-1, 1, 2))
                    keypoints_rect = cv2.boundingRect(convexhull)
                    (x_k, y_k, w_k, h_k) = keypoints_rect
                    print(keypoints_rect)
                    cv2.rectangle(frame, (x_k, y_k), (x_k + w_k, y_k + h_k), (0, 255, 0), 1)

                    # cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
                    frame = draw_image_with_keypoints(
                        image = Image.fromarray(frame),
                        keypoints= torch.from_numpy(keypoints_array),
                        width = frame.shape[1],
                        height = frame.shape[0],
                        normalize= False
                    )
                    frame = np.array(frame)
                    prev_keypoints = keypoints_array
                    prev_frame_gray = frame_gray
                if len(faces) == 0:
                    prev_keypoints = None
                    prev_frame_gray = None
                
                if input_path is None:
                    cv2.imshow('Frame', frame)
                    # Press Q on keyboard to  exit
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                elif out is not None:
                    out.write(frame)
                # print(f'Frame: {count}/{total_frame}')
            else:
                break
            
        if out is not None:
            out.release()
        cap.release()

if __name__ == '__main__':
    import hydra
    from omegaconf import DictConfig
    from src.models.components.filter_resnet import FilterResnet

    config_path = os.path.join(str(root), 'workspace/configs/filters')

    @hydra.main(config_path = config_path, config_name = 'filter_glasses_video.yaml')
    def main(cfg: DictConfig):
        pretrained_weight_path = os.path.join(str(root), 'workspace/inputs/pretrained_weights/weight_resnet50_1.pt')
        image_folder = os.path.join(str(root), 'workspace/inputs/images/filters')
        output_folder = os.path.join(str(root), 'workspace/outputs/videos/')
        input_path = os.path.join(str(root), 'workspace/inputs/videos/test_video.mp4')
        cfg.filter_image.image_folder = image_folder
        cfg.filter_image.model.pretrained_weight_path = pretrained_weight_path
        cfg.output_folder = output_folder

        filter_glasses_video = hydra.utils.instantiate(cfg)
        filter_glasses_video.filter_glasses_video(
            input_path = None,
            output_name = 'test_glasses_video6.mp4',
        )

    main()