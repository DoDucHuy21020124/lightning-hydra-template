import torch
from src.data.components.filter_dataset import FilterDataset
from collections import OrderedDict
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import albumentations
import cv2
import face_recognition as fc
import albumentations.pytorch
import pyrootutils
import os
from workspace.src.filters.utils.transform_workspace import MyTransform

def facial_landmark_video(
        model: torch.nn.Module,
        input_path: str = None,
        output_path: str = None,
        new_width: int = None,
        new_height: int = None,
        transform = None
    ):
    model = model.cuda()

    cap = None
    if input_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(input_path)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if new_height is None or new_width is None:
        new_height = height
        new_width = width
    else:
        ratio = width / height
        new_width = int(ratio * new_height)
    
    assert cap is not None
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0

    out = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (new_width, new_height))

    if transform is None:
        transform = MyTransform()

    while(cap is not None and cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            count += 1
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            face_locations = fc.face_locations(frame, model = 'cnn')
            for face_location in face_locations:
                top, right, bottom, left = face_location
                width_face = right - left + 1
                height_face = bottom - top + 1
                            
                local_frame = frame[top: bottom + 1, left: right + 1].copy()

                local_transformed = transform(local_frame)
                if torch.cuda.is_available():
                    local_transformed = local_transformed.cuda()

                keypoint = torch.squeeze(model(local_transformed[None, :, :, :]))
                keypoint = (keypoint + 0.5) * torch.Tensor([width_face, height_face]).cuda() + torch.Tensor([left, top]).cuda()

                frame = albumentations.pytorch.ToTensorV2()(image = frame)['image']

                frame = FilterDataset.draw_image_with_keypoints(
                    image = T.ToPILImage()(frame),
                    keypoints = keypoint,
                    width = frame.shape[2],
                    height = frame.shape[1],
                    normalize = False
                )

                frame = np.array(frame)
                

            # cv2.imshow('Frame',frame)
            
            if out is not None:
                out.write(frame)
            print(f'Frame: {count}/{total_frame}')

            # Press Q on keyboard to  exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     break
        else:
            break
        
    if out is not None:
        out.release()
    cap.release()

if __name__ == '__main__':
    from src.models.components.filter_resnet import FilterResnet

    def main():
        root = pyrootutils.setup_root(__file__, pythonpath=True)
        input_path = os.path.join(str(root), 'data/jacksparrow.mp4')
        output_path = os.path.join(str(root), 'outputs/output.mp4')
        pretrained_weight_path = os.path.join(str(root), 'outputs/weight_resnet50_1.pt')
        print(input_path)
        print(output_path)
        print(pretrained_weight_path)

        model = FilterResnet(model_name='resnet50')
        model.load_state_dict(pretrained_weight_path)

        facial_landmark_video(
            model = model,
            input_path = input_path,
            output_path = output_path,
        )
    
    print(torch.cuda.is_available())
    main()
