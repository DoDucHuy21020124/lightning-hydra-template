import albumentations
import albumentations.pytorch

class MyTransform():
    def __init__(
        self,
        transform: albumentations.Compose = None,
        height: int = 224,
        width: int = 224,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225)
    ):
        print(type(transform))
        if transform is not None:
            self.transform = transform
        else:
            self.transform = albumentations.Compose([
                albumentations.Resize(height = height, width = width),
                albumentations.Normalize(mean = mean, std = std),
                albumentations.pytorch.ToTensorV2()
            ])
    def __call__(self, image):
        return self.transform(image = image)['image']