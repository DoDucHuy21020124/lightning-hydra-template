import torch
from torch import nn
from torchvision.models import get_model, get_model_weights, get_weight, list_models

class FilterResnet(torch.nn.Module):
    def __init__(
        self,
        model_name: str = 'resnet18',
        weights: str = 'DEFAULT',
        output_shape: list = [68, 2]
    ):
        super().__init__()

        supported = False
        model = get_model(model_name, weights = weights)
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(
                in_features=in_features, out_features=output_shape[0] * output_shape[1]
            )
            supported = True
        elif hasattr(model, 'heads'):
            heads = getattr(model, 'heads')
            if hasattr(model, 'head'):
                in_features = model.heads.head.in_features
                model.heads.head = torch.nn.Linear(
                    in_features=in_features, out_features=output_shape[0] * output_shape[1]
                )
                supported = True

        if not supported:
            print('This model does not support')
            exit(1)
        
        
        self.model = model
        self.output_shape = output_shape
    
    def forward(self, x):
        return self.model(x).reshape(x.shape[0], self.output_shape[0], self.output_shape[1])
    
if __name__ == '__main__':
    model = FilterResnet()
    x = torch.randn((3, 3, 224, 224))
    y = model(x)
    print(y.shape)

    # _ = FilterResnet()