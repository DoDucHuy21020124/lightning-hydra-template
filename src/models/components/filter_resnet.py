import torch
from torch import nn
from torchvision.models import get_model, get_model_weights, get_weight, list_models

class FilterResnet(torch.nn.Module):
    def __init__(
        self,
        model_name: str = 'resnet18',
        weights: str = 'DEFAULT',
        output_shape: list = [68, 2],
        pretrained_weight_path: str = None
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
        elif hasattr(model, 'classifier'):
            in_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(
                in_features = in_features, out_features = output_shape[0] * output_shape[1]
            )
            supported = True

        if not supported:
            print('This model does not support')
            exit(1)

        self.model = model
        self.output_shape = output_shape

        if pretrained_weight_path is not None:
            self.load_state_dict(pretrained_weight_path= pretrained_weight_path)

    def load_state_dict(self, pretrained_weight_path: str, strict: bool = True):
        pretrained_weight = torch.load(pretrained_weight_path)
        state_dict = pretrained_weight['model_state_dict']
        return super().load_state_dict(state_dict, strict)
    
    def forward(self, x):
        return self.model(x).reshape(x.shape[0], self.output_shape[0], self.output_shape[1])

    # def load_my_state_dict(self, state_dict):
    #     own_state = self.model.state_dict()
    #     for name, param in state_dict.items():
    #         if name not in own_state:
    #             continue
    #         if isinstance(param, nn.Parameter):
    #             param = param.data
    #         own_state[name].copy_(param)
    
if __name__ == '__main__':
    model = FilterResnet()
    x = torch.randn((3, 3, 224, 224))
    y = model(x)
    print(y.shape)

    # _ = FilterResnet()