import torch
import torch.nn as nn
import torchvision.models as models

class EmoNet(nn.Module):
    def __init__(self, n_expression=8):
        super(EmoNet, self).__init__()
        self.n_expression = n_expression
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, n_expression)

    def forward(self, x):
        return self.backbone(x), None  # retornamos expresión y placeholder para compatibilidad
