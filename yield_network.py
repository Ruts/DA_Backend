import torch
import torch.nn as nn


class YieldPredictor(nn.Module):
    def __init__(self, soil_dim, img_dim=512):
        super().__init__()
        self.soil_net = nn.Sequential(
            nn.Linear(soil_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, soil, img):
        soil_feat = self.soil_net(soil)
        combined = torch.cat((soil_feat, img), dim=1)
        return self.fusion(combined)
