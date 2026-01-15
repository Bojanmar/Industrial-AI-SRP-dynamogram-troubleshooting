# src/models/hybrid.py
import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, n_classes: int, n_feat: int, dropout: float = 0.2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout),
        )
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)
        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )

        self.feat_net = nn.Sequential(
            nn.Linear(n_feat, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x_sig, x_feat):
        z = self.cnn(x_sig)
        z = self.cnn_pool(z)
        z = self.cnn_fc(z)
        f = self.feat_net(x_feat)
        h = torch.cat([z, f], dim=1)
        return self.classifier(h)
