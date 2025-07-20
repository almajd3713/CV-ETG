import torch
import torch.nn as nn


class TraditionalCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(TraditionalCNNModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3),
        )
        # Calculate the flattened size after the conv layers for 224x224 input
        # Each Conv2d uses kernel_size=3, stride=1, padding=0 (default)
        # Output size after each conv: (W - K + 1)
        # 224 -> 222 -> 220 -> 218 -> 216 -> 214
        # After 5 conv layers with kernel_size=3, stride=1, padding=0:
        # 224 -> 222 -> 220 -> 218 -> 216 -> 214
        # Final feature map: (batch, 256, 214, 214)
        self.flattened_size = 256 * 214 * 214
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 1024),
            nn.GELU(),
            nn.Linear(1024, num_classes)
        )
        self.bounding_box_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 512),
            nn.GELU(),
            nn.Linear(512, 4)  # Predicting 4 values for bounding box (x1, y1, x2, y2)
        )
        

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        bbox = self.bounding_box_predictor(x)
        return x, bbox

def get_model(num_classes):
    """
    Returns an instance of the TraditionalCNNModel with the specified number of classes.
    """
    return TraditionalCNNModel(num_classes=num_classes)