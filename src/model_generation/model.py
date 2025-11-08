# model.py
import torch
import torch.nn as nn
from torchvision import models, transforms

class ImageOnlySteerNet(nn.Module):
    """
    Predicts future steering angles (omega_{t:t+4}) from a single RGB image.

    Inputs
    ------
    x: FloatTensor of shape [B, 3, H, W] in RGB order, range [0, 1] or [0, 255].

    Outputs
    -------
    y: FloatTensor of shape [B, 5] — future omegas.

    Notes
    -----
    - Uses a ResNet-18 encoder (global avg pooled to 512-D), then a small MLP head.
    - Set `freeze_backbone=True` for quick tuning of just the head.
    """
    def __init__(
        self,
        out_len: int = 5,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ---- Backbone (ResNet-18) ----
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = models.resnet18(weights=weights)

        # Replace the classifier with identity; we use pooled features (512-D)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ---- Regression head ----
        self.head = nn.Sequential(
            nn.Linear(in_feats, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, out_len),  # regression (no activation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            x = x.float() / 255.0
        feats = self.backbone(x)          # [B, 512]
        y = self.head(feats)              # [B, 51
        return y.squeeze(-1) # [B] only


# ---------- Recommended transforms ----------
def make_image_transforms(
    resize: int = 224,
    top_crop_ratio: float = 0.0,   # e.g., 0.15 crops top 15% (optional)
    normalize: bool = True,
):
    """
    Returns torchvision transforms for training/eval to feed ImageOnlySteerNet.

    - Keeps aspect via Resize(shorter side = resize), then CenterCrop(resize).
    - Optionally crops the top portion before resizing (helps remove sky/ceiling).
    """
    ops = []
    if top_crop_ratio > 0:
        # Crop out the top portion by ratio (expects PIL Image)
        class TopCrop(object):
            def __init__(self, r): self.r = r
            def __call__(self, im):
                h = im.height
                return im.crop((0, int(h * self.r), im.width, h))
        ops.append(TopCrop(top_crop_ratio))

    ops += [
        transforms.Resize(resize),             # shorter side -> resize
        transforms.CenterCrop(resize),         # square crop
        transforms.ToTensor(),                 # [0,1], RGB
    ]

    if normalize:
        # ImageNet normalization to match pretrained backbone
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        ops.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(ops)


# ---------- Tiny usage example ----------
if __name__ == "__main__":
    net = ImageOnlySteerNet(out_len=5, pretrained=True, freeze_backbone=False)
    dummy = torch.randn(2, 3, 224, 224)   # batch of 2 images
    out = net(dummy)
    print("Output shape:", out.shape)      # torch.Size([2])

