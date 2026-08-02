"""
Fusion Model — Classification + Localization
------------------------------------------------
Takes the 10-channel fused input (RGB + ELA + SRM + DCT) and produces:
  1. Classification: Real / Photoshop-tampered / AI-tampered (3-class)
  2. Localization: pixel-level tampered-region mask (segmentation)

Architecture: a U-Net-style encoder-decoder with a shared encoder.
  - Encoder features feed a classification head (global pooling + FC)
  - Encoder features are also passed through a decoder for the mask output

This is intentionally kept simple (from-scratch conv blocks, not a heavy
pretrained backbone) so it is easy to understand, modify, and train fast
on a free-tier Colab/Kaggle GPU. Swapping in a pretrained ResNet/EfficientNet
encoder later is a straightforward upgrade — see the note at the bottom.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class FusionTamperNet(nn.Module):
    def __init__(self, in_channels: int = 10, num_classes: int = 3):
        super().__init__()

        # ---- Encoder ----
        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.bottleneck = conv_block(256, 512)

        # ---- Decoder (for localization mask) ----
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)

        self.mask_out = nn.Conv2d(32, 1, kernel_size=1)  # sigmoid applied in loss/inference

        # ---- Classification head (branches off the bottleneck) ----
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        # Classification branch
        class_logits = self.classifier(b)

        # Decoder branch (localization mask)
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        mask_logits = self.mask_out(d1)

        return class_logits, mask_logits


if __name__ == "__main__":
    # Sanity check: run a dummy batch through the model
    model = FusionTamperNet(in_channels=10, num_classes=3)
    dummy_input = torch.randn(2, 10, 256, 256)  # batch=2, 10 channels, 256x256
    class_logits, mask_logits = model(dummy_input)
    print(f"class_logits shape: {class_logits.shape}   (expected [2, 3])")
    print(f"mask_logits shape:  {mask_logits.shape}   (expected [2, 1, 256, 256])")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

# ---------------------------------------------------------------------------
# UPGRADE PATH: to use a pretrained backbone (recommended once the pipeline
# works end-to-end) — replace the encoder with e.g. torchvision.models.resnet34
# and adapt the first conv layer to accept 10 input channels instead of 3:
#
#   import torchvision.models as models
#   resnet = models.resnet34(weights="IMAGENET1K_V1")
#   old_conv = resnet.conv1
#   new_conv = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)
#   new_conv.weight.data[:, :3] = old_conv.weight.data  # copy pretrained RGB weights
#   resnet.conv1 = new_conv
# ---------------------------------------------------------------------------
