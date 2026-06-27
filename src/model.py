# -*- coding: utf-8 -*-
"""
SegFormer-B0 Model for Binary Turf Segmentation
=================================================
Wraps HuggingFace SegFormer with a custom binary segmentation head.
Falls back to a lightweight U-Net if SegFormer download fails.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- SegFormer Wrapper -------------------------------------------------------

class TurfSegFormer(nn.Module):
    """
    SegFormer-B0 fine-tuned for binary segmentation (background vs turf).
    Uses HuggingFace transformers library.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation, SegformerConfig

        model_name = "nvidia/mit-b0"

        if pretrained:
            print(f"Loading pretrained SegFormer: {model_name}")
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    model_name,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                )
                print("[OK] SegFormer loaded from HuggingFace")
            except Exception as e:
                print(f"[!] HuggingFace download failed: {e}")
                print("    Falling back to random init SegFormer-B0...")
                config = SegformerConfig(
                    num_labels=num_classes,
                    hidden_sizes=[32, 64, 160, 256],
                    num_attention_heads=[1, 2, 5, 8],
                    depths=[2, 2, 2, 2],
                )
                self.model = SegformerForSemanticSegmentation(config)
        else:
            config = SegformerConfig(
                num_labels=num_classes,
                hidden_sizes=[32, 64, 160, 256],
                num_attention_heads=[1, 2, 5, 8],
                depths=[2, 2, 2, 2],
            )
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, pixel_values, labels=None):
        """
        Args:
            pixel_values: (B, 3, H, W)
            labels:       (B, H, W) int64 optional -- for computing loss
        Returns:
            logits: (B, num_classes, H, W) upsampled to input resolution
        """
        outputs = self.model(pixel_values=pixel_values, labels=labels)

        # SegFormer outputs at 1/4 resolution -- upsample back
        logits = outputs.logits  # (B, num_classes, H/4, W/4)
        H, W = pixel_values.shape[2], pixel_values.shape[3]
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        return logits


# --- Lightweight U-Net fallback ---------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class LightUNet(nn.Module):
    """
    Lightweight U-Net with ~2M parameters.
    CPU-friendly fallback if SegFormer is unavailable.
    """
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        f = [16, 32, 64, 128]

        self.enc1 = DoubleConv(in_channels, f[0])
        self.enc2 = DoubleConv(f[0], f[1])
        self.enc3 = DoubleConv(f[1], f[2])
        self.enc4 = DoubleConv(f[2], f[3])

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(f[3], f[3] * 2)

        self.up4  = nn.ConvTranspose2d(f[3]*2, f[3], 2, stride=2)
        self.dec4 = DoubleConv(f[3]*2, f[3])
        self.up3  = nn.ConvTranspose2d(f[3], f[2], 2, stride=2)
        self.dec3 = DoubleConv(f[2]*2, f[2])
        self.up2  = nn.ConvTranspose2d(f[2], f[1], 2, stride=2)
        self.dec2 = DoubleConv(f[1]*2, f[1])
        self.up1  = nn.ConvTranspose2d(f[1], f[0], 2, stride=2)
        self.dec1 = DoubleConv(f[0]*2, f[0])

        self.out  = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x, labels=None):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b  = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


# --- Loss Functions ----------------------------------------------------------

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1]  # turf class probability
        targets_f = targets.float()
        intersection = (probs * targets_f).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets_f.sum() + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """BCE + Dice combined loss (handles class imbalance well)."""
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce_loss   = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss


def build_model(use_segformer: bool = True):
    """Build and return the model. Falls back to U-Net if needed."""
    if use_segformer:
        try:
            model = TurfSegFormer(num_classes=2, pretrained=True)
            return model, "segformer"
        except Exception as e:
            print(f"[!] SegFormer failed: {e}. Using LightUNet instead.")
    model = LightUNet(in_channels=3, num_classes=2)
    print("[OK] LightUNet built")
    return model, "unet"
