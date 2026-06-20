"""
Task C — Stage 2: Classification (CLIP + EfficientNet-B0)
========================================================

Every detected crop is labelled by two networks that progressively narrow the
decision over the 27-class catalogue:

    CLIP (zero-shot)        -> top-2 most likely BRANDS for the crop
    EfficientNet-B0         -> scores for all 27 classes
    masked softmax          -> suppress classes outside CLIP's top-2 brands,
                               re-normalise, pick the final class

Rationale
---------
EfficientNet-B0 alone tends to give flat scores across visually similar classes
(small synthetic dataset, small network). CLIP restricts the decision space and
stabilises the result; the trade-off is that masking narrows ambiguity rather
than removing it.

SHOWCASE FILE: high-level pipeline + signatures; bodies omitted.
"""

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Crop / drawing helpers
# --------------------------------------------------------------------------- #
def crop_with_margin(img_bgr: np.ndarray, box_xyxy, margin: float = 0.05): ...
def draw_label_box(img, box_xyxy, text, color=(0, 255, 0)): ...
def label_to_color(label: str): ...


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def load_classifier(ckpt_path: str, num_classes: int, device: str):
    """Load EfficientNet-B0 with its final layer adapted to `num_classes`."""
    ...

def clip_top2_brands(clip_model, clip_processor, crop_rgb: np.ndarray,
                     brand_names, device: str):
    """Zero-shot CLIP: return the 2 most likely brand names for a crop."""
    ...

def mask_logits_to_allowed(logits_1d: torch.Tensor, allowed_idx):
    """Keep only logits whose class belongs to CLIP's allowed brands;
    push everything else to a very low value before softmax."""
    ...


# --------------------------------------------------------------------------- #
# Main pipeline (orchestration)
# --------------------------------------------------------------------------- #
def classify_box(crop_rgb, clip_model, clip_processor, cnn_model,
                 brand_names, brand_to_class_idx, device):
    """Fuse CLIP (top-2 brands) and EfficientNet-B0 (27-class logits) via a
    masked softmax, returning the final class for one detected box."""
    # 1) CLIP narrows the brand to a top-2 shortlist
    top_brands = clip_top2_brands(clip_model, clip_processor, crop_rgb,
                                  brand_names, device)
    allowed_idx = [i for b in top_brands for i in brand_to_class_idx[b]]

    # 2) EfficientNet-B0 scores all 27 classes
    logits = cnn_model(_preprocess(crop_rgb).to(device)).squeeze(0)

    # 3) mask to allowed classes -> softmax -> argmax
    masked = mask_logits_to_allowed(logits, allowed_idx)
    probs  = torch.softmax(masked, dim=0)
    return int(torch.argmax(probs))


def _preprocess(crop_rgb):
    """Resize / normalise a crop for EfficientNet-B0."""
    ...  # implementation withheld
