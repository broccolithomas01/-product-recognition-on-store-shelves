"""
Task C — Stage 1: Object detection (Detectron2)
===============================================

On the full, densely packed dataset, keypoint matching is no longer robust.
The problem is split in two: detect first, classify second (see
`task_c_classification.py`).

This stage trains a single-class product detector:
    backbone : ResNet-50 + FPN  (COCO-pretrained)
    head     : RetinaNet         (COCO-pretrained, adapted to 1 class)
    fine-tune: SKU-110k (cluttered-shelf dataset)

SHOWCASE FILE: configuration + signatures; training body omitted.
"""

import os
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo


# --------------------------------------------------------------------------- #
# Custom trainer (COCO-style evaluation hook)
# --------------------------------------------------------------------------- #
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Return a COCOEvaluator for the single 'product' class."""
        ...  # implementation withheld


# --------------------------------------------------------------------------- #
# Configuration (RetinaNet R50-FPN, single class)
# --------------------------------------------------------------------------- #
def build_cfg():
    """Assemble the Detectron2 config: COCO-pretrained RetinaNet fine-tuned
    on SKU-110k to detect one generic product class on cluttered shelves."""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.RETINANET.NUM_CLASSES = 1   # single "product" class
    # ... dataset registration, solver schedule, LR, batch size, etc. (omitted)
    return cfg


def main():
    """Register SKU-110k splits, build the config and launch training."""
    ...  # implementation withheld


if __name__ == "__main__":
    main()
