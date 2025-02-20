from typing import List
import torch
from torch import nn
from torchmetrics.detection import MeanAveragePrecision


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCEWithLogitsLoss(reduction="none")
        self.mseloss = nn.MSELoss(reduction="none")
        self.weight_objectness = 1.0
        self.weight_class = 1.0
        self.weight_coordinates = 1.0

    def forward_objectness(self, objectness, gt_boxes_label, gt_and_neg_boxes_label):
        loss = self.bceloss(objectness, gt_boxes_label)
        loss = (loss * gt_and_neg_boxes_label).sum() / gt_and_neg_boxes_label.sum()
        return loss

    def forward_class(self, class_logits, classification_label, gt_boxes_label):
        num_gt_boxes = gt_boxes_label.sum()
        if num_gt_boxes == 0:
            return 0
        loss = self.bceloss(class_logits, classification_label)
        loss = (loss * gt_boxes_label).sum() / num_gt_boxes
        return loss

    def forward_coordinates(self, tx_ty_tw_th, coordinates_label, gt_boxes_label):
        num_gt_boxes = gt_boxes_label.sum()
        if num_gt_boxes == 0:
            return 0
        loss = self.mseloss(tx_ty_tw_th, coordinates_label)
        loss = (loss * gt_boxes_label).sum() / num_gt_boxes
        return loss

    def forward(self, preds, targets):
        """
        preds: dictionary of tensors output from the Yolo model
            {
                "tx_ty_tw_th": torch.Tensor, e.g. shape [10, 22743, 4]
                "objectness": torch.Tensor, e.g. shape [10, 22743, 1]
                "class_logits": torch.Tensor, e.g. shape [10, 22743, 80]
            }
        targets: dictionary of tensor labels
            {
                "gt_boxes_label": torch.Tensor, e.g. shape [10, 22743, 1]
                "gt_and_neg_boxes_label": torch.Tensor, e.g. shape [10, 22743, 1]
                "classification_label": torch.Tensor, e.g. shape [10, 22743, 80]
                "coordinates_label": torch.Tensor, e.g. shape [10, 22743, 4]
            }

        """

        objectness_loss = self.weight_objectness * self.forward_objectness(
            preds["objectness"],
            targets["gt_boxes_label"],
            targets["gt_and_neg_boxes_label"],
        )
        class_loss = self.weight_class * self.forward_class(
            preds["class_logits"],
            targets["classification_label"],
            targets["gt_boxes_label"],
        )
        coordinates_loss = self.weight_coordinates * self.forward_coordinates(
            preds["tx_ty_tw_th"],
            targets["coordinates_label"],
            targets["gt_boxes_label"],
        )
        loss = objectness_loss + class_loss + coordinates_loss
        return {
            "loss": loss,
            "objectness_loss": objectness_loss,
            "class_loss": class_loss,
            "coordinates_loss": coordinates_loss,
        }


class DetectionMetrics:
    """
    https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
    """

    def __init__(self, class_names: List[str]):
        # Use the default IOU thresholds of [0.5,...,0.95] with step 0.01
        self.metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
        self.class_names = class_names

    def update(self, preds, batch):
        """
        batch is a dictionary of tensors with keys
        {"image", "image_id", "boxes", "class_id", "iscrowd", etc.}

        preds is a list of dictionaries, each dictionary cooresponding to a batch item
        with keys {"boxes", "scores", "labels", etc}

        boxes must be XYXY format
        """
        targets = [
            {"boxes": boxes, "labels": labels, "iscrowd": iscrowd}
            for boxes, labels, iscrowd in zip(batch["boxes"], batch["class_idx"], batch["iscrowd"])
        ]
        self.metric.update(preds, targets)

    def compute(self):
        output = self.metric.compute()
        metrics = {
            "AP": output["map"].item(),
            "AP50": output["map_50"].item(),
            "AP75": output["map_75"].item(),
            "AP-large": output["map_large"].item(),
            "AP-medium": output["map_medium"].item(),
            "AP-small": output["map_small"].item(),
        }
        map_per_class = torch.zeros(len(self.class_names), dtype=torch.float)
        map_per_class[output["classes"]] = output["map_per_class"]
        for class_name, map_score in zip(self.class_names, map_per_class):
            metrics[f"AP-per-class/{class_name}"] = map_score.item()
        return metrics

    def reset(self):
        self.metric.reset()
