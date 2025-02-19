import evaluate
from torch import nn


class CombinedEvaluations(evaluate.CombinedEvaluations):
    """
    Extended this class https://github.com/huggingface/evaluate/blob/v0.4.3/src/evaluate/module.py#L872
    """

    def compute(self, predictions=None, references=None, **kwargs):
        results = []
        kwargs_per_evaluation_module = {
            name: {} for name in self.evaluation_module_names
        }
        for key, value in kwargs.items():
            if key not in kwargs_per_evaluation_module:
                for k in kwargs_per_evaluation_module:
                    kwargs_per_evaluation_module[k].update({key: value})
            elif key in kwargs_per_evaluation_module and isinstance(value, dict):
                kwargs_per_evaluation_module[key].update(value)

        for evaluation_module in self.evaluation_modules:
            batch = {
                "predictions": predictions,
                "references": references,
                **kwargs_per_evaluation_module[evaluation_module.name],
            }
            results.append(evaluation_module.compute(**batch))

        return self._merge_results(results)


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCEWithLogitsLoss(reduction="none")
        self.mseloss = nn.MSELoss(reduction="none")

    def forward_objectness(self, objectness, gt_boxes_label, gt_and_neg_boxes_label):
        loss = self.bceloss(objectness, gt_boxes_label)
        loss = (loss * gt_and_neg_boxes_label).sum()
        return loss

    def forward_class(self, class_logits, classification_label, gt_boxes_label):
        loss = self.bceloss(class_logits, classification_label)
        loss = (loss * gt_boxes_label).sum()
        return loss

    def forward_coordinates(self, tx_ty_tw_th, coordinates_label, gt_boxes_label):
        loss = self.mseloss(tx_ty_tw_th, coordinates_label)
        loss = (loss * gt_boxes_label).sum()
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

        objectness_loss = self.forward_objectness(
            preds["objectness"],
            targets["gt_boxes_label"],
            targets["gt_and_neg_boxes_label"],
        )
        class_loss = self.forward_class(
            preds["class_logits"],
            targets["classification_label"],
            targets["gt_boxes_label"],
        )
        tx_ty_tw_th_loss = self.forward_coordinates(
            preds["tx_ty_tw_th"],
            targets["coordinates_label"],
            targets["gt_boxes_label"],
        )
        loss = objectness_loss + class_loss + tx_ty_tw_th_loss
        return {
            "loss": loss,
            "objectness_loss": objectness_loss,
            "class_loss": class_loss,
            "coordinates_loss": tx_ty_tw_th_loss,
        }
