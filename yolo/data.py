from typing import Optional, Callable, Any, Dict, List, Tuple
import os
import torch
from torchvision.transforms import v2
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

from torchvision import tv_tensors
from torchvision.transforms.v2._utils import _get_fill

from yolo.anchors import (
    make_spatial_anchors,
    make_scale_map,
    make_anchor_map,
    make_spatial_anchor_mask,
    encode_boxes,
)


def labels_getter(inputs: Any) -> List[torch.Tensor]:
    # targets dictionary should be the second element in the tuple
    if isinstance(inputs, (tuple, list)):
        inputs = inputs[1]

    label_names = ["class_idx", "class_id", "iscrowd"]
    return [inputs[label_name] for label_name in label_names]


# Other possible transforms to consider
# https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#sphx-glr-auto-examples-transforms-plot-transforms-e2e-py
def get_val_transforms(
    resize_size=608,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    # TODO: clip the boxes to be within the limit of the image, but not equal, just slightly smaller
    return v2.Compose(
        [
            v2.ToImage(),  # convert PIL image to tensor
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize((resize_size, resize_size)),  # resize the image. bilinear
            v2.ClampBoundingBoxes(),  # clamp bounding boxes to be within the image
            v2.SanitizeBoundingBoxes(
                labels_getter=labels_getter
            ),  # Remove degenerate/invalid bounding boxes and their corresponding labels and masks.
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=mean, std=std),
            # v2.ToPureTensor(),
        ]
    )


class CocoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        assert split in ("train", "validation")
        root = os.path.join(dataset_root, split, "data")
        annFile = os.path.join(dataset_root, split, "labels.json")
        # https://pytorch.org/vision/main/generated/torchvision.datasets.wrap_dataset_for_transforms_v2.html#torchvision.datasets.wrap_dataset_for_transforms_v2
        self.dataset = wrap_dataset_for_transforms_v2(
            CocoDetection(root=root, annFile=annFile),
            target_keys=["image_id", "boxes", "labels", "iscrowd"],
        )
        self.transform = transform

        self.class_id2name = {id: d["name"] for id, d in self.dataset.coco.cats.items()}
        self.class_id2idx = {
            id: idx for idx, id in enumerate(self.dataset.coco.cats.keys())
        }

        self.class_names = [d["name"] for d in self.dataset.coco.cats.values()]
        self.class_idx2id = {idx: id for id, idx in self.class_id2idx.items()}

    def num_classes(self):
        return len(self.dataset.coco.cats)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        target contains:
            "image_id": int
            "boxes": torch.tensor (N,4)
            "labels": torch.tensor (N,)

        updates the target to include:
            "image_id": int
            "boxes": torch.tensor (N,4)
            "class_name": List[str]
            "class_idx": torch.tensor (N,)
            "class_id": torch.tensor (N,)
        """
        img, target = self.dataset[idx]
        class_ids = target.pop("labels")
        # target["class_name"] = [self.class_id2name[id.item()] for id in class_ids]
        target["class_idx"] = torch.tensor(
            [self.class_id2idx[id.item()] for id in class_ids]
        )
        target["class_id"] = class_ids

        if self.transform:
            img, target = self.transform(img, target)
        return img, target


class CollateWithAnchors:
    """
    Assume that all the images have been resized to the same size.

    """

    def __init__(
        self,
        anchors: List[Tuple[int, int]],
        scales: List[int],
        image_height: int,
        image_width: int,
        num_anchors_per_scale: int = 3,
        num_classes: int = 80,
        verbose: bool = False,
    ):
        self.anchors = anchors
        self.scales = scales
        self.image_height = image_height
        self.image_width = image_width
        self.num_anchors_per_scale = num_anchors_per_scale
        self.num_classes = num_classes
        self.verbose = verbose
        self.spatial_anchors = make_spatial_anchors(
            anchors, scales, image_height, image_width, num_anchors_per_scale, verbose
        )
        self.scale_map = make_scale_map(
            scales, image_height, image_width, num_anchors_per_scale
        )
        self.anchor_map = make_anchor_map(
            anchors, scales, image_height, image_width, num_anchors_per_scale
        )

    def make_label_tensors(self, batch_boxes, batch_class_idx):
        """
        Inputs:
            batch_boxes: List[Tensor(B,4)]
            batch_class_idx: List[Tensor(B,)]
            B is the number of boxes in the image. List is length N, where N is the number of images in the batch.

        Need to construct label tensors for the objectness, coordinates, and classification losses
        objectness: (N,L,1) where L = 22743. {1: if spatial_anchor_mask >= 0, 0: if mask == -2}, BCELoss reduction="none", then apply mask to ignore mask == -1
        coordinates: (N,L,4) where ground-truth encoded txtytwth is only defined for the spatial_anchor_mask >= 0 values. MSELoss reduction="none", then apply mask to ignore mask < 0
        classification: (N,L,C) where C=80 is one-hot encoding for class at the anchor box, only defined for spatial_anchor_mask >= 0 values. BCELoss reduction="none", then apply mask to ignore mask < 0
        """
        spatial_anchor_mask = make_spatial_anchor_mask(
            batch_boxes,
            self.spatial_anchors,
            self.scales,
            self.image_height,
            self.image_width,
            self.num_anchors_per_scale,
            verbose=self.verbose,
        )

        objectness_label = (spatial_anchor_mask >= 0).to(torch.long).unsqueeze(-1)
        coord_label = torch.zeros(*spatial_anchor_mask.shape, 4)
        classification_label = torch.zeros(*spatial_anchor_mask.shape, self.num_classes)

        for batch_idx, L_idx in zip(*torch.where(spatial_anchor_mask >= 0)):
            box_idx = spatial_anchor_mask[batch_idx, L_idx]

            # set encoded box
            scale = self.scale_map[L_idx]
            anchor = self.anchor_map[L_idx]
            gt_box = batch_boxes[batch_idx][box_idx]
            coord_label[batch_idx, L_idx] = encode_boxes(gt_box, scale, anchor)

            # set classification label
            class_idx = batch_class_idx[batch_idx][box_idx]
            classification_label[batch_idx, L_idx][class_idx] = 1
        return {
            "objectness_label": objectness_label,
            "coordinates_label": coord_label,
            "classification_label": classification_label,
        }

    def __call__(self, batched_image_target):

        batch_list = []
        for image, target in batched_image_target:
            batch_list.append(
                {
                    "image": image,
                    "image_id": target["image_id"],
                }
            )
        batch = torch.utils.data.default_collate(batch_list)
        label_names = ["boxes", "labels", "class_idx", "class_id", "iscrowd"]
        for label_name in label_names:
            batch[label_name] = [x[label_name] for x in batched_image_target]

        label_tensors = self.make_label_tensors(batch["boxes"], batch["class_idx"])
        batch.update(label_tensors)
        return batch


class PadToMultipleOf32(v2.Pad):
    def __init__(self, fill=0, padding_mode="constant"):
        super().__init__(padding=0, fill=fill, padding_mode=padding_mode)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # bounding boxes don't need padding since padding is applied
        # to the bottom and right sides
        padding = 0
        if isinstance(inpt, tv_tensors.Image):
            height, width = inpt.shape[-2:]
            pad_width = (32 - width % 32) % 32
            pad_height = (32 - height % 32) % 32
            padding = (0, 0, pad_width, pad_height)
        fill = _get_fill(self._fill, type(inpt))
        return self._call_kernel(
            v2.functional.pad,
            inpt,
            padding=padding,
            fill=fill,
            padding_mode=self.padding_mode,
        )
