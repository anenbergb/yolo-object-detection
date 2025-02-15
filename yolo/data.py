from typing import Optional, Callable, Any, Dict
import os
import torch
from torchvision.transforms import v2
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

from torchvision import tv_tensors
from torchvision.transforms.v2._utils import _get_fill



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
            v2.ToDtype(torch.float32, scale=True),  # Normalize expects float input
            v2.Normalize(mean=mean, std=std),
            # v2.ToPureTensor(),
        ]
    )

class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_root: str, split: str = "train", transform: Optional[Callable] = None):
        assert split in ("train", "validation")
        root = os.path.join(dataset_root, split, "data")
        annFile = os.path.join(dataset_root, split, "labels.json")
        self.dataset = wrap_dataset_for_transforms_v2(CocoDetection(root = root, annFile = annFile))
        self.transform = transform

        self.class_id2name = {id: d["name"] for id,d in self.dataset.coco.cats.items()}
        self.class_id2idx = {id: idx for idx, id in enumerate(self.dataset.coco.cats.keys())}
        
        self.class_names = [d["name"] for d in self.dataset.coco.cats.values()]
        self.class_idx2id = {idx: id for id, idx in self.class_id2idx.items()}

    def num_classes(self):
        return len(self.dataset.coco.cats)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        class_ids = target["labels"]
        target["class_names"] = [self.class_id2name[id.item()] for id in class_ids]
        target["labels"] = torch.tensor([self.class_id2idx[id.item()] for id in class_ids])
        target["class_ids"] = class_ids

        if self.transform:
            img, target = self.transform(img, target)
        return img, target
    

class Collate:
    def __call__(self, batch):
        """
        batch should be a list of dictionaries with keys "pil_image", "label", "image", and "class_name"
        """
        images = [sample["pil_image"] for sample in batch]
        for sample in batch:
            sample.pop("pil_image")

        batch = torch.utils.data.default_collate(batch)
        batch["pil_image"] = images
        return batch
    
class PadToMultipleOf32(v2.Pad):
    def __init__(self, fill=0, padding_mode='constant'):
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
        return self._call_kernel(v2.functional.pad, inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)