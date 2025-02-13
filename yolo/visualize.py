from PIL import ImageDraw, ImageFont


import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


# adopted from https://github.com/pytorch/vision/blob/main/gallery/transforms/helpers.py
def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

def render_bounding_boxes(image, annotations, category_names):
    """
    Render bounding boxes and category_id tags on an image.

    :param image: PIL Image.
    :param annotations: List of annotations, where each annotation is a dictionary
                        with keys 'bbox' and 'category_id'.
    :param category_names: Dictionary mapping category_id to category name.
    :return: PIL.Image object with bounding boxes and category_id tags.
    """
    # Load the image
    draw = ImageDraw.Draw(image)

    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    # Draw each bounding box and category_id tag
    for annotation in annotations:
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]
        category_name = category_names.get(category_id, "Unknown")

        # Draw the bounding box
        x, y, width, height = bbox
        draw.rectangle(((x, y), (x + width, y + height)), outline="red", width=2)

        # Draw the category_id tag
        text = f"{category_name} ({category_id})"
        text_bbox = draw.textbbox((x, y), text, font=font)
        text_location = (x, y - (text_bbox[3] - text_bbox[1]))
        draw.rectangle(
            (
                (x, y - (text_bbox[3] - text_bbox[1])),
                (x + (text_bbox[2] - text_bbox[0]), y),
            ),
            fill="red",
        )
        draw.text(text_location, text, fill="white", font=font)

    return image
