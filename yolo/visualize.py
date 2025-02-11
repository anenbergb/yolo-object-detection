from PIL import ImageDraw, ImageFont


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
