# yolo-object-detection
A (mostly) from scratch implementation of the YoloV3 neural network object detector. Rather implementing Darknet-53, I've used an ImageNet pre-trained [ResNeXt backbone from torchvision](https://pytorch.org/vision/main/models/resnext.html). The [darknet github repo](https://github.com/pjreddie/darknet) and [this "PyLesson blog post"](https://pylessons.com/YOLOv3-code-explanation) were referenced for the design of the feature pyramid network U-Net because the [YoloV3 paper](https://arxiv.org/abs/1804.02767) does not provide sufficient detail. The same 9 anchor boxes identified in the paper were used because I intended to train the model on the COCO dataset. The bounding box encoding, decoding, and ground truth anchor box assignment are the most complex elements of the implementation. All of the ground truth anchor box assignment and encoding is performed in collate function of the dataloader. A batch of N images, each of which contains a variable number of ground truth objects, is transformed into label tensors, "coordinates_label" (N,L,4), "classification_label" (N,L,80), "gt_boxes_label" (N,L,1), and "gt_and_neg_boxes_label" (N,L,1) where L is the multi-scale flattened anchor box grid. For an input image of size (608,608) the YoloV3 feature pyramid will predict boxes from feature maps at three scales: (76,76), (38,38), and (19,19). The multi-scale flattened anchor grid will be of size L = 3*(76*76 + 38*38 + 19*19) = 22743.

As described in the [YoloV3 paper](https://arxiv.org/abs/1804.02767), the objectness score is predicted for each bounding box using logistic regression. The objectness loss is only calculated for the singular anchor box that best overlaps the ground truth bounding box, and for the anchor boxes with less than or equal to 0.5 overlap. Anchor boxes that overlap with the ground truth box by greater than 0.5 are ignored in the objectness loss calculation. The mean squared error bounding box loss and the binary cross entropy multilabel classification loss are only calculated for the anchor boxes that most overlap.

I use fairly simple data augmentation such as RandomPhotometricDistort and RandomHorizontalFlip. More sophisticated data augmentation would certainly improve the model performance. I also only train the model on fixed size input -- e.g., I resize all the images to 608x608 square dimensions. The model performance could be improved by randomly resizing the images while maintaining the aspect ratio, and then applying letterbox padding such that the minibatch is of a consistent size.


[Huggingface Accelerate library](https://huggingface.co/docs/accelerate/en/index) is used to manage the distributed and mixed precision training configuration rather than directly calling underlying torch primatives. 

The [Mean Average Precision routine from the Torchmetrics library](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html) is used for periodic mAP calculation.

The YoloV3 box encoding equations are shown below. Notice that the scale factor of the feature pyramid (e.g. 8x, 16x, 32x) should be multiplied against $b_x$ and $b_y$ to get the bounding box center in the original image space.

![image](https://github.com/user-attachments/assets/d2698001-e2d9-4d55-b0b3-5a5d5cdb32b3)

# Example Results

Tensorboard is used to log the loss, evaluation metrics (mean average precision), and image predictions on the validation set.

See [example-training-run.md](example-training-run.md) for the results from training Yolo for 100 epochs on the COCO dataset.


# FiftyOne COCO dataset
FiftyOne is used to download and visualize the COCO dataset. See the [FiftyOne COCO Integration](https://docs.voxel51.com/integrations/coco.html#coco) guide for more details.
FiftyOne can be configured by editing `~/.fiftyone/config.json`. See the [FiftyOne User Guide](https://docs.voxel51.com/user_guide/config.html) for further instruction.

# References
* [Yolo paper](https://arxiv.org/abs/1506.02640)
* [YoloV2 / Yolo9000 paper](https://arxiv.org/abs/1612.08242)
* [YoloV3 paper](https://arxiv.org/abs/1804.02767)
* [YoloV4 paper](https://arxiv.org/abs/2004.10934)
