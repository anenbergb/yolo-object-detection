# Example Results

Tensorboard is used to log the loss, evaluation metrics (mean average precision), and image predictions on the validation set.

As a trial run, I trained the Yolo model for 100 epochs with the default settings. Unfortunately the model only achieved 7.6% AP and 12.6% AP50, which is well below the 33.0% AP and 57.9% AP50 results quoted in the YoloV3 paper. 
![image](https://github.com/user-attachments/assets/575f29d8-3dc1-4a8a-bd88-8cffc4ddb880)
![image](https://github.com/user-attachments/assets/9cb3a16e-31fb-46fc-94b9-b0e1b7682e0b)



The reduced performance may be attributed to:
* overfitting the training dataset. I observed higher loss on the validation set than on the training dataset.
* possibly incorrect normalization of the objectness loss. At this time, the objectness loss was normalized by dividing by the total number of positive anchors (anchors best matching the ground truth boxes)
and negative anchors (those with less than or equal to 0.5 IOU). This normalization scheme significantly diluted the contribution of the the positive anchors, making the loss dominated by the negative anchors.
The objectness loss curve on the training dataset shows that the objectness loss was minimized from an early iteration, which reflects that the model is not properly weighing false negative objectness predictions
since they number of positive anchors is vastly outnumbered by the number of negative anchors. 
![image](https://github.com/user-attachments/assets/6b6fe3b8-7635-47e2-ba41-b8da7889f60b)


* different choice of backbone. Our implementation relies on a pre-trained ResNeXt-50 backbone, whereas the original YoloV3 model relies on Darknet-53.  
* different implementation of the feature pyramid network and U-Net. Due to our choice of ResNeXt-50 backbone, the number of channels per low dimensional feature map (layer2=512, layer3=1024, layer4=2048) output from ResNeXt was greater than that output by Darknet-53 (128, 256, 512), which required additional 1x1 laternal convolutional kernels to reduce the channel dimension to align with the YoloV3 upscaling (U-Net) blocks.
* training only at a single resolution 608x608. Our implementation simply rescales every image to 608x608, whereas the original YoloV3 model relies on "multi-scale training" where the input image is randomly resized. Multi-scale training exposes the model to objects at different sizes, which will impact the features learned at each intermediate feature pyramid scale. For example, the higher resolution lower layers (layer2) which typically capture fine-grain details and are responsible for small object detection, will now need to also recognize larger objects. Overall multi-scale training should improve the model's robustness to object size variation.
* simple resizing rather than letterbox padding. Our implementation simply rescales every image to 608x608 which will stretches images and does not preserve the aspect ratio of objects. Letterbox padding (adding padding on both sides of the image) will preserve the aspect ratio and avoid introducing distortions that may confuse the model.
* difference in data augmentation. Our implementation applies simple data augmentation such as horizontal flipping and random photometric distortions such as brightness, contrast, saturation, or hue adjustment. More aggressive data augmentation such as scaling and resizing, cropping, rotations or affine transformations, noise injection, cutout or random erasing, may help regularize the model to avoid overfitting to the training dataset.

## Other Tensorboard plots
![image](https://github.com/user-attachments/assets/23380223-91ae-4250-9c8e-609b6919e69a)

![image](https://github.com/user-attachments/assets/d4f53d1a-235d-4a06-86cb-25094762eb71)

Ground Truth detections on the validation set
![image](https://github.com/user-attachments/assets/49013c7a-77e7-496b-93cd-5c76c5c49bb0)

Predictions on the validation set with IOU threshold = 0.5, and objectness threshold = 0.5
![image](https://github.com/user-attachments/assets/0f2b5b02-5ba5-40ba-8bca-d0b50a02d5d4)

Predictions on the validation set with IOU threshold = 0.5, and objectness threshold = 0.25
![image](https://github.com/user-attachments/assets/99a2dc6b-37c3-4ae0-b02a-0ff7ebe05057)
