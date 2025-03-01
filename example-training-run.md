# Example Training Run

Tensorboard is used to log the loss, evaluation metrics (mean average precision), and image predictions on the validation set.

I trained the Yolo model for 100 epochs with the default settings.


The model achieved 15.8% AP and 27.41% AP50, which is well below the 33.0% AP and 57.9% AP50 results quoted in the YoloV3 paper. 

<img src="https://github.com/user-attachments/assets/b3ca7807-b476-4879-a945-3c97897bf6ce" width="400" height="400" />
<img src="https://github.com/user-attachments/assets/9a173135-a123-419b-b033-0886a0333fb6" width="400" height="400" />




The reduced performance may be attributed to:
* overfitting the training dataset. I observed higher loss on the validation set than on the training dataset.
For example, the minimum loss on the training set was ~10 whereas the minimum loss on the validation set was ~200.
* Different learning schedule. I only train the model for 100 epochs, whereas the original YoloV3 paper trained the model for 300 epochs.
Furthermore, I trained the model with AdamW optimizer, an initial learning rate of 2e-4, and a one cycle cosine annealing schedule with linear warmup,
whereas the original YoloV3 paper used SGD with momentum, an initial learning rate of 1e-3 with a step-wise linear decay schedule, and a linear warmup.
* different choice of backbone. Our implementation relies on a pre-trained ResNeXt-50 backbone, whereas the original YoloV3 model relies on Darknet-53.  
* different implementation of the feature pyramid network and U-Net. Due to our choice of ResNeXt-50 backbone, the number of channels per low dimensional feature map (layer2=512, layer3=1024, layer4=2048) output from ResNeXt was greater than that output by Darknet-53 (128, 256, 512), which required additional 1x1 laternal convolutional kernels to reduce the channel dimension to align with the YoloV3 upscaling (U-Net) blocks.
* training only at a single resolution 608x608. Our implementation simply rescales every image to 608x608, whereas the original YoloV3 model relies on "multi-scale training" where the input image is randomly resized. Multi-scale training exposes the model to objects at different sizes, which will impact the features learned at each intermediate feature pyramid scale. For example, the higher resolution lower layers (layer2) which typically capture fine-grain details and are responsible for small object detection, will now need to also recognize larger objects. Overall multi-scale training should improve the model's robustness to object size variation.
* simple resizing rather than letterbox padding. Our implementation simply rescales every image to 608x608 which will stretches images and does not preserve the aspect ratio of objects. Letterbox padding (adding padding on both sides of the image) will preserve the aspect ratio and avoid introducing distortions that may confuse the model.
* difference in data augmentation. Our implementation applies simple data augmentation such as horizontal flipping and random photometric distortions such as brightness, contrast, saturation, or hue adjustment. More aggressive data augmentation such as scaling and resizing, cropping, rotations or affine transformations, noise injection, cutout or random erasing, may help regularize the model to avoid overfitting to the training dataset.

## Other Tensorboard plots
![image](https://github.com/user-attachments/assets/71dfd4ef-f78f-4eaa-a874-2af9a788dfa0)
![image](https://github.com/user-attachments/assets/101b3d1b-e67c-42d5-b785-be7a0cce6e79)

Ground Truth detections on the validation set
![image](https://github.com/user-attachments/assets/49013c7a-77e7-496b-93cd-5c76c5c49bb0)

Predictions on the validation set with IOU threshold = 0.5, and objectness threshold = 0.5
![image](https://github.com/user-attachments/assets/ad500b12-56c5-4899-bb64-5210ebd4660b)


Predictions on the validation set with IOU threshold = 0.5, and objectness threshold = 0.25
![image](https://github.com/user-attachments/assets/d05b4b26-5d92-4b2c-a985-91c11fc6ff48)
