# Object-Detection-in-Satellite-Images-using-Mask-R-CNN

This project is to detect ships on satellite images using Mask R-CNN which is a deep neural network used to solve instance segmentation problems. This generates bounding boxes and masks around each ship detected in the satellite image. 

This repository includes the following,
- Dataset - Contains both train and validation images which were obtained by taking screenshots of several ports from Google Earth. 
- Annotations - Images were annotated using VGG Images Annotator
- Pre-trained weights for MS COCO - Transfer learning approach is used here. Even though, COCO dataset does not contain ship class, it has been trained on 120k other images which means its weights have learnt a lot of common features of natural images which is useful for this project. https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
- Third Party Mask R-CNN implementation - Obtained from Mask R-CNN project by Matterport https://github.com/matterport/Mask_RCNN. 

Some of the predicted images obtained from the trained model were as follows.


![24](https://user-images.githubusercontent.com/53529711/131241182-06be8e7b-a4ba-4cf4-844b-35effbef1cb0.png)


![22](https://user-images.githubusercontent.com/53529711/131241186-6d20b9f9-d698-4031-b056-aac4454b8df9.png)

