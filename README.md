# Semantic Segmentation of An Aerial Urban Environment Image Using U-Net Deep Learning Model

<figure>
  <img
  src="https://github.com/must23/unet_urban_segmentation/blob/main/documents/process-diagram.png"
  alt="Figure 1 rqt graph showing the nodes used in the auto_race-car.">
</figure>

In this project, I employ a U-Net model architecture using an aerial urban environment dataset from the International Society of Photogrammetry and Remote Sensing (ISPRS). The study case involves the hyperparameter tuning technique and the solving issue of the insufficient dataset for smaller object segmentation and model training issues such as overfitting.

## Required packages
- [torchvision](https://pytorch.org/)
- [skimage](https://scikit-image.org/)
- [tqdm](https://pypi.org/project/tqdm/3.8.0/)


## Important files
- `src/Segment_Train.py `: To run the training process 
- `src/Segment_Test.py`: To run the testing process 
- `src/Unet.py` : File contains configuration for the Unet design.
