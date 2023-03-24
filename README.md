# IA321-LIDARSegmentation


This is the repository for  AI Project (IA321) at ENSTA Paris.

Our project is about LIDAR Point Cloud segmentation.

There are 4 models: Unet, SETR, SegFormer and RangeViT.

Link to the dataset : https://drive.google.com/u/0/uc?id=1nuLgpfC26ErnmclcXjxjWDuHCqJt4ZRA. put it in the rootfolder of the project.

To train a model run : train_"model name".py, and it will train and save the model in TrainedModels folder.
To evaluate a model run : train_"model name".py <PATH_TO_TRAINED_MODEL>.
To save the prdiction on an example run : infer_"model name".py <PATH_TO_TRAINED_MODEL> <INDEX_OF_THE_EXAMPLE_IN_TEST_DATA_SET>
