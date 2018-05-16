# Networks
Currently, we implemented 3/4-ish networks:
* Inception V3 (Total Params: __22.3 mil__)
* Resnet 50 (Total Params: __26 mil__)
* MobileNet (Total Params: __3.46 mil__)
* Small Test Net

## Freezing Layers
We have routines for training only the added top layers and freezing the base layers. These work as is. However, there is also a routine which is able to freeze the first N layers and train the rest. In this section, we will report noteworthy layers to train from the base models.

* __Inception V3__: Training the top 2 Inception Blocks; _freeze the first __249__ layers_
* __ResNet 50__: Training from _res5c_branch2a_; _freeze the first __163__ layers_
* __MobileNet__: Training the top 2 Conv Blocks; _freeze the first __69__ layers_