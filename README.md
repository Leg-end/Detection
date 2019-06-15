# Detection
This a Faster R-CNN model based on paper << Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.>>
http://arxiv.org/pdf/1506.01497.pdf written for learning purposes
The program structure is base on Tensorflow's 'nmt'https://github.com/tensorflow/nmt and endernewton's workship tf-faster-rcnn
https://github.com/endernewton/tf-faster-rcnn
PS:For a good and more up-to-date implementation for faster/mask RCNN with multi-gpu support, please see endernewton's new example
https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN

# DataSet
MCOCO2014

# Pre-trained model
download location:https://github.com/tensorflow/models/tree/master/research/slim
VGG_16
ResNetV1_50
Inception_V3 unable to work temporarily,cause Tensorflow's avg pooling method don't supoort dynamic ksize and stride temporarily,
see Issue https://github.com/tensorflow/tensorflow/issues/26961
