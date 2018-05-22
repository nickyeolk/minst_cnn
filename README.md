# Classifying the MNIST dataset with CNNs
## Demonstrating different methods of implementing Convolutional Neural Networks (CNN) using tensorflow.
Two methods of implementing CNNs in tensorflow are used: Using custom estimators, and building from the low level API.
The CNN example using custom estimators are in cnn_minst.py, and the low level API is built in cnn_minst_low_level.py. The MNIST dataset is used, and the custom estimators are adapted from the [tensorflow tutorial](https://www.tensorflow.org/tutorials/layers). Changes are made so that the model can be monitored in tensorboard. The low level API implementation is adapted from [the tensorboard demo](https://gist.github.com/decentralion/4f02ab8f1451e276fea1f165a20336f1#file-mnist-py). Once again, changes are made to output things I found interesting.
## Sample outputs
![Sample convolutional filter outputs](./conv1_filter_outputs.JPG)
