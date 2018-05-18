steps to get the detection to work:

`git clone https://github.com/tensorflow/models/`

install the protobuff compiler and other packages:
```
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
sudo pip install Cython
sudo pip install jupyter
sudo pip install matplotlib
```


From tensorflow/models/research/, (this is in the cloned directory)

`protoc object_detection/protos/*.proto --python_out=.`

open the jupyter notebook models/research/object_detection/object_detection.ipynb
 and run everything.
 
 You can change the used model by changing the MODEL_NAME and maybe the PATH_TO_LABELS if you use a different source dataset.
 [Available Models](https://github.com/tensorflow/models/blob/676a4f70c20020ed41b533e0c331f115eeffe9a3/research/object_detection/g3doc/detection_model_zoo.md)

 [Original tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) 


# Todo
 1. Use the detection in some way
    1. Use the generated labels as an additional feature
    2. use the detected areas to do an extra prediction on
    3. ...
4. Adapt the ipynb in models/research to use it 