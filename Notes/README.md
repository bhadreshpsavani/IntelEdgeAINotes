Edge? : Local or Near Local Process not just anywhere on the cloud

Edge vs Cloud:
* Edge  : Edge means no need to send to the cloud;it can be more secure and have less impact on your network.
* Cloud : Cloud applications get data locally, send to the cloud for processing then send back a response. 

Importance of Edge in AI:
* Network Impacts
* Latency considerations
* Security concerns
* Optimizations for local inference

Application:
* Self Driving Cars
* Remote Nature Camera

Edge Application Life Cycle:

![image1](https://github.com/bhadreshpsavani/MachineLearningOnEdge/blob/master/images/edgeApplicationLifeCycle.png)

### [OpenVINO toolkit](https://software.intel.com/en-us/openvino-toolkit):

- "Open Visual Inferencing and Neural Network Optimization"

-  By optimizing for model speed and size, OpenVINO™ enables running at the edge

### [Computer Vision Problems](https://medium.com/analytics-vidhya/image-classification-vs-object-detection-vs-image-segmentation-f36db85fe81):
* Image Classification: Classify Image in classes

* Image Localization: Find location of Single Object in image

* Image Detection: Find Location of Multiple Objects and create bounding boxes on image

* Image Segmentation: create Segements in the image based on the pixel values 


### Common Computer Vision Architectures:

* #### [SSD](https://arxiv.org/abs/1512.02325) #### 
is an object detection network that combined classification with object detection through the use of default bounding boxes at different network levels. 

* #### [ResNet](https://arxiv.org/abs/1512.03385) #### 
utilized residual layers to “skip” over sections of layers, helping to avoid the vanishing gradient problem with very deep neural networks. 

* #### [MobileNet](https://arxiv.org/abs/1704.04861) #### 
utilized layers like 1x1 convolutions to help cut down on computational complexity and network size, leading to fast inference without substantial decrease in accuracy.

* #### [YOLO](https://arxiv.org/abs/1506.02640) ####
You Only Look Once
A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation

* #### [RCNN](https://arxiv.org/pdf/1311.2524.pdf) ####

introduce a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position


### [Available Pre-Trained Models in OpenVINO](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models)

* #### Pre-Trained Models: Types
    * Classification : Age & Gender Recognition
    * Detection : Pedestrian Detection
    * segmentation : Advanced Roadside Identification
    
* #### Pre-Trained Models: Architectures
    * SSD : Enhanced Model-Face Detection
    * MobileNet : Standard Model-Face Detection
    * SSD+MobileNet : Pedestrian and Vehicle Detection
    
### How Pretrained Model are Optimized for better Edge Deployment? ###
* Different Precisions: less memory, less compute (trade-offs with accuracy when using lower precision)
* Fusing Layers in to Fewer Layers

### Preprocessing Input to the Model: ###
* Depends on selected model
* Required to do common image based preprocessing like resize, normalization etc.
* Checkout input section of pretrained model documentation to determin shape of input

### Process the Output: ###
Depending upon the application we need to process output to get desired output.

### Optimization Techniques ###
* Quantization : Quantization is the process of reducing the precision of a model(Precision of the model can be 32bit,16bit,8bit)
* Freezing : Freezing a TensorFlow model is usually a good idea whether before performing direct inference or converting with the Model Optimizer
* Fusion : Fusion relates to combining multiple layer operations into a single operation

## Convert Trained Model in to IR form using Model Optimizer: ##
The supported frameworks with the OpenVINO™ Toolkit are:
* [Caffe](https://caffe.berkeleyvision.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [MXNet](https://mxnet.apache.org/)
* [ONNX](https://onnx.ai/)(which can support PyTorch and Apple ML models through another conversion step)
* [Kaldi](https://kaldi-asr.org/doc/dnn.html)

### 1. [Tensorflow Models to IR](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html): ###
* Whether to use a frozen or unfrozen model, 
* Unfrozen models usually need the `--mean_values` and `--scale` parameters fed to the Model Optimizer, while the frozen models from the Object Detection Model Zoo don’t need those parameters.
* The frozen models will need TensorFlow-specific parameters like `--tensorflow_use_custom_operations_config` and `--tensorflow_object_detection_api_pipeline_config`. Also, `--reverse_input_channels` is usually needed, as TF model zoo models are trained on RGB images, while OpenCV usually loads as BGR. 
* Certain models, like YOLO, DeepSpeech, and more, have their own separate pages.

### 2. [Caffee Model to IR](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe.html): ###
* No need to consider Freezing/UnFreezing Model since its a TensorFlow concept 
* Caffe models need to feed both the `.caffemodel` file, as well as a `.prototxt` file, into the Model Optimizer. 
* If the `.prototxt` file has same name like `.caffemodel` model file, only the model needs to be directly input as an argument, while if the `.prototxt` file has a different name than the model, it should be fed in with `--input_proto` as well.

### 3. [ONNX Model to IR](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html): ###
* No need of aditional arguments, conversion is possible with general arguments of earlier representations

### [Cutting Model as an Optimization Technique](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Cutting_Model.html): ###
![Why Cutting](/images/cuttingModel.png)
Some common reasons for cutting are:
* The model has pre- or post-processing parts that don’t translate to existing Inference Engine layers.
* The model has a training part that is convenient to be kept in the model, but is not used during inference.
* The model is too complex with many unsupported operations, so the complete model cannot be converted in one shot.
* The model is one of the supported SSD models. In this case, you need to cut a post-processing part off.
* There could be a problem with model conversion in the Model Optimizer or with inference in the Inference Engine. To localize the issue, cutting the model could help to find the problem

There’s two main command line arguments to use for cutting a model with the Model Optimizer, named intuitively as `--input` and `--output`, where they are used to feed in the layer names that should be either the new entry or exit points of the model.


## Inferance Engine:
* Provide hardware based optimization.
* API to integrate app with edge devices
* Build in C++ can also use Python Wrapper
* It also provides computer vision functions for inferance 
* CPU, VPU(Visual Processing Unit), GPU, FPGAs(Field Processing Graphical Arrays)
* Use the Inferenace Engine with IR: [`IECore`](https://docs.openvinotoolkit.org/latest/classie__api_1_1IECore.html)(Python wrapper) and [`IENetwork`](https://docs.openvinotoolkit.org/latest/classie__api_1_1IENetwork.html)(holds the network)
