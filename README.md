## onnx-parser

Open Neural Network Exchange model parser in C



### 1. Build instructions:

```
cd examples/
scons
```



### 2. Examples

#### 2.1 onnx-parser

This example loads a model from file and prints out its structure.

Run:

```
./onnx-parser mnist-sm.onnx
```

Output:

```
--- Reading from mnist.onnx ---
---- Model info ----
IR Version is 5
Produceer name is keras2onnx
Produceer version is 1.5.1
Produceer version is onnx
---- Graph Info ----
Graph inputs number: 1
---- Graph Input Info ----
Input name conv2d_1_input
Input type FLOAT
Input dimension 4
N x 28 x 28 x 1
Graph outputs number: 1
---- Graph Output Info ----
Output name dense_2/Softmax:0
Output type FLOAT
Output dimension 2
? x 10
---- Graph Node Info ----
Graph nodes number: 15
Transpose   : conv2d_5_input                 ->    adjusted_input1                [Transpose6]
Conv        : adjusted_input1                ->    convolution_output1            [conv2d_5]
Relu        : convolution_output1            ->    conv2d_5/Relu:0                [Relu1]
MaxPool     : conv2d_5/Relu:0                ->    pooling_output1                [max_pooling2d_5]
Conv        : pooling_output1                ->    convolution_output             [conv2d_6]
Relu        : convolution_output             ->    conv2d_6/Relu:0                [Relu]
MaxPool     : conv2d_6/Relu:0                ->    pooling_output                 [max_pooling2d_6]
Transpose   : pooling_output                 ->    max_pooling2d_6/MaxPool:0      [Transpose1]
Reshape     : max_pooling2d_6/MaxPool:0      ->    flatten_3/Reshape:0            [flatten_3]
MatMul      : flatten_3/Reshape:0            ->    transformed_tensor1            [dense_5]
Add         : transformed_tensor1            ->    biased_tensor_name1            [Add1]
MatMul      : biased_tensor_name1            ->    transformed_tensor             [dense_6]
Add         : transformed_tensor             ->    biased_tensor_name             [Add]
Softmax     : biased_tensor_name             ->    dense_6/Softmax:01             [Softmax]
Identity    : dense_6/Softmax:01             ->    dense_6/Softmax:0              [Identity1]

```



#### 2.2 mnist in RAM

This minimum example stores model in RAM, thus external model is not required.

```
/onnx-mnist
```



```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@                          @@@@@@@@@@@@@@@@@@@@
@@@@@@                                @@@@@@@@@@@@@@@@@@
@@@@              @@@@@@@@@@@@@@        @@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      @@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      @@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      @@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@        @@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@                  @@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@                    @@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@                @@@@      @@@@@@@@@@@@@@
@@@@@@@@@@@@@@        @@@@@@@@@@@@@@      @@@@@@@@@@@@@@
@@@@@@@@@@@@      @@@@@@@@@@@@@@@@@@      @@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      @@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@        @@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      @@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      @@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

Predictions: 
0.000498 0.000027 0.017220 0.028220 0.000643 0.002182 0.000000 0.753116 0.026616 0.171477 

The number is 7
```



#### 2.3 mnist-sm

This example constructs a model manually, and load weights from file.

```
./onnx-mnist-sm
```



#### 2.4 mnist-model

This example loads a model from file system and run inference automatically.

```
./onnx-mnist-model 
```



```
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@        @@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@              @@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@                    @@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@          @@@@@@@@    @@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@        @@@@@@@@@@    @@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@@@    @@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@  @@@@      @@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@          @@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@              @@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@            @@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@          @@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      @@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@  @@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@    @@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@  @@@@@@@@@@@@@@@@@@      @@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@    @@@@@@@@@@@@        @@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@                      @@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@                  @@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

[ 0] Transpose  Transpose6
[ 1] Conv       conv2d_5             [28, 28,  1] --> [28, 28,  2]
[ 2] Relu       Relu1                [28, 28,  2] --> [28, 28,  2]
[ 3] MaxPool    max_pooling2d_5      [28, 28,  2] --> [14, 14,  2]
[ 4] Conv       conv2d_6             [14, 14,  2] --> [14, 14,  2]
[ 5] Relu       Relu                 [14, 14,  2] --> [14, 14,  2]
[ 6] MaxPool    max_pooling2d_6      [14, 14,  2] --> [ 7,  7,  2]
[ 7] Transpose  Transpose1
[ 8] Reshape    flatten_3            [ 7,  7,  2] --> [ 1, 98,  1]
[ 9] MatMul     dense_5              [ 1, 98,  1] --> [ 1,  4,  1]
[10] Add        Add1                 [ 1,  4,  1] --> [ 1,  4,  1]
[11] MatMul     dense_6              [ 1,  4,  1] --> [ 1, 10,  1]
[12] Add        Add                  [ 1, 10,  1] --> [ 1, 10,  1]
[13] Softmax    Softmax              [ 1, 10,  1] --> [ 1, 10,  1]
[14] Identity   Identity1

Predictions:
0.007383 0.000000 0.057510 0.570970 0.000000 0.105505 0.000000 0.000039 0.257576 0.001016

The number is 3
```

