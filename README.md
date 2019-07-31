## onnx-parser

Open Neural Network Exchange model parser in C



Build instructions:

```
scons
```

Run:

```
./onnx-parser mnist.onnx
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

```
