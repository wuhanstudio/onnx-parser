#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <float.h>
#include <math.h>

#include <onnx-parser.h>

#include "mnist.h"
#include "backend.h"

float* conv2D_layer(Onnx__GraphProto* graph, const float *input, long* shapeInput, const char* weight, const char* bias)
{
    // Get weight shape
    long* shapeW = onnx_graph_get_dims_by_name(graph, weight);
    if(shapeW == NULL)
    {
        return NULL;
    }
    long dimW = onnx_graph_get_dim_by_name(graph, weight);
    if(dimW < 0)
    {
        return NULL;
    }

    // Get weights
    // NCWH --> NWHC
    int permW_t[] = { 0, 2, 3, 1};
    float* W = onnx_graph_get_weights_by_name(graph, weight);
    if(W == NULL)
    {
        return NULL;
    }
    float* W_t = onnx_tensor_transpose(W, shapeW, dimW, permW_t);

    // Get bias
    float* B = onnx_graph_get_weights_by_name(graph, bias);
    if(B == NULL)
    {
        return NULL;
    }

    float* output = (float*) malloc(sizeof(float)*shapeW[0]*shapeInput[1]*shapeInput[2]);
    memset(output, 0, sizeof(sizeof(float)*shapeW[0]*shapeInput[1]*shapeInput[2]));
    conv2D(input, shapeInput[1], shapeInput[2], shapeW[1], W, shapeW[0], shapeW[2], shapeW[3], 1, 1, 1, 1, B, output, shapeInput[1], shapeInput[2]);

    free(shapeW);
    free(W);
    free(B);
    free(W_t);

    return output;
}

float* dense_layer(Onnx__GraphProto* graph, const float *input, long* shapeInput, const char* weight, const char* bias)
{
    long* shapeW =  onnx_graph_get_dims_by_name(graph, weight);
    if(shapeW == NULL)
    {
        return NULL;
    }
    long dimW = onnx_graph_get_dim_by_name(graph, weight);
    if(dimW < 0)
    {
        return NULL;
    }

    int permW_t[] = {1, 0};
    float* W = onnx_graph_get_weights_by_name(graph, weight);
    if(W == NULL)
    {
        return NULL;
    }
    float* W_t = onnx_tensor_transpose(W, shapeW, dimW, permW_t);
    float* B = onnx_graph_get_weights_by_name(graph, bias);
    if(B == NULL)
    {
        return NULL;
    }
    float* output = (float*) malloc(sizeof(float)*shapeW[1]);
    memset(output, 0, sizeof(sizeof(float)*shapeW[1]));
    dense(input, W_t, shapeW[0], shapeW[1], B, output);

    free(shapeW);
    free(W);
    free(W_t);
    free(B);

    return output;
}

int main(int argc, char const *argv[])
{
    if (argc < 2) 
    {
        printf("Usage: %s onnx_file_name [image_num]\n", argv[0]);
        return 0;
    }

    // Load Model
    Onnx__ModelProto* model = onnx_load_model(argv[1]);
    if(model == NULL)
    {
        printf("Failed to load model %s\n", argv[1]);
        return -1;
    }

    // Read Input
    int img_index = 0;
    if(argc == 3)
    {
        img_index = atoi(argv[2]);
    }
    print_img(img[img_index]);

    // 0. Transpose Input
    // N C W H --> N W H C
    long shapeInput[] = {1, 28, 28};
    long dimInput = 3;
    int perm[] = { 1, 2, 0};
    float* input = onnx_tensor_transpose(img[img_index], shapeInput, dimInput, perm);

    // 1. Conv2D
    // N W H C
    float* conv1 = conv2D_layer(model->graph, input, shapeInput, "W3", "B3");
    free(input);

    // 2. Relu
    relu(conv1, 28*28*2);

    // 3. Maxpool
    float* maxpool1 = (float*) malloc(sizeof(float)*14*14*2);
    memset(maxpool1, 0, sizeof(sizeof(float)*14*14*2));
    maxpool(conv1, 28, 28, 2, 2, 2, 0, 0, 2, 2, 14, 14, maxpool1);
    free(conv1);

    // 4. Conv2D
    // N W H C 
    long shapeMaxpool1[] = {2, 14, 14};
    float* conv2 = conv2D_layer(model->graph, maxpool1, shapeMaxpool1, "W2", "B2");
    free(maxpool1);

    // 5. Relu
    relu(conv2, 14*14*2);

    // 6. Maxpool
    float* maxpool2 = (float*) malloc(sizeof(float)*7*7*2);
    memset(maxpool2, 0, sizeof(sizeof(float)*7*7*2));
    maxpool(conv2, 14, 14, 2, 2, 2, 0, 0, 2, 2, 7, 7, maxpool2);
    free(conv2);

    // Flatten NOT REQUIRED

    // 7. Dense
    long shapeMaxpool2[] = {1, 98};
    float* dense1 = dense_layer(model->graph, maxpool2, shapeMaxpool2, "W1", "B1");
    free(maxpool2);

    // 8. Dense
    long shapeDense1[] = {1, 4};
    float* dense2 = dense_layer(model->graph, dense1, shapeDense1, "W", "B");
    free(dense1);

    // 9. Softmax
    float* output = (float*) malloc(sizeof(float)*10);
    memset(output, 0, sizeof(sizeof(float)*10));
    softmax(dense2, 10, output);

    // 10. Result
    float max = 0;
    int max_index = 0;
    printf("\nPredictions: \n");
    for(int i = 0; i < 10; i++)
    {
        printf("%f ", output[i]);
        if(output[i] > max)
        {
            max = output[i];
            max_index = i;
        }
    }
    printf("\n");
    printf("\nThe number is %d\n", max_index);

    free(dense2);
    free(output);

    return 0;
}
