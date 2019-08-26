#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <float.h>
#include <math.h>

#include <onnx-parser.h>

#include "mnist.h"
#include "backend.h"

float* conv2D_layer(Onnx__GraphProto* graph, const float *input, long* shapeInput, const char* layer_name)
{
    Onnx__NodeProto* node = onnx_graph_get_node_by_name(graph, layer_name);
    const char* weight = node->input[1];
    const char* bias = node->input[2];

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

    float* output = (float*) malloc(sizeof(float)*shapeW[0]*shapeInput[0]*shapeInput[1]);
    memset(output, 0, sizeof(sizeof(float)*shapeW[0]*shapeInput[0]*shapeInput[1]));
    conv2D(input, shapeInput[0], shapeInput[1], shapeW[1], W_t, shapeW[0], shapeW[2], shapeW[3], 1, 1, 1, 1, B, output, shapeInput[0], shapeInput[1]);

    free(W_t);

    return output;
}

float* maxpool_layer(Onnx__GraphProto* graph, float* input, long* shapeInput, const char* layer_name)
{
    Onnx__NodeProto* node = onnx_graph_get_node_by_name(graph, layer_name);
    
    uint16_t kernel_x = -1;
    uint16_t kernel_y = -1;
    uint16_t padding_x = 0;
    uint16_t padding_y = 0;
    uint16_t stride_x = -1;
    uint16_t stride_y = -1;

    for(int i = 0; i < node->n_attribute; i++)
    {
        if( strcmp(node->attribute[i]->name, "kernel_shape") == 0 )
        {
            kernel_x = node->attribute[i]->ints[0];
            kernel_y = node->attribute[i]->ints[1];
        }
        if( strcmp(node->attribute[i]->name, "strides") == 0 )
        {
            stride_x = node->attribute[i]->ints[0];
            stride_y = node->attribute[i]->ints[1];
        }
    }

    uint16_t out_x = (shapeInput[0] - kernel_x + 2 * padding_x) / stride_x + 1;
    uint16_t out_y = (shapeInput[1] - kernel_y + 2 * padding_y) / stride_y + 1;

    float* output = (float*) malloc(sizeof(float)*out_x*out_y*shapeInput[2]);
    memset(output, 0, sizeof(sizeof(float)*out_x*out_y*shapeInput[2]));
    maxpool(input, shapeInput[0], shapeInput[1], shapeInput[2], kernel_x, kernel_y, padding_x, padding_y, stride_x, stride_y, out_x, out_y, output);

    return output;
}

float* matmul_layer(Onnx__GraphProto* graph, const float *input, long* shapeInput, const char* layer_name)
{
    Onnx__NodeProto* node = onnx_graph_get_node_by_name(graph, layer_name);
    const char* weight = node->input[1];

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

    float* output = (float*) malloc(sizeof(float)*shapeW[1]);
    memset(output, 0, sizeof(sizeof(float)*shapeW[1]));
    matmul(input, W_t, shapeW[0], shapeW[1], output);

    free(W_t);

    return output;
}

float* add_layer(Onnx__GraphProto* graph, const float *input, long* shapeInput, const char* layer_name)
{
    Onnx__NodeProto* node = onnx_graph_get_node_by_name(graph, layer_name);
    const char* bias = node->input[1];

    float* B = onnx_graph_get_weights_by_name(graph, bias);
    long* shapeB =  onnx_graph_get_dims_by_name(graph, bias);
    if(shapeB == NULL)
    {
        return NULL;
    }

    float* output = (float*) malloc(sizeof(float)*shapeB[0]);
    memset(output, 0, sizeof(sizeof(float)*shapeB[0]));
    add(input, B, shapeB[0], output);

    return output;
}

float* softmax_layer(Onnx__GraphProto* graph, const float *input, long len)
{
    float* output = (float*) malloc(sizeof(float)*len);
    memset(output, 0, sizeof(sizeof(float)*len));
    softmax(input, len, output);

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
        if(img_index < 0 || img_index > TOTAL_IMAGE - 1)
        {
            img_index = 0;
        }
    }
    print_img(img[img_index]);

    // 1. Conv2D
    long shapeInput[] = {28, 28, 1};
    float* conv1 = conv2D_layer(model->graph, img[img_index], shapeInput, "conv2d_5");

    // 2. Relu
    relu(conv1, 28*28*2);

    // 3. Maxpool
    long shapeConv1[] = {28, 28, 2};
    float* maxpool1 = maxpool_layer(model->graph, conv1, shapeConv1, "max_pooling2d_5");
    free(conv1);

    // 4. Conv2D
    long shapeMaxpool1[] = {14, 14, 2};
    float* conv2 = conv2D_layer(model->graph, maxpool1, shapeMaxpool1, "conv2d_6");
    free(maxpool1);

    // 5. Relu
    // Input: N W H C
    relu(conv2, 14*14*2);

    // 6. Maxpool
    // Input: N W H C
    long shapeConv2[] = {14, 14, 2};
    float* maxpool2 = maxpool_layer(model->graph, conv2, shapeConv2, "max_pooling2d_6");
    free(conv2);

    // Flatten NOT REQUIRED

    // 7. Dense
    long shapeMaxpool2[] = {1, 98};
    float* matmul1 = matmul_layer(model->graph, maxpool2, shapeMaxpool2, "dense_5");
    free(maxpool2);
    long shapeMatmul1[] = {98};
    float* dense1 = add_layer(model->graph, matmul1, shapeMatmul1, "Add1");
    free(matmul1);

    // 8. Dense
    long shapeDense1[] = {1, 4};
    float* matmul2 = matmul_layer(model->graph, dense1, shapeDense1, "dense_6");
    free(dense1);
    long shapeMatmul2[] = {4};
    float* dense2 = add_layer(model->graph, matmul2, shapeMatmul2, "Add");
    free(matmul2);

    // 9. Softmax
    float* output = softmax_layer(model->graph, dense2, 10);
    free(dense2);

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

    // Free model
    free(output);
    onnx__model_proto__free_unpacked(model, NULL);

    return 0;
}
