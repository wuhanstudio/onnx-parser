#include "onnx.h"

void relu(const float *input, uint32_t size, float* output)
{
    uint32_t i;
    memcpy(output, input, sizeof(float) * size);
    for (i = 0; i < size; i++)
    {
        if (output[i] < 0)
            output[i] = 0;
    }
}

float* relu_layer(Onnx__GraphProto* graph, const float *input, long* shapeInput, long* shapeOutput, const char* layer_name)
{
    assert(graph != NULL && input != NULL && layer_name != "" );

    long len = shapeInput[0] * shapeInput[1] * shapeInput[2];
    float* output = (float*) malloc(sizeof(float)*len);
    memset(output, 0, sizeof(sizeof(float)*len));

    relu(input, len, output);

    memcpy(shapeInput, shapeOutput, sizeof(long)*3);

    return output;
}
