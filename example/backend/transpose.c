#include "onnx.h"

float* transpose(const float* A, long* shape, long dim, long* perm)
{
    // Get array size
    int elem = 1;
    for(int i = 0; i < dim; i++)
    {
        elem = elem * shape[i];
    }

    // Malloc memory for B
    float* B = malloc(sizeof(float) * elem);
    if(B == NULL)
    {
        return NULL;
    }

    // Malloc memory for shapeB
    int* shapeB = malloc(sizeof(int) * dim);
    if( shapeB == NULL)
    {
        return NULL;
    }
    for(int i = 0; i < dim; i++)
    {
        shapeB[i] = shape[perm[i]];
    }

    // Transpose
    for(int src = 0; src < elem; src++)
    {
        // Get transposed B array
        // A[1][0][3] -> B[3][1][0]
        int temp = src;
        int* indexA = malloc(sizeof(int) * dim);
        if(indexA == NULL)
        {
            return NULL;
        }
        int* indexB = malloc(sizeof(int) * dim);
        if(indexB == NULL)
        {
            return NULL;
        }
        for(int i = dim-1; i >= 0; i--)
        {
            indexA[i] = temp % shape[i];
            temp = temp / shape[i];
        }
        for(int i = 0; i < dim; i++)
        {
            indexB[i] = indexA[perm[i]];
        }

        // Get transposed B index 
        // #15 A[1][0][3] -> B[3][1][0] #21
        int dst = 0;
        temp = 1;
        for(int i = dim - 1; i >= 0; i--)
        {
            dst = dst + indexB[i] * temp;
            temp = temp * shapeB[i];
        }

        B[dst] = A[src];

        free(indexA);
        free(indexB);
    }

    free(shapeB);

    return B;
}

float* transpose_layer(Onnx__GraphProto* graph, const float *input, long* shapeInput, long* shapeOutput, const char* layer_name)
{
    assert(graph != NULL && input != NULL && layer_name != "" );

    Onnx__NodeProto* node = onnx_graph_get_node_by_name(graph, layer_name);
    if(node == NULL)
    {
        return NULL;
    }

    long perm_t[3];
    long* perm = node->attribute[0]->ints;
    perm_t[0] = perm[1] - 1;
    perm_t[1] = perm[2] - 1;
    perm_t[2] = perm[3] - 1;

    float* output = transpose(input, shapeInput, 3, perm_t);

    shapeOutput[0] = shapeInput[perm_t[0]];
    shapeOutput[1] = shapeInput[perm_t[1]];
    shapeOutput[2] = shapeInput[perm_t[2]];

    return output;
}
