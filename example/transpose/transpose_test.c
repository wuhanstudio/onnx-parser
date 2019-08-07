#include <stdio.h>  
#include <malloc.h>

#include "backend.h"

int main(int argc, char const *argv[])
{
    // ------------------------------------------------------
    // Original tensor A
    float A[24] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        
        13, 14, 15, 16,
        17, 18, 19, 20,
        21, 22, 23, 24
    };
    int shapeA[] = {2, 3, 4};
    int dimA = 3;

    // Print A
    onnx_tensor_info(A, shapeA, dimA);

    // Transpose
    int perm[] = { 2, 0, 1};
    float* B = onnx_tensor_transpose(A, shapeA, dimA, perm);

    // Print B
    int shapeB[] = {4, 2, 3};
    int dimB = 3;
    onnx_tensor_info(B, shapeB, dimB);

    // Free memory
    free(B);

    return 0;
}

