#include "backend.h"

void add(const float *input,              // pointer to vector
         const float *bias,             // pointer to matrix
         const uint16_t dim_vec,         // length of the vector
         float *output)
{
    for (int i = 0; i < dim_vec; i++)
    {
        output[i] = input[i] + bias[i];
    }
}
