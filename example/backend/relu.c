#include "backend.h"

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
