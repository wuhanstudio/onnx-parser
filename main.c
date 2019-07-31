#include <stdio.h>  
#include "onnx-parser.h"

int main(int argc, char const *argv[])
{
    if (argc < 2) {
        printf("Usage: %s onnx_file_name \n", argv[0]);
        return 0;
    }
    printf("--- Reading from %s ---\n", argv[1]);

    // Load Model
    Onnx__ModelProto* model = onnx_load_model(argv[1]);

    // Print Model Info
    if( model != NULL)
    {
        onnx_model_info(*model);
    }

    // Print Graph Info
    Onnx__GraphProto* graph = model->graph;
    if(graph != NULL)
    {
        onnx_graph_info_sorted(*graph);
    }

    // Free Model
    onnx__model_proto__free_unpacked(model, NULL);

    return 0;
}
