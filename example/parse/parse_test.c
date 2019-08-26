#include <stdio.h>  
#include <onnx-parser.h>

int main(int argc, char const *argv[])
{
    if (argc < 2) {
        printf("Usage: %s onnx_file_name \n", argv[0]);
        return 0;
    }
    printf("--- Reading from %s ---\n", argv[1]);
    printf("\n");

    // Load Model
    Onnx__ModelProto* model = onnx_load_model(argv[1]);

    // Print Model Info
    if( model != NULL)
    {
        onnx_model_info(model);
    }
    printf("\n");

    // Print Graph Info
    Onnx__GraphProto* graph = model->graph;
    if(graph != NULL)
    {
        onnx_graph_info_sorted(graph);
    }
    printf("\n");

    // Print Selected Node
    // onnx_graph_node_info(onnx_graph_get_node_by_name(graph, "Transpose6"));

    // Print Weights
    // Onnx__TensorProto** initializer =  graph->initializer;
    // for(int i = 0; i < graph->n_initializer; i++)
    // {
    //     onnx_graph_initializer_info(initializer[i]);
    // } 
    // printf("\n");

    // Free Model
    onnx__model_proto__free_unpacked(model, NULL);

    return 0;
}

