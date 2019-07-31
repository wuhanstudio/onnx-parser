#include "onnx-parser.h"

const char* onnx_tensor_proto_data_type[] = {
    "Undefined", 
    "FLOAT",
    "UINT8",
    "INT8",
    "UINT16",
    "INT16",
    "INT32",
    "INT64",
    "STRING",
    "BOOL",
    "FLOAT16",
    "DOUBLE",
    "UINT32",
    "UINT64",
    "COMPLEX64",
    "COMPLEX128"
};

Onnx__ModelProto* onnx_load_model(const char* onnx_file_name)
{
    unsigned char* buffer;
    FILE *fp;

    // Get File Size
    fp = fopen(onnx_file_name,"rb"); 
    fseek(fp, 0L, SEEK_END);
    int sz = ftell(fp);
    fseek(fp, 0L, SEEK_SET);
    // printf("File size %s is %d\n", onnx_file_name, sz);

    // Read File
    buffer = (unsigned char*) malloc(sizeof(unsigned char) * sz);
    if(buffer == NULL)
    {
        printf("Failed to malloc %d bytes memory for %s\n", sz, onnx_file_name);
        return NULL;
    }
    fread(buffer, sz, 1, fp);

    return onnx__model_proto__unpack(NULL, sz, buffer);
}

void onnx_model_info(Onnx__ModelProto model)
{
    printf("---- Model info ----\n");
    printf("IR Version is %ld\n", model.ir_version);
    printf("Produceer name is %s\n", model.producer_name);
    printf("Produceer version is %s\n", model.producer_version);
    printf("Produceer version is %s\n", model.domain);
}

void onnx_graph_info(Onnx__GraphProto graph)
{
    printf("---- Graph Info ----\n");
    
    printf("Graph inputs number: %ld\n", graph.n_input);
    for(int i = 0; i < graph.n_input; i++)
    {
        onnx_graph_input_info(*graph.input[i]);
    }

    printf("Graph outputs number: %ld\n", graph.n_output);
    for(int i = 0; i < graph.n_output; i++)
    {
        onnx_graph_output_info(*graph.output[i]);
    }
}

void onnx_graph_input_info(Onnx__ValueInfoProto input)
{
    printf("---- Graph Input Info ----\n");
    printf("Input name %s\n", input.name);

    Onnx__TypeProto type = *(input.type);
    Onnx__TypeProto__Tensor tensor_type = *(type.tensor_type);
    Onnx__TensorShapeProto shape = *(tensor_type.shape);

    printf("Input type %s\n", onnx_tensor_proto_data_type[tensor_type.elem_type]);
    printf("Input dimension %ld\n", shape.n_dim);
    
    for(int i = 0; i < shape.n_dim; i++)
    {
        onnx_graph_value_tensor_shape_dimension_info(*(shape.dim[i]));
        if( i != shape.n_dim - 1)
        {
            printf(" x ");
        }
    }
    printf("\n");
}

void onnx_graph_value_tensor_shape_dimension_info(Onnx__TensorShapeProto__Dimension dim)
{
    
    switch (dim.value_case)
    {
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE__NOT_SET:
            printf("?");
            break;
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE:
            printf("%ld",dim.dim_value);
            break;
        case ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_PARAM:
            printf("%s",dim.dim_param);
            break;
        default:
            printf("?");
            break;
    }
}

void onnx_graph_output_info(Onnx__ValueInfoProto output)
{
    printf("---- Graph Output Info ----\n");
    printf("Output name %s\n", output.name);

    Onnx__TypeProto type = *(output.type);
    Onnx__TypeProto__Tensor tensor_type = *(type.tensor_type);
    Onnx__TensorShapeProto shape = *(tensor_type.shape);

    printf("Output type %s\n", onnx_tensor_proto_data_type[tensor_type.elem_type]);
    printf("Output dimension %ld\n", shape.n_dim);
    
    for(int i = 0; i < shape.n_dim; i++)
    {
        onnx_graph_value_tensor_shape_dimension_info(*(shape.dim[i]));
        if( i != shape.n_dim - 1)
        {
            printf(" x ");
        }
    }
    printf("\n");
}