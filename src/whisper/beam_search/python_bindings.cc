#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "whisper/beam_search/tensor_bridge.h"
#include <iostream>

namespace py = pybind11;

// Helper function to extract the raw CUDA pointer from a PyTorch tensor
void* get_tensor_data_ptr(py::object tensor) {
    auto data_ptr_obj = tensor.attr("data_ptr")();
    return reinterpret_cast<void*>(data_ptr_obj.cast<intptr_t>());
}

// Function to check if tensor is contiguous and on CUDA
bool check_tensor(py::object tensor) {
    // Check if tensor is on CUDA
    bool is_cuda = tensor.attr("is_cuda").cast<bool>();
    if (!is_cuda) {
        throw std::runtime_error("Tensor must be on CUDA device");
    }
    
    // Check if tensor is contiguous
    bool is_contiguous = tensor.attr("is_contiguous")().cast<bool>();
    if (!is_contiguous) {
        throw std::runtime_error("Tensor must be contiguous");
    }
    
    // Check data type
    auto dtype = tensor.attr("dtype");
    auto dtype_name = py::str(dtype).cast<std::string>();
    if (dtype_name.find("float32") == std::string::npos) {
        throw std::runtime_error("Tensor must be of dtype float32");
    }
    
    return true;
}

PYBIND11_MODULE(cuda_beam_search, m) {
    m.doc() = "Python bindings for CUDA beam search decoder";
    
    py::class_<whisper::beam_search::TensorBridge>(m, "TensorBridge")
        .def(py::init<>())
        .def("set_logits_tensor", [](whisper::beam_search::TensorBridge& self, py::object tensor) {
            // Validate tensor
            if (!check_tensor(tensor)) {
                return false;
            }
            
            // Get shape information
            auto shape = tensor.attr("shape").cast<std::vector<int>>();
            if (shape.size() != 3) {
                throw std::runtime_error("Logits tensor must have 3 dimensions [batch_size, seq_len, vocab_size]");
            }
            
            int batch_size = shape[0];
            int seq_len = shape[1];
            int vocab_size = shape[2];
            
            // Get data pointer
            float* data_ptr = static_cast<float*>(get_tensor_data_ptr(tensor));
            
            // Set data in TensorBridge
            return self.set_logits(data_ptr, batch_size, seq_len, vocab_size);
        })
        .def("get_shape", &whisper::beam_search::TensorBridge::get_shape);
} 