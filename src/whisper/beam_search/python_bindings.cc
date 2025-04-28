#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "whisper/beam_search/tensor_bridge.h"
#include "whisper/beam_search/beam_types.h"
#include "whisper/beam_search/beam_array.h"
#include "whisper/beam_search/logit_processor.h"
#include <iostream>

namespace py = pybind11;

// Helper function to extract the raw CUDA pointer from a PyTorch tensor
void* get_tensor_data_ptr(py::object tensor) {
    auto data_ptr_obj = tensor.attr("data_ptr")();
    return reinterpret_cast<void*>(data_ptr_obj.cast<intptr_t>());
}

// Function to check if tensor is contiguous and on CUDA
bool check_tensor(py::object tensor) {
    bool is_cuda = tensor.attr("is_cuda").cast<bool>();
    if (!is_cuda) {
        throw std::runtime_error("Tensor must be on CUDA device");
    }
    
    bool is_contiguous = tensor.attr("is_contiguous")().cast<bool>();
    if (!is_contiguous) {
        throw std::runtime_error("Tensor must be contiguous");
    }
    
    auto dtype = tensor.attr("dtype");
    auto dtype_name = py::str(dtype).cast<std::string>();
    if (dtype_name.find("float32") == std::string::npos) {
        throw std::runtime_error("Tensor must be of dtype float32");
    }
    
    return true;
}

PYBIND11_MODULE(cuda_beam_search, m) {
    m.doc() = "Python bindings for CUDA beam search decoder";
    
    // Bind Token struct
    py::class_<whisper::beam_search::Token>(m, "Token")
        .def(py::init<>())
        .def(py::init<float, int, int>())
        .def_readwrite("score", &whisper::beam_search::Token::score)
        .def_readwrite("token_id", &whisper::beam_search::Token::token_id)
        .def_readwrite("prev_index", &whisper::beam_search::Token::prev_index);
    
    // Bind BeamSearchWorkspace
    py::class_<whisper::beam_search::BeamSearchWorkspace>(m, "BeamSearchWorkspace")
        .def(py::init<size_t>(), py::arg("initial_size") = 16 * 1024 * 1024)
        .def("reset", &whisper::beam_search::BeamSearchWorkspace::Reset)
        .def("get_used_size", &whisper::beam_search::BeamSearchWorkspace::GetUsedSize)
        .def("get_capacity", &whisper::beam_search::BeamSearchWorkspace::GetCapacity);
    
    // Bind BeamArray
    py::class_<whisper::beam_search::BeamArray>(m, "BeamArray")
        .def(py::init<size_t, whisper::beam_search::BeamSearchWorkspace*>())
        .def("reset", &whisper::beam_search::BeamArray::Reset)
        .def("size", &whisper::beam_search::BeamArray::Size)
        .def("capacity", &whisper::beam_search::BeamArray::Capacity)
        .def("add_token", &whisper::beam_search::BeamArray::AddToken)
        .def("sort_by_score", &whisper::beam_search::BeamArray::SortByScore)
        .def("prune", &whisper::beam_search::BeamArray::Prune)
        .def("get_token", &whisper::beam_search::BeamArray::GetToken)
        .def("copy_to_host", [](const whisper::beam_search::BeamArray& self) {
            std::vector<whisper::beam_search::Token> tokens;
            self.CopyToHost(tokens);
            return tokens;
        });
    
    // Bind TensorBridge
    py::class_<whisper::beam_search::TensorBridge>(m, "TensorBridge")
        .def(py::init<>())
        .def("set_logits_tensor", [](whisper::beam_search::TensorBridge& self, py::object tensor) {
            if (!check_tensor(tensor)) {
                return false;
            }
            
            auto shape = tensor.attr("shape").cast<std::vector<int>>();
            if (shape.size() != 3) {
                throw std::runtime_error("Logits tensor must have 3 dimensions [batch_size, seq_len, vocab_size]");
            }
            
            int batch_size = shape[0];
            int seq_len = shape[1];
            int vocab_size = shape[2];
            
            float* data_ptr = static_cast<float*>(get_tensor_data_ptr(tensor));
            
            return self.set_logits(data_ptr, batch_size, seq_len, vocab_size);
        })
        .def("get_shape", &whisper::beam_search::TensorBridge::get_shape);
    
    // Bind LogitProcessor
    py::class_<whisper::beam_search::LogitProcessor>(m, "LogitProcessor")
        .def(py::init<whisper::beam_search::BeamSearchWorkspace*, float, int, float>(),
             py::arg("workspace"),
             py::arg("temperature") = 1.0f,
             py::arg("top_k") = 0,
             py::arg("top_p") = 1.0f)
        .def("process_logits", [](whisper::beam_search::LogitProcessor& self, py::object tensor) {
            if (!check_tensor(tensor)) {
                return false;
            }
            
            auto shape = tensor.attr("shape").cast<std::vector<int>>();
            if (shape.size() != 3) {
                throw std::runtime_error("Logits tensor must have 3 dimensions [batch_size, seq_len, vocab_size]");
            }
            
            int batch_size = shape[0];
            int seq_len = shape[1];
            int vocab_size = shape[2];
            
            float* data_ptr = static_cast<float*>(get_tensor_data_ptr(tensor));
            
            return self.ProcessLogits(data_ptr, batch_size, seq_len, vocab_size);
        })
        .def("score_next_tokens", &whisper::beam_search::LogitProcessor::ScoreNextTokens)
        .def("score_and_prune", &whisper::beam_search::LogitProcessor::ScoreAndPrune)
        .def("set_sampling_params", &whisper::beam_search::LogitProcessor::SetSamplingParams);
} 