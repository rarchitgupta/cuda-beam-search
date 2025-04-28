#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "whisper/beam_search/tensor_bridge.h"
#include "whisper/beam_search/beam_types.h"
#include "whisper/beam_search/beam_array.h"
#include "whisper/beam_search/logit_processor.h"
#include "whisper/utils/cuda_stream_manager.h"
#include <iostream>

namespace py = pybind11;

// Helper function to extract the raw CUDA pointer from a PyTorch tensor
void* get_tensor_data_ptr(py::object tensor) {
    auto data_ptr_obj = tensor.attr("data_ptr")();
    return reinterpret_cast<void*>(data_ptr_obj.cast<intptr_t>());
}

// Function to check if tensor is contiguous and on CUDA
bool check_tensor(py::object tensor, bool& is_half) {
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
    
    if (dtype_name.find("float32") != std::string::npos) {
        is_half = false;
        return true;
    } else if (dtype_name.find("float16") != std::string::npos) {
        is_half = true;
        return true;
    }
    
    throw std::runtime_error("Tensor must be of dtype float32 or float16");
    return false;
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
    
    // Bind BeamSearchConfig
    py::class_<whisper::beam_search::BeamSearchConfig>(m, "BeamSearchConfig")
        .def(py::init<>())
        .def_readwrite("beam_width", &whisper::beam_search::BeamSearchConfig::beam_width)
        .def_readwrite("temperature", &whisper::beam_search::BeamSearchConfig::temperature)
        .def_readwrite("top_k", &whisper::beam_search::BeamSearchConfig::top_k)
        .def_readwrite("top_p", &whisper::beam_search::BeamSearchConfig::top_p)
        .def_readwrite("max_length", &whisper::beam_search::BeamSearchConfig::max_length)
        .def_readwrite("stop_token_ids", &whisper::beam_search::BeamSearchConfig::stop_token_ids);
    
    // Bind CudaStreamManager
    py::class_<whisper::utils::CudaStreamManager>(m, "CudaStreamManager")
        .def(py::init<>())
        .def("synchronize", &whisper::utils::CudaStreamManager::Synchronize);
    
    // Create a typedef for the execute_beam_search overload we want to use in Python
    typedef bool (whisper::beam_search::TensorBridge::*execute_beam_search_t)(const whisper::beam_search::BeamSearchConfig&);
    
    // Bind TensorBridge with stream manager support
    py::class_<whisper::beam_search::TensorBridge>(m, "TensorBridge")
        .def(py::init<whisper::utils::CudaStreamManager*>(), py::arg("stream_manager") = nullptr)
        .def("set_logits_tensor", [](whisper::beam_search::TensorBridge& self, py::object tensor) {
            bool is_half = false;
            if (!check_tensor(tensor, is_half)) {
                return false;
            }
            
            auto shape = tensor.attr("shape").cast<std::vector<int>>();
            if (shape.size() != 3) {
                throw std::runtime_error("Logits tensor must have 3 dimensions [batch_size, seq_len, vocab_size]");
            }
            
            int batch_size = shape[0];
            int seq_len = shape[1];
            int vocab_size = shape[2];
            
            void* data_ptr = get_tensor_data_ptr(tensor);
            
            if (is_half) {
                return self.set_logits_half(data_ptr, batch_size, seq_len, vocab_size);
            } else {
                return self.set_logits(static_cast<float*>(data_ptr), batch_size, seq_len, vocab_size);
            }
        })
        .def("get_shape", &whisper::beam_search::TensorBridge::get_shape)
        .def("get_dtype", [](const whisper::beam_search::TensorBridge& self) {
            return self.get_dtype() == whisper::beam_search::TensorDType::FLOAT16 ? "float16" : "float32";
        })
        .def("get_stream_manager", &whisper::beam_search::TensorBridge::get_stream_manager, 
             py::return_value_policy::reference)
        .def("execute_beam_search", static_cast<execute_beam_search_t>(&whisper::beam_search::TensorBridge::execute_beam_search))
        .def("get_beam_search_results", &whisper::beam_search::TensorBridge::get_beam_search_results)
        .def("reset_beam_search", &whisper::beam_search::TensorBridge::reset_beam_search);
    
    // Bind LogitProcessor
    py::class_<whisper::beam_search::LogitProcessor>(m, "LogitProcessor")
        .def(py::init<whisper::beam_search::BeamSearchWorkspace*, float, int, float, whisper::utils::CudaStreamManager*>(),
             py::arg("workspace"),
             py::arg("temperature") = 1.0f,
             py::arg("top_k") = 0,
             py::arg("top_p") = 1.0f,
             py::arg("stream_manager") = nullptr)
        .def("process_logits", [](whisper::beam_search::LogitProcessor& self, py::object tensor) {
            bool is_half = false;
            if (!check_tensor(tensor, is_half)) {
                return false;
            }
            
            auto shape = tensor.attr("shape").cast<std::vector<int>>();
            if (shape.size() != 3) {
                throw std::runtime_error("Logits tensor must have 3 dimensions [batch_size, seq_len, vocab_size]");
            }
            
            int batch_size = shape[0];
            int seq_len = shape[1];
            int vocab_size = shape[2];
            
            void* data_ptr = get_tensor_data_ptr(tensor);
            
            if (is_half) {
                return self.ProcessLogitsHalf(data_ptr, batch_size, seq_len, vocab_size);
            } else {
                return self.ProcessLogits(static_cast<float*>(data_ptr), batch_size, seq_len, vocab_size);
            }
        })
        .def("score_next_tokens", &whisper::beam_search::LogitProcessor::ScoreNextTokens)
        .def("score_and_prune", &whisper::beam_search::LogitProcessor::ScoreAndPrune)
        .def("set_sampling_params", &whisper::beam_search::LogitProcessor::SetSamplingParams)
        .def("get_stream_manager", &whisper::beam_search::LogitProcessor::GetStreamManager, 
             py::return_value_policy::reference);
} 