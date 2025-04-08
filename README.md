# CUDA Beam Search for Whisper

A CUDA-accelerated beam search implementation for OpenAI's Whisper model.

## Project Structure

```
cuda-beam-search/
├── include/whisper/
│   ├── decoder/
│   │   └── beam_search.h
│   └── utils/
│       └── cuda_helpers.h
├── src/whisper/
│   ├── decoder/
│   │   ├── beam_search.cc
│   │   └── beam_search.cu
│   └── utils/
│       └── cuda_helpers.cu
├── python/           # Python bindings
├── tests/           # Test cases
├── examples/        # Example usage
└── CMakeLists.txt
```

## Dependencies

- CUDA Toolkit (11.2.1 or greater)
- CMake (3.18 or greater)
- C++17 compatible compiler
- PyTorch (for Whisper model)
- OpenAI Whisper

## Building

```bash
mkdir build
cd build
cmake ..
make -j
```

## Usage

1. Extract logits from Whisper model
2. Run CUDA beam search on the logits
3. Decode the resulting tokens back to text

## Performance Targets

- Target: 1.5x-3x speedup over CPU beam search
- Focus: Beam search step only (not full model)
- Hardware: RTX 4060 Ti (16GB) for development

## License

MIT License 