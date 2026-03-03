NVCC ?= nvcc

# Detect GPU architecture: default to sm_80 (A100), override with CUDA_ARCH=sm_90 for H100
CUDA_ARCH ?= sm_80

NVCCFLAGS = -O3 -arch=$(CUDA_ARCH) -shared -Xcompiler -fPIC --std=c++17

TARGET = libmps_decoder.so

all: $(TARGET)

$(TARGET): mps_decoder.cu mps_decoder.h
	$(NVCC) $(NVCCFLAGS) -o $@ mps_decoder.cu

clean:
	rm -f $(TARGET)

.PHONY: all clean
