CXX := /usr/local/cuda-13.0/bin/nvcc

gpu_demos : gpu_demos.cu
	$(CXX) -o gpu_demos gpu_demos.cu
