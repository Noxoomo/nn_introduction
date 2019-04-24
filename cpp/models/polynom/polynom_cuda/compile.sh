/usr/local/cuda-10.0/bin/nvcc  -c -o libsoft_polynom.so  --ptxas-options=-v --compiler-options '-fPIC' -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70   -gencode arch=compute_61,code=compute_61 -gencode arch=compute_70,code=compute_70  --shared soft_polynom.cu

