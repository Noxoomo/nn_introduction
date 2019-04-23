FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update -q -y && apt-get install -y -q zlib1g-dev wget curl unzip python3.7 python3-distutils

RUN curl https://cmake.org/files/v3.13/cmake-3.13.2-Linux-x86_64.sh --output /cmake-3.13.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.13.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN ln -s /opt/cmake/bin/ctest /usr/local/bin/ctest

RUN wget https://download.pytorch.org/libtorch/nightly/cu100/libtorch-shared-with-deps-latest.zip -P /
RUN apt-get install -y -q 
RUN unzip libtorch-shared-with-deps-latest.zip
RUN wget http://bitbucket.org/eigen/eigen/get/3.3.7.zip
RUN unzip 3.3.7.zip
RUN mv /eigen* /eigen
RUN mkdir /eigen/build
RUN cd /eigen/build/ && cmake .. && make install

COPY . /app
WORKDIR /app/build
RUN cmake .. -DCMAKE_PREFIX_PATH=/libtorch -DCMAKE_BUILD_TYPE=Release
RUN make soft_tree_ut
#WORKDIR /app/build/cpp/apps/cifar_networks 
#ENTRYPOINT ./resnet CUDA
