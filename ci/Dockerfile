FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    ninja-build \
    libhdf5-dev \
    hdf5-tools \
    libbz2-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/include/hdf5/serial /usr/include/hdf5_include \
    && ln -s /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so /usr/local/lib/libhdf5.so \
    && ln -s /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.so /usr/local/lib/libhdf5_hl.so \
    && ldconfig

WORKDIR /build/std_compat
 
RUN git clone --depth=1 https://github.com/robertu94/std_compat.git . \
    && cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_TESTING=OFF \
    && cmake --build build -j$(nproc) \
    && cmake --install build \
    && rm -rf /build/std_compat

WORKDIR /build/libpressio

RUN git clone --depth=1 https://github.com/robertu94/libpressio.git . \
    && cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=ON \
        -DLIBPRESSIO_HAS_SZ3=OFF \
        -DLIBPRESSIO_HAS_BZIP2=ON \
        -DLIBPRESSIO_HAS_JSON=ON \
        -DLIBPRESSIO_HAS_MPI=OFF \
        -DLIBPRESSIO_HAS_CUDA=OFF \
    && cmake --build build -j$(nproc) \
    && cmake --install build \
    && rm -rf /build/libpressio

RUN ldconfig

WORKDIR /workspace