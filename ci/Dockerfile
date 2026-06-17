FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    ninja-build \
    # HDF5
    libhdf5-dev \
    hdf5-tools \
    # bzip2 (pressio bzip2 compressor)
    libbz2-dev \
    # zfp
    libzfp-dev \
    # nlohmann-json (header-only, needed for pressio +json)
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build/libpressio

RUN git clone --depth=1 https://github.com/robertu94/libpressio.git . \
    && cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=ON \
        -DLIBPRESSIO_HAS_SZ3=OFF \
        -DLIBPRESSIO_HAS_BZIP2=ON \
        -DLIBPRESSIO_HAS_ZFP=ON \
        -DLIBPRESSIO_HAS_JSON=ON \
        -DLIBPRESSIO_HAS_MPI=OFF \
        -DLIBPRESSIO_HAS_CUDA=OFF \
    && cmake --build build -j$(nproc) \
    && cmake --install build \
    && rm -rf /build/libpressio

RUN ldconfig

WORKDIR /workspace