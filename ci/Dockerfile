FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    libbz2-dev \
    zlib1g-dev \
    python3 \
    python3-pip \
    ninja-build \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

# Build hdf5
ARG HDF5_VERSION=1.14.3
WORKDIR /build/hdf5

RUN wget -q \
    https://github.com/HDFGroup/hdf5/releases/download/hdf5_${HDF5_VERSION}/hdf5-${HDF5_VERSION}.tar.gz \
    && tar -xzf hdf5-${HDF5_VERSION}.tar.gz \
    && cmake -S hdf5-${HDF5_VERSION} -B hdf5-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DHDF5_ENABLE_PARALLEL=OFF \
        -DHDF5_BUILD_HL_LIB=ON \
        -DHDF5_BUILD_TOOLS=ON \
    && cmake --build hdf5-build -j$(nproc) \
    && cmake --install hdf5-build \
    && rm -rf /build/hdf5

# Build ZFP
WORKDIR /build/zfp

RUN git clone --depth=1 --branch 1.0.1 https://github.com/LLNL/zfp.git . \
    && cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=ON \
    && cmake --build build -j$(nproc) \
    && cmake --install build \
    && rm -rf /build/zfp

# Build SZ3
WORKDIR /build/sz3

RUN git clone --depth=1 https://github.com/szcompressor/SZ3.git . \
    && cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=ON \
    && cmake --build build -j$(nproc) \
    && cmake --install build \
    && rm -rf /build/sz3

# Build json
WORKDIR /build/json

RUN git clone --depth=1 --branch v3.11.3 \
    https://github.com/nlohmann/json.git . \
    && cmake -S . -B build \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DJSON_BuildTests=OFF \
    && cmake --install build \
    && rm -rf /build/json

# Build libpressio with sz3, bzip2, zfp, json
WORKDIR /build/libpressio

RUN git clone --depth=1 https://github.com/robertu94/libpressio.git . \
    && cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_SHARED_LIBS=ON \
        -DLIBPRESSIO_HAS_SZ3=ON \
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