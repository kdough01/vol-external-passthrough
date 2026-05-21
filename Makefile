#Your HDF5 install path
HDF5_DIR=../build_hdf5/hdf5
HDF5_DIR=/home/kevindougherty.guest/spack/opt/spack/linux-m1/hdf5-1.14.6-wulcdf6scirrbv5slmvczpdudzchageg
MPI_DIR=/usr/local
LIBPRESSIO_DIR=/home/kevindougherty.guest/spack/opt/spack/linux-m1/libpressio-1.0.6-ijzws7v5tcpa3t7wrhfjojurximk3kx2

CC=mpicc
# CC=gcc
AR=ar

ifdef USE_CUDA
    NVCC=nvcc
    CUDA_DIR=/usr/local/cuda
    CUDA_INCLUDES=-I$(CUDA_DIR)/include
    CUDA_LIBS=-L$(CUDA_DIR)/lib64 -lcudart
    CUDA_FLAGS=-DUSE_CUDA
    CUDA_SRC=gpu_transform.cu
else
    CUDA_INCLUDES=
    CUDA_LIBS=
    CUDA_FLAGS=
    CUDA_SRC=
endif

DYNSRC = H5VLpassthru_ext.c cpu_compress.c $(CUDA_SRC)

DEBUG=-DENABLE_EXT_PASSTHRU_LOGGING -g -O0
#INCLUDES=-I$(MPI_DIR)/include -I$(HDF5_DIR)/include
INCLUDES=-I$(HDF5_DIR)/include -I$(MPI_DIR)/include -I$(LIBPRESSIO_DIR)/include $(CUDA_INCLUDES)
CFLAGS = $(DEBUG) -fPIC $(INCLUDES) $(CUDA_FLAGS) -Wall
#LIBS=-L$(HDF5_DIR)/lib -L$(MPI_DIR)/lib -lhdf5 -lz
LIBS=-L$(HDF5_DIR)/lib -L$(MPI_DIR)/lib -L$(LIBPRESSIO_DIR)/lib64 -lhdf5 -lz -llibpressio $(CUDA_LIBS)
# Uncomment this line Linux builds:
DYNLDFLAGS = $(DEBUG) -shared -fPIC $(LIBS)
# Uncomment this line MacOS builds:
# DYNLDFLAGS = $(DEBUG) -dynamiclib -current_version 1.0 -fPIC $(LIBS)
LDFLAGS = $(DEBUG) $(LIBS)
ARFLAGS = rs

DYNSRC = H5VLpassthru_ext.c cpu_compress.c $(CUDA_SRC)
DYNOBJ = $(DYNSRC:.c=.o)
# Uncomment this line Linux builds:
DYNLIB = libh5passthrough_vol.so
# Uncomment this line MacOS builds:
# DYNLIB = libh5passthrough_vol.dylib
DYNDBG = libh5passthrough_vol.dylib.dSYM

STATSRC = new_h5api.c
STATOBJ = $(STATSRC:.c=.o)
STATLIB = libnew_h5api.a

EXSRC = new_h5api_ex.c
EXOBJ = $(EXSRC:.c=.o)
EXEXE = new_h5api_ex.exe
EXDBG = new_h5api_ex.exe.dSYM

ASYNC_EXSRC = async_new_h5api_ex.c
ASYNC_EXOBJ = $(ASYNC_EXSRC:.c=.o)
ASYNC_EXEXE = async_new_h5api_ex.exe
ASYNC_EXDBG = async_new_h5api_ex.exe.dSYM

DATAFILE = testfile.h5

all: $(EXEXE) $(ASYNC_EXEXE) $(DYNLIB) $(STATLIB)

$(EXEXE): $(EXSRC) $(STATLIB) $(DYNLIB)
	$(CC) $(CFLAGS) $^ -o $(EXEXE) $(LDFLAGS) -L. -lnew_h5api

$(ASYNC_EXEXE): $(ASYNC_EXSRC) $(STATLIB) $(DYNLIB)
	$(CC) $(CFLAGS) $^ -o $(ASYNC_EXEXE) $(LDFLAGS) -L. -lnew_h5api

$(DYNLIB): $(DYNSRC)
	$(CC) $(CFLAGS) $(DYNLDFLAGS) $^ -o $@

ifdef USE_CUDA
%.o: %.cu
	$(NVCC) -c $(CUDA_FLAGS) $(INCLUDES) $< -o $@
endif

$(STATOBJ): $(STATSRC)
	$(CC) -c $(CFLAGS) $^ -o $(STATOBJ)

$(STATLIB): $(STATOBJ)
	$(AR) $(ARFLAGS) $@ $^

.PHONY: clean all
clean:
	rm -rf $(DYNOBJ) $(DYNLIB) $(DYNDBG) \
            $(STATOBJ) $(STATLIB) \
            $(EXOBJ) $(EXEXE) $(EXDBG) \
            $(ASYNC_EXOBJ) $(ASYNC_EXEXE) $(ASYNC_EXDBG) \
            $(DATAFILE)
