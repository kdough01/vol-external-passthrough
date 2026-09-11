# Basic round trip

This example creates a small 16 x 16 floating-point dataset, writes it through the external passthrough VOL, closes the file, reopens it, and reads the data back. The default compressor is `bzip2`; change `config.compressor` in `basic_roundtrip.c` to `noop` if your LibPressio installation does not provide `bzip2`.

## Build

From the repository root, use the HDF5 compiler wrapper and link against the already-built connector:

```sh
h5cc -I../../src basic_roundtrip.c -L../../build/lib -lhdf5_vol_passthrough -lpressio -lm -o basic_roundtrip
```

Adjust `../../build/lib` to the directory containing `libhdf5_vol_passthrough`.

## Run

The connector must be selected before HDF5 opens the file:

```sh
HDF5_PLUGIN_PATH=../../build/lib \
HDF5_VOL_CONNECTOR='pass_through_ext under_vol=0;under_info={}' \
./basic_roundtrip
```

Expected output is a compressor name followed by a near-zero error. The generated `basic_roundtrip.h5` stores the compressed payload and the metadata required to reopen it through the VOL.
