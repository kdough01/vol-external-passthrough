# Progressive read with SPERR

This example writes a 32 x 32 x 32 float field with the `sperr` LibPressio compressor, then reads it twice: once at full fidelity and once at 25 percent requested fidelity. The reduced-fidelity request is attached to a real HDF5 dataset transfer property list with `H5Pset_vol_progressive_pct`; `H5P_DEFAULT` cannot carry this request.

The dataset uses VOL chunking with `vol:chunk_n` so each compressed chunk can be truncated independently. Progressive reads are supported by the `sperr` path; other compressors are rejected for reduced-fidelity requests.

## Build and run

```sh
h5cc -I../../src -I../../src/compress progressive_read.c \
  -L../../build/lib -lhdf5_vol_passthrough -lpressio -lsperr -lm \
  -o progressive_read
HDF5_PLUGIN_PATH=../../build/lib \
HDF5_VOL_CONNECTOR='pass_through_ext under_vol=0;under_info={}' \
VOL_PROGRESSIVE_LOG=1 \
./progressive_read
```

The output reports RMS error for both reads. The full read should have the normal SPERR reconstruction error; the 25 percent read is intentionally less faithful and should require less compressed-stream data. This example requires a LibPressio build with SPERR support.
