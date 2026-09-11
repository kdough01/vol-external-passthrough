# Per-dataset LibPressio options

This example demonstrates that compressor selection and LibPressio options can be stored in each dataset creation property list. It creates two datasets with the same `bzip2` compressor but different compression levels. The connector snapshots the effective options in dataset metadata so they can be replayed when the dataset is opened later.

The important properties are:

- `pressio:compressor`: the LibPressio compressor identifier.
- `vol:options_json`: JSON passed to LibPressio, including connector chunking options such as `vol:chunking_mode` and `vol:chunk_n`.

## Build and run

```sh
h5cc -I../../src per_dataset_options.c \
  -L../../build/lib -lhdf5_vol_passthrough -lpressio -lm \
  -o per_dataset_options
HDF5_PLUGIN_PATH=../../build/lib \
HDF5_VOL_CONNECTOR='pass_through_ext under_vol=0;under_info={}' \
./per_dataset_options
```

Use `h5dump -A per_dataset_options.h5` to inspect the hidden `_VOL_*` metadata written by the connector. The exact compression ratio depends on the installed LibPressio `bzip2` implementation.
