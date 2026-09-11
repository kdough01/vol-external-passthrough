# HDF5 Compression VOL

This repository contains an HDF5 Virtual Object Layer (VOL) connector that stores HDF5 datasets as LibPressio-compressed byte streams. It is designed for compressor-agnostic CPU compression and optional CUDA-aware compression, with support for connector-managed chunking, timing/metrics output, and progressive reads for SPERR.

The connector is a pass-through VOL: ordinary HDF5 applications continue to use `H5Fcreate`, `H5Dcreate2`, `H5Dwrite`, `H5Dread`, and the rest of the HDF5 API. The connector intercepts dataset creation, writes, opens, and reads, while the native VOL remains the underlying connector.

The [project wiki](https://github.com/kdough01/vol-external-passthrough/wiki) contains design notes and a longer code walk-through.

## Capabilities

- CPU compression through any compatible LibPressio compressor registered in the installation.
- Optional CUDA builds for device-resident compression paths (`-DUSE_CUDA=ON`).
- Dataset-level compressor selection and JSON options.
- Several storage strategies: native whole-dataset streams, VOL chunking, LibPressio chunking, shared metadata, and progressive SPERR storage.
- Full-fidelity round trips through standard HDF5 reads and writes.
- Progressive reads for the `sperr` compressor using a requested fidelity percentage.
- Optional LibPressio size/time metrics and connector timing CSV output.

## Repository map

- `src/H5VLpassthru_ext.c`: VOL callbacks, dataset metadata, compression dispatch, and plugin entry points.
- `src/H5VLpassthru_ext.h`: public connector registration macro and connector constants.
- `src/metadata_structs.h`: compression context and chunking state.
- `src/compress/compress.cc`: LibPressio compression/decompression and chunking implementations.
- `src/compress/vol_progressive.h`: progressive-read API for SPERR.
- `src/compress/vol_shared_meta.*`: shared metadata support, including Zstandard metadata sharing.
- `tests/bench_config.h`: benchmark dataset/configuration format and useful reference code.
- `tests/`: round-trip tests, timing programs, benchmark harnesses, and batch scripts.
- `examples/`: small, focused applications. The legacy top-level examples are built by CMake; the new examples are grouped into their own directories and can be compiled directly with the commands in their READMEs.

## Requirements

The normal build requires:

- HDF5 1.14 or a compatible HDF5 installation with VOL support.
- LibPressio and the LibPressio compressors you intend to use.
- CMake 3.18 or newer.
- A C and C++ compiler. MPI is optional for the connector, but the current test targets link MPI when it is found.
- SPERR development headers and library. The CMake build currently locates this through `pkg-config` as `SPERR`.
- CUDA toolkit only when building with `USE_CUDA=ON`.

The compressor name is not supplied by this repository. Confirm that a compressor such as `bzip2`, `zstd`, `zfp`, `sz3`, or `sperr` is present in the LibPressio installation before selecting it. `noop` is the built-in no-compression fallback and is useful for checking VOL wiring.

## Build with CMake

Configure and build from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Useful options are:

```sh
cmake -S . -B build \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_TESTS=ON \
	-DBUILD_EXAMPLES=ON \
	-DUSE_CUDA=OFF
```

Set `CMAKE_PREFIX_PATH` or the package-specific CMake hints when HDF5, LibPressio, or SPERR are installed outside the compiler's default search paths. For example:

```sh
cmake -S . -B build \
	-DCMAKE_PREFIX_PATH="$HDF5_PREFIX;$LIBPRESSIO_PREFIX;$SPERR_PREFIX"
```

The connector library is normally written to `build/lib` and executables to `build/bin`. The exact paths can be changed by CMake toolchain settings.

## Enable the connector

HDF5 selects VOL connectors through environment variables. The connector must be discoverable before the application makes its first HDF5 call that initializes the library:

```sh
export HDF5_PLUGIN_PATH="$PWD/build/lib"
export HDF5_VOL_CONNECTOR='pass_through_ext under_vol=0;under_info={}'
```

`HDF5_PLUGIN_PATH` must name the directory containing the built connector library. On a build that places the library directly in `build`, use `HDF5_PLUGIN_PATH="$PWD/build"` instead. `under_vol=0` selects the native HDF5 VOL as the underlying connector.

You can check that the connector is being selected by enabling a verbose CMake build definition or by running a small round trip such as `build/bin/test_ci`. The CI configuration uses the same two environment variables.

## Select a compressor

There are three configuration levels, with the most specific dataset setting taking precedence:

1. Set `HDF5_VOL_PRESSIO_COMPRESSOR` for the file's default compressor.
2. Set `HDF5_VOL_PRESSIO_LEVEL` for the default `<compressor>:compression_level` option.
3. Add `pressio:compressor` and `vol:options_json` to an individual dataset creation property list.

For example:

```sh
export HDF5_VOL_PRESSIO_COMPRESSOR=bzip2
export HDF5_VOL_PRESSIO_LEVEL=5
```

Applications that use per-dataset properties must register those properties on `H5P_DATASET_CREATE` before calling `H5Pset`:

```c
char compressor[64] = "bzip2";
char options_json[4096] = "{\"bzip2:compression_level\": 5}";

H5Pset(dcpl_id, "pressio:compressor", compressor);
H5Pset(dcpl_id, "vol:options_json", options_json);
```

The complete registration and creation pattern is shown in [per_dataset_options.c](examples/per_dataset_options/per_dataset_options.c). JSON is passed to LibPressio after the default compression level is applied, so JSON can override compressor-specific defaults.

## Dataset and file behavior

When compression is active, dataset creation stores the logical rank, dimensions, datatype, compressor, and effective options as hidden `_VOL_*` attributes. The underlying HDF5 dataset is an extendible, chunked byte dataset. Opening it through this VOL reconstructs the logical dataset and decompresses reads into the application's requested datatype.

Use the native VOL when inspecting the physical byte representation or reading a source file without compression. A native HDF5 file opened by this connector is treated as an ordinary uncompressed dataset unless it contains the connector's metadata.

The connector's `noop` compressor is useful for diagnosing setup. It still exercises the VOL path, but it does not provide compression.

## Chunking options

Chunking is configured inside `vol:options_json`:

```json
{
	"vol:chunking_mode": "vol",
	"vol:chunk_n": 4096
}
```

Supported modes in the connector are `vol`, `pressio` (also accepted as `libpressio`), and `shared`. The default is no connector chunking. `vol:chunk_n` is the number of logical elements per chunk; zero or an omitted value lets the connector choose its default based on the dataset shape and datatype. Chunking is especially important for progressive reads and large datasets because it bounds the amount of data needed for each read region.

## Progressive reads

Progressive reads currently require the `sperr` compressor and a chunked VOL representation. Create a transfer property list and request a percentage from 1 through 100:

```c
hid_t dxpl_id = H5Pcreate(H5P_DATASET_XFER);
H5Pset_vol_progressive_pct(dxpl_id, 25);
H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
				dxpl_id, output);
H5Pclose(dxpl_id);
```

The helper is declared in `src/compress/vol_progressive.h` and is exported by the connector library. `H5P_DEFAULT` means a full-fidelity read and cannot carry a progressive percentage. For harnesses that cannot create a DXPL, `VOL_PROGRESSIVE_PCT=25` provides a process-level fallback. Set `VOL_PROGRESSIVE_LOG=1` to log SPERR truncation activity.

See [progressive_read.c](examples/progressive_read/progressive_read.c) for an end-to-end example.

## Metrics and diagnostics

Set these variables before running the application when needed:

```sh
export HDF5_VOL_PRESSIO_METRICS=1  # LibPressio size/time metrics
export VOL_COMP_CHUNK_LOG=1        # connector chunk decisions
export VOL_PROGRESSIVE_LOG=1       # SPERR progressive truncation
```

The benchmark programs in `tests/` expose additional timing controls and CSV output. They are intended for large datasets and batch systems, not as the shortest path to validating a local installation.

## Run tests

After building and exporting the connector variables, run the normal CTest suite:

```sh
ctest --test-dir build --output-on-failure
```

The small CI smoke test can also be run directly:

```sh
HDF5_VOL_PRESSIO_COMPRESSOR=noop ./build/bin/test_ci
```

For a compressor round trip, use a compressor available in your LibPressio build. The CI test currently enables `noop` and `bzip2`; lossy `sz3` and `zfp` cases require tolerances appropriate to their selected options.

## Examples

- [Basic round trip](examples/basic_roundtrip/README.md): write, close, reopen, and verify a compressed dataset.
- [Per-dataset options](examples/per_dataset_options/README.md): choose a compressor and JSON options independently for each dataset.
- [Progressive SPERR read](examples/progressive_read/README.md): request a reduced-fidelity read with a dataset transfer property list.

The example READMEs use `h5cc` and the built connector library. If your HDF5 installation does not provide `h5cc`, replace it with your HDF5 compiler and linker flags.

## Troubleshooting

**`H5Fcreate` or `H5Dcreate2` fails immediately:** verify `HDF5_VOL_CONNECTOR`, confirm that `HDF5_PLUGIN_PATH` contains the connector library, and make sure the HDF5 runtime and connector were built against compatible HDF5 versions.

**`compressor ... not found in libpressio registry`:** the selected compressor is not installed or its LibPressio plugin is not visible. Try `HDF5_VOL_PRESSIO_COMPRESSOR=noop` to isolate VOL loading from compressor availability.

**A custom property is missing:** register `pressio:compressor` and `vol:options_json` on `H5P_DATASET_CREATE` before setting them. The examples show the required `H5Pregister2` calls.

**Progressive read is rejected:** use `sperr`, create the dataset with VOL chunking, and pass a real `H5P_DATASET_XFER` property list. A progressive request with another compressor is intentionally unsupported.

**CUDA data does not stay on the device:** build with `-DUSE_CUDA=ON`, use a LibPressio compressor that supports the connector's CUDA stream option, and inspect the connector logs and timing output. CPU builds remain valid for ordinary host buffers.

## Citation

Add the project paper citation here when it is published. Until then, cite this repository and include the commit used for your experiment.