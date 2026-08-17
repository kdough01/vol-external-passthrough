#!/bin/bash
#PBS -N vol_progressive_sperr
#PBS -l select=1:ncpus=16:ngpus=1:mpiprocs=1:mem=256gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -A SDR
#PBS -q gpu

set -u

cd $PBS_O_WORKDIR/..
WORK="$(pwd)/tests"

RESULTS="$WORK/results/prog_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID%%.*}"
mkdir -p "$RESULTS"
ln -sfn "$RESULTS" "$WORK/results/latest_prog"

echo "Job started on $(date) / $(hostname)"
echo "Results: $RESULTS"

module load gcc/11.4.0
module load hpcx/2.26-gcc
module load cuda/12.6.0

SYSTEM_MPICC=$(which mpicc)
SYSTEM_MPICXX=$(which mpicxx)
GCC_LIB=$(gcc --print-file-name=libstdc++.so | xargs dirname 2>/dev/null)
[ "$GCC_LIB" = "." ] && GCC_LIB=/usr/lib/gcc/x86_64-linux-gnu/11

LP_VIEW=/home/kdougherty/libpressio_cuda/.spack-env/view
SPERR_PREFIX="${SPERR_PREFIX:-/gpfs/fs1/home/kdougherty/spack/opt/spack/linux-zen2/sperr-0.8.2-qsqz7rfwktbmkju43ucbrmtpiyn6wyb6}"
[ -d "$SPERR_PREFIX/include" ] || { echo "ERROR: bad SPERR_PREFIX"; exit 1; }

LP_CMAKE_DIR=""
for candidate in "${LP_VIEW}/lib64/cmake/LibPressio" "${LP_VIEW}/lib/cmake/LibPressio" \
                 "${LP_VIEW}/lib64/cmake/libpressio" "${LP_VIEW}/lib/cmake/libpressio"; do
    if [ -f "${candidate}/LibPressioConfig.cmake" ] || \
       [ -f "${candidate}/libpressio-config.cmake" ]; then LP_CMAKE_DIR="$candidate"; break; fi
done
[ -n "$LP_CMAKE_DIR" ] || { echo "ERROR: no LibPressioConfig.cmake under $LP_VIEW"; exit 1; }

CLEAN_PATH=$(echo "$PATH" | tr ':' '\n' | grep -v spack-env | tr '\n' ':')

# ---------------------------------------------------------------------------
# BUILD  (skip with PROG_SKIP_BUILD=1 when iterating on the sweep itself)
# ---------------------------------------------------------------------------
if [ "${PROG_SKIP_BUILD:-0}" != "1" ]; then
    echo ""
    echo "=== Building ==="
    rm -rf build
    env PATH="$CLEAN_PATH" LD_LIBRARY_PATH="$GCC_LIB" \
    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE="${VOL_BUILD_TYPE:-Release}" \
        -DCMAKE_C_COMPILER="$SYSTEM_MPICC" \
        -DCMAKE_CXX_COMPILER="$SYSTEM_MPICXX" \
        -DUSE_CUDA=ON \
        -DCMAKE_PREFIX_PATH="$LP_VIEW" \
        -DLibPressio_DIR="$LP_CMAKE_DIR" \
        -DSPERR_PREFIX="$SPERR_PREFIX" \
        -Wno-dev || exit 1
    env PATH="$CLEAN_PATH" LD_LIBRARY_PATH="$GCC_LIB" \
        cmake --build build -j16 || exit 1
fi

BIN=""
for d in build/bin build/tests build; do
    [ -x "$d/bench_vol_timing" ] && { BIN="$(pwd)/$d"; break; }
done
[ -n "$BIN" ] || { echo "ERROR: bench_vol_timing not found"; exit 1; }

PLUGIN_DIR=""
for d in build/lib build; do
    [ -e "$d/libhdf5_vol_passthrough.so" ] && { PLUGIN_DIR="$(pwd)/$d"; break; }
done
[ -n "$PLUGIN_DIR" ] || { echo "ERROR: libhdf5_vol_passthrough.so not found"; exit 1; }

echo "Binary dir: $BIN"
echo "Plugin dir: $PLUGIN_DIR"

source "$PBS_O_WORKDIR/zfp64_preamble.sh"
export HDF5_USE_FILE_LOCKING=FALSE

H5DIR="${VOL_H5DIR:-/lcrc/project/SDR/$USER/vol_bench_h5}"
mkdir -p "$H5DIR" || { echo "cannot create $H5DIR"; exit 1; }

# ---------------------------------------------------------------------------
# STEP 0 -- gate: is the SPERR stream prefix-truncatable at all?
#
# If sperr_trunc_3d cannot work from a prefix for this configuration, every
# reduced-fidelity read below fails and the sweep produces nothing but error
# rows. Ten seconds, and it decides whether the rest of the job is worth
# running. Build the probe alongside the sweep.
# ---------------------------------------------------------------------------
PROBE="$BIN/sperr_trunc_probe"
if [ ! -x "$PROBE" ] && [ -f "$PBS_O_WORKDIR/sperr_trunc_probe.cpp" ]; then
    echo ""
    echo "=== [0] building the truncation probe ==="
    g++ -O2 -std=c++17 "$PBS_O_WORKDIR/sperr_trunc_probe.cpp" -o "$RESULTS/sperr_trunc_probe" \
        -I"$SPERR_PREFIX/include" -L"$SPERR_PREFIX/lib" -L"$SPERR_PREFIX/lib64" \
        -lSPERR -Wl,-rpath,"$SPERR_PREFIX/lib" -Wl,-rpath,"$SPERR_PREFIX/lib64" \
        && PROBE="$RESULTS/sperr_trunc_probe"
fi
if [ -x "$PROBE" ]; then
    echo ""
    echo "=== [0] SPERR prefix-truncation gate ==="
    "$PROBE" --dims ${PROG_DIMS:-384 384 256} | tee "$RESULTS/probe_default_chunking.txt"
    PROBE_RC=${PIPESTATUS[0]}
    echo ""
    echo "--- forced single chunk ---"
    "$PROBE" --dims ${PROG_DIMS:-384 384 256} --chunk ${PROG_DIMS:-384 384 256} \
        | tee "$RESULTS/probe_single_chunk.txt"
    if [ "$PROBE_RC" -ne 0 ]; then
        echo ""
        echo "*** PROBE FAILED under default chunking. Reduced-fidelity reads on"
        echo "*** the native path will error out. Either force single-chunk SPERR"
        echo "*** via opts_json, or land Route A (per-chunk ranges) first."
        [ "${PROG_FORCE:-0}" = "1" ] || exit 1
    fi
else
    echo "=== [0] SKIPPED: probe not built (put sperr_trunc_probe.cpp beside this script) ==="
fi

# ---------------------------------------------------------------------------
# THE SWEEP
#
# PROG_COMP must name an entry in BENCH_COMPRESSORS whose pressio_id is sperr.
# PROG_CHUNKING: "" = native single stream (uniform pct, works today),
#                "progressive" = Route A per-chunk (needs the patches).
# ---------------------------------------------------------------------------
PROG_DSET="${PROG_DSET:-miranda}"
PROG_SEL="${PROG_SEL:-miranda}"
PROG_COMP="${PROG_COMP:-sperr_pwe1e3}"
PROG_CHUNKING="${PROG_CHUNKING:-}"
PROG_CHUNK_N="${PROG_CHUNK_N:-}"
PCTS="${PROG_PCTS:-5 10 25 50 75 100}"
REPS="${PROG_REPS:-3}"

XCSV_HEADER_24="dataset,compressor,codec_kind,chunk_n,rep,logical_bytes,stored_bytes,ratio,create_ms,write_ms,flush_ms,sync_ms,close_ms,csync_ms,evict_ms,open_ms,read_ms,rmse,abs_thresh,maxae,bound_ok,write_wall_ms,d2h_ms,read_wall_ms"

echo ""
echo "=== progressive sweep ==="
echo "    dataset:    $PROG_DSET ($PROG_SEL)"
echo "    compressor: $PROG_COMP"
echo "    chunking:   ${PROG_CHUNKING:-native}"
echo "    pcts:       $PCTS"
echo "    reps:       $REPS"

for pct in $PCTS; do
  for rep in $(seq 0 $((REPS - 1))); do
    label="p${pct}_r${rep}"
    h5="$H5DIR/prog_${label}.h5"
    rcsv="$RESULTS/results_vol_${label}.csv"
    xcsv="$RESULTS/results_ext_${label}.csv"
    pcsv="$RESULTS/connector_phases_${label}.csv"
    plan="$RESULTS/plan_${label}.csv"

    rm -f "$H5DIR"/prog_"${label}"*.h5 "$plan"

    echo ""
    echo "--- pct=${pct} rep=${rep} ---"
    env HDF5_VOL_CONNECTOR="pass_through_ext under_vol=0;under_info={}" \
        HDF5_PLUGIN_PATH="$PLUGIN_DIR" \
        VOL_PROGRESSIVE_PCT="$pct" \
        VOL_PROGRESSIVE_LOG=1 \
        VOL_READ_PLAN_LOG=1 \
        VOL_PLAN_CSV="$plan" \
        BENCH_CSV="$pcsv" \
        BENCH_FSYNC=1 \
        BENCH_SYNC_DIR=1 \
        BENCH_DROP_CACHE=1 \
        BENCH_VERIFY=0 \
        BENCH_KEEP_H5=0 \
        VOL_TIMING_STRICT=0 \
        ${PROG_CHUNKING:+VOL_COMP_CHUNKING="$PROG_CHUNKING"} \
        ${PROG_CHUNK_N:+VOL_COMP_CHUNK_N="$PROG_CHUNK_N"} \
        BENCH_ONLY="$PROG_SEL" \
        BENCH_COMP="$PROG_COMP" \
        timeout --signal=KILL "${VOL_TIMEOUT:-1800}" \
            "$BIN"/bench_vol_timing "$h5" "$rcsv" "$xcsv"
    rc=$?
    rm -f "$H5DIR"/prog_"${label}"*.h5

    if [ $rc -ne 0 ]; then
        [ -f "$xcsv" ] || printf '%s\n' "$XCSV_HEADER_24" > "$xcsv"
        printf '%s,%s,NA,0,-1,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1\n' \
            "$PROG_SEL" "$PROG_COMP" >> "$xcsv"
        echo "  [FAIL] pct=$pct rep=$rep rc=$rc -- recorded as rep=-1"
    fi
  done
done

# ---------------------------------------------------------------------------
# JOIN -- one tidy CSV for the plot script.
#
# bytes_read comes from the connector's own plan log, so it is the number of
# bytes the VOL actually asked the filesystem for, not an estimate.
# ---------------------------------------------------------------------------
JOINED="$RESULTS/progressive.csv"
echo "pct,rep,logical_bytes,stored_bytes,bytes_read,cont_bytes,read_wall_ms,read_cpu_ms,open_ms,evict_ms,rmse,maxae,ratio" > "$JOINED"

for pct in $PCTS; do
  for rep in $(seq 0 $((REPS - 1))); do
    label="p${pct}_r${rep}"
    xcsv="$RESULTS/results_ext_${label}.csv"
    plan="$RESULTS/plan_${label}.csv"
    [ -f "$xcsv" ] || continue

    # Largest bytes_needed across this run's plans: the dataset read, not the
    # small header probes.
    if [ -f "$plan" ]; then
        read BN CB <<<"$(awk -F, '{ if ($4+0 > m) { m=$4+0; c=$5+0 } } END { print m, c }' "$plan")"
    else
        BN=-1; CB=-1
    fi

    awk -F, -v P="$pct" -v R="$rep" -v BN="${BN:--1}" -v CB="${CB:--1}" \
        'FNR>1 && $5!="-1" {
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
                   P, R, $6, $7, BN, CB, $24, $17, $16, $15, $18, $20, $8
         }' "$xcsv" >> "$JOINED"
  done
done

echo ""
echo "=== joined results ==="
column -t -s, "$JOINED"

echo ""
echo "=== runs that FAILED (rep=-1) ==="
awk -F, 'FNR>1 && $5=="-1" {print FILENAME": pct row"}' "$RESULTS"/results_ext_*.csv 2>/dev/null \
    || echo "(none)"

echo ""
echo "Wrote $JOINED"
echo "Plot with:  python3 plot_progressive.py $JOINED -o progressive.png"
echo "Job finished on $(date)"