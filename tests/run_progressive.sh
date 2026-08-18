#!/bin/bash
#PBS -N vol_progressive_sperr
#PBS -l select=1:ncpus=16:ngpus=1:mpiprocs=1:mem=256gb
#PBS -l walltime=00:30:00
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
# BUILD
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

PROG_DSET="${PROG_DSET:-miranda}"
PROG_SEL="${PROG_SEL:-miranda}"
PROG_COMPS="${PROG_COMPS:- sperr_pwe1e6}"
PROG_CHUNKING="${PROG_CHUNKING:-}"
PROG_CHUNK_N="${PROG_CHUNK_N:-}"
PCTS="${PROG_PCTS:-5 10 25 50 75 100}"
REPS="${PROG_REPS:-3}"
PROG_DIMS="${PROG_DIMS:-384 384 256}"

XCSV_HEADER_24="dataset,compressor,codec_kind,chunk_n,rep,logical_bytes,stored_bytes,ratio,create_ms,write_ms,flush_ms,sync_ms,close_ms,csync_ms,evict_ms,open_ms,read_ms,rmse,abs_thresh,maxae,bound_ok,write_wall_ms,d2h_ms,read_wall_ms"

# ===========================================================================
# PREFLIGHT
#
# Five things that can each produce a full, plausible-looking sweep that means
# nothing. Every one is cheap to check and expensive to discover afterwards.
# ===========================================================================
PF_FAIL=0
pf_ok()   { echo "  [ OK ] $*"; }
pf_bad()  { echo "  [FAIL] $*"; PF_FAIL=$((PF_FAIL + 1)); }
pf_warn() { echo "  [WARN] $*"; }

echo ""
echo "###########################################################"
echo "# PREFLIGHT"
echo "###########################################################"

# --- P1: is the sperr plugin actually loaded? ------------------------------
echo ""
echo "=== [P1] libpressio-sperr preload ==="
echo "  LD_PRELOAD:"
echo "${LD_PRELOAD:-}" | tr ':' '\n' | sed 's/^/    /'
if echo "${LD_PRELOAD:-}" | grep -q 'liblibpressio_sperr'; then
    pf_ok "sperr plugin is in LD_PRELOAD"
else
    pf_bad "sperr plugin NOT in LD_PRELOAD -- add it to the END of zfp64_preamble.sh."
    echo "         Without it the connector dies mid-sweep with"
    echo "         'invalid compressor id sperr'."
fi
if echo "${LD_PRELOAD:-}" | grep -q 'zfp'; then
    pf_ok "zfp bsws=64 preload still present (not clobbered)"
else
    pf_warn "no zfp entry in LD_PRELOAD -- fine for sperr, but check you appended"
    pf_warn "rather than overwrote if you also want the zfp arms."
fi

# --- P2: does libpressio hand us a sperr compressor? -----------------------
echo ""
echo "=== [P2] sperr registered with libpressio ==="
DUMP="$BIN/dump_opts"
[ -x "$DUMP" ] || DUMP="$WORK/dump_opts"
if [ -x "$DUMP" ]; then
    if "$DUMP" sperr > "$RESULTS/sperr_options.txt" 2>&1; then
        pf_ok "compressor 'sperr' resolves"
        CHUNKDEF=$(grep -o 'has_data=\[[^]]*\]' "$RESULTS/sperr_options.txt" | head -1)
        echo "         sperr:chunks default -> ${CHUNKDEF:-<not found>}"
    else
        pf_bad "dump_opts could not get a sperr compressor -- see $RESULTS/sperr_options.txt"
    fi
else
    pf_warn "dump_opts not built; skipping (build it, it is 5 seconds)"
fi

# --- P3: the truncation invariant ------------------------------------------
# EXPECTATION IS INVERTED from the old script: default chunking SHOULD fail.
echo ""
echo "=== [P3] SPERR prefix-truncation invariant ==="
PROBE="$BIN/sperr_trunc_probe"
if [ ! -x "$PROBE" ] && [ -f "$PBS_O_WORKDIR/sperr_trunc_probe.cpp" ]; then
    g++ -O2 -std=c++17 "$PBS_O_WORKDIR/sperr_trunc_probe.cpp" \
        -o "$RESULTS/sperr_trunc_probe" \
        -I"$SPERR_PREFIX/include" -L"$SPERR_PREFIX/lib" -L"$SPERR_PREFIX/lib64" \
        -lSPERR -Wl,-rpath,"$SPERR_PREFIX/lib" -Wl,-rpath,"$SPERR_PREFIX/lib64" \
        && PROBE="$RESULTS/sperr_trunc_probe"
fi
if [ -x "$PROBE" ]; then
    echo "  --- default 256^3 chunking (EXPECTED TO FAIL) ---"
    "$PROBE" --dims $PROG_DIMS > "$RESULTS/probe_default.txt" 2>&1
    RC_DEFAULT=$?
    tail -4 "$RESULTS/probe_default.txt" | sed 's/^/    /'

    echo "  --- pinned to the volume (MUST PASS) ---"
    "$PROBE" --dims $PROG_DIMS --chunk $PROG_DIMS > "$RESULTS/probe_pinned.txt" 2>&1
    RC_PINNED=$?
    tail -4 "$RESULTS/probe_pinned.txt" | sed 's/^/    /'

    if [ $RC_PINNED -ne 0 ]; then
        pf_bad "pinned probe FAILED -- prefix truncation is broken even with one"
        echo "         internal chunk. Nothing downstream can work. See"
        echo "         $RESULTS/probe_pinned.txt"
    else
        pf_ok "pinned: single chunk, prefix truncation valid"
    fi
    if [ $RC_DEFAULT -eq 0 ]; then
        pf_warn "default chunking ALSO passed. For $PROG_DIMS that is unexpected --"
        pf_warn "it means the pinning is not being exercised, so a regression in"
        pf_warn "vol_sperr_set_chunks would go unnoticed. Not fatal."
    else
        pf_ok "default: multi-chunk and unusable, as expected -- pinning is load-bearing"
    fi
else
    pf_warn "probe not built; skipping the invariant check"
fi

# --- P4 + P5: one real run, end to end -------------------------------------
echo ""
echo "=== [P4] smoke run (one pct, one rep, full VOL path) ==="
SMOKE_COMP=$(echo $PROG_COMPS | awk '{print $1}')
SMOKE_LOG="$RESULTS/smoke_${SMOKE_COMP}.log"
SMOKE_PLAN="$RESULTS/smoke_plan.csv"
SMOKE_X="$RESULTS/smoke_ext.csv"
rm -f "$SMOKE_PLAN" "$H5DIR"/smoke_*.h5

echo "  compressor: $SMOKE_COMP   pct: 25"
env HDF5_VOL_CONNECTOR="pass_through_ext under_vol=0;under_info={}" \
    HDF5_PLUGIN_PATH="$PLUGIN_DIR" \
    VOL_PROGRESSIVE_PCT=25 \
    VOL_PROGRESSIVE_LOG=1 \
    VOL_READ_PLAN_LOG=1 \
    VOL_COMP_CHUNK_LOG=1 \
    VOL_PLAN_CSV="$SMOKE_PLAN" \
    BENCH_FSYNC=1 BENCH_SYNC_DIR=1 BENCH_DROP_CACHE=1 \
    BENCH_VERIFY=0 BENCH_KEEP_H5=0 VOL_TIMING_STRICT=0 \
    BENCH_ONLY="$PROG_SEL" BENCH_COMP="$SMOKE_COMP" \
    timeout --signal=KILL "${VOL_TIMEOUT:-1800}" \
        "$BIN"/bench_vol_timing "$H5DIR/smoke.h5" \
            "$RESULTS/smoke_vol.csv" "$SMOKE_X" > "$SMOKE_LOG" 2>&1
SMOKE_RC=$?
rm -f "$H5DIR"/smoke_*.h5

echo "  --- connector chatter ---"
grep -E '\[sperr\]|\[prog\]|\[read-plan\]|\[VOL\]|invalid compressor' "$SMOKE_LOG" \
    | head -20 | sed 's/^/    /'

if [ $SMOKE_RC -ne 0 ]; then
    pf_bad "smoke run exited rc=$SMOKE_RC -- see $SMOKE_LOG"
    grep -iE 'error|fail|abort' "$SMOKE_LOG" | head -10 | sed 's/^/         /'
else
    pf_ok "smoke run completed"
fi

# chunk pinning reached the codec?
if grep -q '\[sperr\] chunks pinned' "$SMOKE_LOG"; then
    NPIN=$(grep -c '\[sperr\] chunks pinned' "$SMOKE_LOG")
    pf_ok "vol_sperr_set_chunks fired ($NPIN call(s))"
else
    pf_bad "no '[sperr] chunks pinned' line -- vol_sperr_set_chunks never ran."
    echo "         SPERR used its 256^3 default and the stream is multi-chunk."
fi

# progressive truncation actually happened?
if grep -q '\[prog\]' "$SMOKE_LOG"; then
    pf_ok "progressive truncation ran ($(grep -c '\[prog\]' "$SMOKE_LOG") chunk(s))"
else
    pf_bad "no '[prog]' line -- the reduced-fidelity path never engaged."
    echo "         Check vol_codec_is_progressive against the registered id."
fi

echo ""
echo "=== [P5] did the measurement come out measurable? ==="

# 24 fields => the read_wall_ms patch landed
if [ -f "$SMOKE_X" ]; then
    NF=$(head -1 "$SMOKE_X" | tr ',' '\n' | wc -l)
    if [ "$NF" = "24" ]; then
        pf_ok "xcsv has 24 fields (read_wall_ms present)"
    else
        pf_bad "xcsv has $NF fields, expected 24 -- the read_wall_ms patch is missing."
        echo "         The join below reads \$24 and would produce empty throughput."
    fi

    # A header with no rows means the run measured NOTHING. Checking the header
    # alone is a false green light -- it is what let a zero-measurement run look
    # healthy the first time.
    NROWS=$(awk 'END{print NR-1}' "$SMOKE_X")
    if [ "${NROWS:-0}" -ge 1 ] 2>/dev/null; then
        pf_ok "xcsv has $NROWS data row(s)"
    else
        pf_bad "xcsv has a header but ZERO data rows -- nothing was measured."
        echo "         Either BENCH_COMP='$SMOKE_COMP' matches no entry in"
        echo "         BENCH_COMPRESSORS (check: grep -n sperr tests/bench_config.h),"
        echo "         or run_rep failed before writing a row (check $SMOKE_LOG)."
    fi
    # wall vs cpu: a cold read should spend real time blocked on GPFS
    read RCPU RWALL <<<"$(awk -F, 'NR==2 {print $17, $24}' "$SMOKE_X")"
    echo "         read_ms(cpu)=${RCPU:-?}  read_wall_ms=${RWALL:-?}"
    if [ -n "${RWALL:-}" ] && [ -n "${RCPU:-}" ]; then
        awk -v w="$RWALL" -v c="$RCPU" 'BEGIN{ exit !(w+0 > c+0 * 1.05) }' \
            && pf_ok "wall > cpu -- the read really is blocking on I/O" \
            || pf_warn "wall ~= cpu. Either the file was still cached (check
         BENCH_DROP_CACHE / fadvise on GPFS) or the read is decode-bound.
         Compare against pct=100 before trusting the throughput curve."
    fi
else
    pf_bad "no xcsv produced"
fi

# partial read really was partial
if [ -s "$SMOKE_PLAN" ]; then
    echo "  --- plan rows (kind,pct,nranges,needed,cont,partial) ---"
    sed 's/^/    /' "$SMOKE_PLAN"
    read BN CB <<<"$(awk -F, '{ if ($4+0 > m) { m=$4+0; c=$5+0 } } END { print m, c }' "$SMOKE_PLAN")"
    if [ -n "${BN:-}" ] && [ "${CB:-0}" -gt 0 ] 2>/dev/null; then
        PCTREAD=$(awk -v b="$BN" -v c="$CB" 'BEGIN{printf "%.1f", 100.0*b/c}')
        echo "         bytes_needed/cont_bytes = ${PCTREAD}% (requested 25%)"
        awk -v p="$PCTREAD" 'BEGIN{ exit !(p+0 < 60.0) }' \
            && pf_ok "the read really was partial" \
            || pf_bad "read ~${PCTREAD}% of the container at pct=25 -- truncation
         did not take effect. Check vol_codec_is_progressive and that the
         container magic is CHUNKED (kind=3) or NATIVE (kind=1)."
    fi
else
    pf_bad "no plan CSV -- VOL_PLAN_CSV patch missing from vol_container_plan."
    echo "         bytes_read would be -1 for every point and the mechanism"
    echo "         panel would be empty."
fi

echo ""
echo "###########################################################"
if [ $PF_FAIL -gt 0 ]; then
    echo "# PREFLIGHT: $PF_FAIL FAILURE(S)"
    echo "###########################################################"
    if [ "${PROG_FORCE:-0}" != "1" ]; then
        echo "Not starting the sweep. Fix the above, or re-submit with"
        echo "  qsub -v PROG_FORCE=1 run_progressive.sh"
        exit 1
    fi
    echo "PROG_FORCE=1 -- continuing anyway."
else
    echo "# PREFLIGHT: all checks passed"
    echo "###########################################################"
fi

if [ "${PROG_PREFLIGHT_ONLY:-0}" = "1" ]; then
    echo ""
    echo "PROG_PREFLIGHT_ONLY=1 -- stopping before the sweep."
    echo "Artifacts in $RESULTS"
    exit 0
fi

# ===========================================================================
# THE SWEEP
# ===========================================================================
echo ""
echo "=== progressive sweep ==="
echo "    dataset:     $PROG_DSET ($PROG_SEL)"
echo "    compressors: $PROG_COMPS"
echo "    pcts:        $PCTS"
echo "    reps:        $REPS"
echo "    runs:        $(echo $PROG_COMPS | wc -w) x $(echo $PCTS | wc -w) x $REPS"
echo ""
echo "    NOTE: each run rewrites the dataset. Pinning SPERR to one internal"
echo "    chunk removes its cross-chunk threading, so the native arm's writes"
echo "    are slow by design. If walltime looks tight, drop PROG_REPS to 1 for"
echo "    a first pass -- the read numbers are what matter."

for comp in $PROG_COMPS; do
  echo ""
  echo "#### compressor: $comp ####"
  for pct in $PCTS; do
    for rep in $(seq 0 $((REPS - 1))); do
      label="${comp}_p${pct}_r${rep}"
      h5="$H5DIR/prog_${label}.h5"
      rcsv="$RESULTS/results_vol_${label}.csv"
      xcsv="$RESULTS/results_ext_${label}.csv"
      pcsv="$RESULTS/connector_phases_${label}.csv"
      plan="$RESULTS/plan_${label}.csv"

      rm -f "$H5DIR"/prog_"${label}"*.h5 "$plan"

      echo ""
      echo "--- $comp pct=${pct} rep=${rep} ---"
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
          BENCH_COMP="$comp" \
          timeout --signal=KILL "${VOL_TIMEOUT:-1800}" \
              "$BIN"/bench_vol_timing "$h5" "$rcsv" "$xcsv"
      rc=$?
      rm -f "$H5DIR"/prog_"${label}"*.h5

      if [ $rc -ne 0 ]; then
          [ -f "$xcsv" ] || printf '%s\n' "$XCSV_HEADER_24" > "$xcsv"
          printf '%s,%s,NA,0,-1,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1\n' \
              "$PROG_SEL" "$comp" >> "$xcsv"
          echo "  [FAIL] $comp pct=$pct rep=$rep rc=$rc -- recorded as rep=-1"
      fi
    done
  done
done

# ===========================================================================
# JOIN -- one tidy CSV per compressor, matching plot_progressive.py's schema.
#
# bytes_read comes from the connector's own plan log: the bytes the VOL asked
# the filesystem for, not an estimate.
# ===========================================================================
for comp in $PROG_COMPS; do
  JOINED="$RESULTS/progressive_${comp}.csv"
  echo "pct,rep,logical_bytes,stored_bytes,bytes_read,cont_bytes,read_wall_ms,read_cpu_ms,open_ms,evict_ms,rmse,maxae,ratio" > "$JOINED"

  for pct in $PCTS; do
    for rep in $(seq 0 $((REPS - 1))); do
      label="${comp}_p${pct}_r${rep}"
      xcsv="$RESULTS/results_ext_${label}.csv"
      plan="$RESULTS/plan_${label}.csv"
      [ -f "$xcsv" ] || continue

      # Largest bytes_needed across this run's plans: the dataset read, not the
      # small header probes.
      if [ -f "$plan" ]; then
          read BN CB <<<"$(awk -F, '{ if ($4+0 > m) { m=$4+0; c=$5+0 } } END { print m+0, c+0 }' "$plan")"
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
  echo "=== joined: $comp ==="
  column -t -s, "$JOINED" 2>/dev/null || cat "$JOINED"
done

echo ""
echo "=== runs that FAILED (rep=-1) ==="
awk -F, 'FNR>1 && $5=="-1" {print FILENAME}' "$RESULTS"/results_ext_*.csv 2>/dev/null \
    | sort -u || echo "(none)"

echo ""
echo "=== next ==="
for comp in $PROG_COMPS; do
  echo "  python3 plot_progressive.py $RESULTS/progressive_${comp}.csv \\"
  echo "      -o $RESULTS/progressive_${comp}.png --cpu-clock \\"
  echo "      --subtitle '$PROG_DSET, SPERR $comp, GPFS cold cache, $REPS reps'"
done
echo ""
echo "Job finished on $(date)"
