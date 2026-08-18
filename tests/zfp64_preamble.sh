#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# zfp64_preamble.sh -- force libpressio onto a bsws=64 zfp so the CUDA
# execution policy actually works.
#
# WHY THIS EXISTS
#   The libpressio_cuda spack env resolves zfp to hash vxzs7uh, which is
#   zfp@0.5.5 bsws=8. h5z-zfp declares depends_on("zfp bsws=8") and, under
#   `unify: true`, that requirement wins for the whole env.  But zfp's CUDA
#   backend requires ZFP_BIT_STREAM_WORD_SIZE == 64, so with bsws=8 the cuda
#   execution policy is unavailable -- zfp_stream_set_execution() fails and
#   libpressio may surface that as a generic error, or silently fall back.
#
#   zfp@0.5.5 bsws=64 (hash e6nrl7c) is already in the store.  Same version,
#   same headers, same soname (libzfp.so.0) -- the ONLY difference is the
#   word-size macro.  So we redirect the loader to it at runtime instead of
#   reconcretizing the env.
#
# THIS IS A WORKAROUND, NOT A FIX
#   The permanent fix is to drop h5z-zfp from the env (or move it to its own
#   env) and set `zfp: require: +cuda~openmp bsws=64 cuda_arch=80`.  Until
#   then, ALWAYS check the pair invariant described at the bottom of this file
#   before believing any zfp GPU timing.
#
# USAGE (from bench_vol_timing.pbs, after activating the spack env):
#   source /path/to/zfp64_preamble.sh
#
# KNOBS
#   BENCH_ZFP64=0        skip the swap entirely (run stock env zfp, CPU only)
#   BENCH_ZFP_SMOKE=1    run a standalone GPU zfp check before the sweep
#   ZFP64_PREFIX=...     override the bsws=64 prefix
#   LP_VIEW=... GCC_LIB=...   override paths
# ---------------------------------------------------------------------------

: "${LP_VIEW:=$HOME/libpressio_cuda/.spack-env/view}"
: "${GCC_LIB:=/usr/lib/gcc/x86_64-linux-gnu/11}"
: "${BENCH_ZFP64:=1}"
: "${BENCH_ZFP_SMOKE:=0}"

zfp64_log() { printf '[zfp64] %s\n' "$*" >&2; }

# Find something that links libzfp, to probe which libzfp the loader picks.
#
# NOTE: libpressio installs as liblibpressio.so -- the CMake project is named
# "libpressio", so the SONAME gets a second "lib" prefix.  It may also land in
# lib64 rather than lib.  Glob rather than guess.
#
# ZFP64_PROBE can name any ELF object that pulls in libzfp; the built VOL
# connector is preferred when present because it is the exact object whose
# binding we care about.
# Which libzfp does the loader actually pick for $1?
#
# ldd prints two different shapes and both must be handled:
#   DT_NEEDED resolution :  libzfp.so.0 => /path/libzfp.so.0 (0x...)   -> $3
#   LD_PRELOAD injection :  /path/libzfp.so.0 (0x...)                  -> $1
# Reading $3 unconditionally returns empty for the preload case, which reads
# as "unresolved" when in fact the preload is exactly what we wanted.
# Prefer whichever line yields an absolute path.
zfp64_resolved_zfp() {
    ldd "$1" 2>/dev/null | awk '
        /libzfp/ {
            p = ($2 == "=>") ? $3 : $1
            if (p ~ /^\//) { print p; exit }
        }'
}

zfp64_probe_target() {
    local c
    if [ -n "${ZFP64_PROBE:-}" ] && [ -e "$ZFP64_PROBE" ]; then
        printf '%s\n' "$ZFP64_PROBE"; return 0
    fi
    for c in "$PWD/build/lib/libhdf5_vol_passthrough.so" \
             "$PWD/build/libhdf5_vol_passthrough.so"; do
        [ -e "$c" ] && { printf '%s\n' "$c"; return 0; }
    done
    for c in "$LP_VIEW"/lib64/lib*pressio*.so* "$LP_VIEW"/lib/lib*pressio*.so*; do
        [ -e "$c" ] && { printf '%s\n' "$c"; return 0; }
    done
    return 1
}

# --- baseline library path (unchanged from your existing scripts) -----------
export LD_LIBRARY_PATH="$GCC_LIB:$LP_VIEW/lib64:$LP_VIEW/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- durability / cache controls for the io_ms phases ----------------------
: "${BENCH_FSYNC:=1}"
: "${BENCH_DROP_CACHE:=1}"
: "${BENCH_SYNC_DIR:=1}"
export BENCH_FSYNC BENCH_DROP_CACHE BENCH_SYNC_DIR

# ---------------------------------------------------------------------------
# Optional pre-flight: does GPU zfp work on this node at all?
# Runs the zfp@1.0.2 CLI in a CLEAN environment -- that binary needs
# libzfp.so.1, and preloading our 0.5.5 libzfp.so.0 into it would interpose
# 0.5.5 symbols under a 1.0.2 binary.  Hence `env -u`.
# ---------------------------------------------------------------------------
zfp64_smoke() {
    local z102 src=/tmp/zfp64_smoke_$$.f64 out=/tmp/zfp64_smoke_$$.zfp rc

    z102=$(spack location -i zfp@1.0.2 2>/dev/null) || {
        zfp64_log "smoke: zfp@1.0.2 not locatable, skipping"; return 0; }
    [ -x "$z102/bin/zfp" ] || {
        zfp64_log "smoke: no CLI at $z102/bin/zfp, skipping"; return 0; }

    head -c 2097152 \
        /lcrc/project/ECP-EZ/public/compression/Miranda/SDRBENCH-Miranda-256x384x384/density.d64 \
        > "$src" 2>/dev/null || { zfp64_log "smoke: could not stage input"; return 0; }

    env -u LD_PRELOAD -u LD_LIBRARY_PATH \
        "$z102/bin/zfp" -d -3 64 64 64 -r 8 -x cuda -i "$src" -z "$out"
    rc=$?
    rm -f "$src" "$out"

    if [ $rc -ne 0 ]; then
        zfp64_log "SMOKE FAILED (rc=$rc): GPU zfp does not run on this node."
        zfp64_log "  If this is a login node, that is expected -- run inside a job."
        return 1
    fi
    zfp64_log "smoke: GPU zfp OK on $(hostname)"
    return 0
}

# ---------------------------------------------------------------------------
# The swap
# ---------------------------------------------------------------------------
zfp64_apply() {
    local prefix libdir lib resolved lp

    if ! lp=$(zfp64_probe_target); then
        zfp64_log "ERROR: no probe target found (no built VOL .so, and no"
        zfp64_log "       lib*pressio*.so* under $LP_VIEW/{lib64,lib})."
        zfp64_log "       Set ZFP64_PROBE to any object that links libzfp."
        return 1
    fi
    zfp64_log "probe target: $lp"

    if ! ldd "$lp" 2>/dev/null | grep -q libzfp; then
        zfp64_log "ERROR: $lp does not link libzfp -- wrong probe target."
        return 1
    fi

    prefix="${ZFP64_PREFIX:-}"
    if [ -z "$prefix" ] && command -v spack >/dev/null 2>&1; then
        prefix=$(spack location -i zfp@0.5.5 bsws=64 2>/dev/null)
    fi
    : "${prefix:=/gpfs/fs1/home/kdougherty/spack/opt/spack/linux-zen2/zfp-0.5.5-e6nrl7coy6muddatslhbbmjm3uphjrg6}"

    if [ ! -d "$prefix" ]; then
        zfp64_log "ERROR: bsws=64 prefix not found: $prefix"
        return 1
    fi

    libdir=""
    for d in "$prefix/lib64" "$prefix/lib"; do
        [ -e "$d/libzfp.so.0" ] && { libdir="$d"; break; }
    done
    if [ -z "$libdir" ]; then
        zfp64_log "ERROR: no libzfp.so.0 under $prefix"
        return 1
    fi
    lib="$libdir/libzfp.so.0"

    # Attempt 1: search-path ordering. Works only if libpressio carries
    # DT_RUNPATH; DT_RPATH takes precedence over LD_LIBRARY_PATH and would
    # silently keep the bsws=8 build.
    export LD_LIBRARY_PATH="$GCC_LIB:$libdir:$LP_VIEW/lib64:$LP_VIEW/lib"

    resolved=$(zfp64_resolved_zfp "$lp")
    case "$resolved" in
        "$prefix"/*) zfp64_log "resolved via LD_LIBRARY_PATH -> $resolved" ;;
        *)
            zfp64_log "LD_LIBRARY_PATH insufficient (got: ${resolved:-none}); using LD_PRELOAD"
            export LD_PRELOAD="$lib${LD_PRELOAD:+:$LD_PRELOAD}"
            resolved=$(zfp64_resolved_zfp "$lp")
            ;;
    esac

    case "$resolved" in
        "$prefix"/*) : ;;
        *)
            zfp64_log "ERROR: libpressio still resolves libzfp to '${resolved:-none}'."
            zfp64_log "       Refusing to continue -- GPU zfp rows would be invalid."
            return 1
            ;;
    esac

    # --- provenance for the job log / paper methods section ----------------
    zfp64_log "-------- zfp binding --------"
    zfp64_log "  probe target     : $lp"
    zfp64_log "  env zfp (stock)  : $(readlink -f "$LP_VIEW/lib/libzfp.so" 2>/dev/null)"
    zfp64_log "  active zfp       : $(readlink -f "$resolved")"
    zfp64_log "  LD_PRELOAD       : ${LD_PRELOAD:-<unset>}"
    zfp64_log "  DT tags on libpressio: $(readelf -d "$lp" 2>/dev/null \
                                          | grep -oE 'RPATH|RUNPATH' | sort -u | tr '\n' ' ')"
    zfp64_log "-----------------------------"
    return 0
}

# ---------------------------------------------------------------------------
if [ "$BENCH_ZFP64" != "0" ]; then
    if [ "$BENCH_ZFP_SMOKE" = "1" ]; then
        zfp64_smoke || zfp64_log "continuing despite smoke failure (set -e to abort)"
    fi
    if ! zfp64_apply; then
        zfp64_log "FALLING BACK: unsetting the swap; run zfp CPU rows only."
        zfp64_log "  suggested: BENCH_COMP=\"noop,bzip2,sz3_1e3,sz3_1e6,cuszp_1e3,cuszp_1e6,zfp_1e3,zfp_1e6\""
        unset LD_PRELOAD
        export LD_LIBRARY_PATH="$GCC_LIB:$LP_VIEW/lib64:$LP_VIEW/lib"
        export BENCH_ZFP64=0
    fi
else
    zfp64_log "BENCH_ZFP64=0 -- using the env's stock bsws=8 zfp (GPU zfp will not work)"
fi

# ---------------------------------------------------------------------------
# AFTER THE RUN -- verify the swap held.  zfp guarantees the compressed stream
# is independent of execution policy, so within a rate pair the CPU and GPU
# rows must agree EXACTLY on stored_bytes.  Disagreement means the ABI swap
# went wrong; do not report those timings.
#
#   awk -F, 'NR==1{next} $2 ~ /^zfp_(cpu|gpu)_r/ {print $2, $7, $18}' \
#       results_vol.csv.ext.csv | sort
#
# Also note: with LD_PRELOAD active, ANY tool in this job that uses h5z-zfp
# will now be running against bsws=64 and will produce streams incompatible
# with your stock env.  Keep h5z-zfp work in a separate job.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# libpressio-sperr is a SEPARATE spack package, not a +sperr variant of
# libpressio, so it builds its own shared object whose registration
# constructor only runs if something loads it. Without this, anything asking
# libpressio for "sperr" gets 'invalid compressor id sperr' -- dump_opts on
# the login node, and the VOL connector mid-sweep.
#
# Appended, never assigned: the zfp bsws=64 swap above already owns LD_PRELOAD.
# ---------------------------------------------------------------------------
: "${LP_VIEW:=/home/kdougherty/libpressio_cuda/.spack-env/view}"
SPERR_PLUGIN="${LP_VIEW}/lib/liblibpressio_sperr.so"
if [ -e "$SPERR_PLUGIN" ]; then
    export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}$SPERR_PLUGIN"
    echo "[preamble] sperr plugin preloaded: $SPERR_PLUGIN"
else
    echo "[preamble] WARNING: $SPERR_PLUGIN missing -- sperr runs will fail" >&2
fi
