# Interactive-shell environment for the VOL work.
#
#   source env.sh
#
# The build variables live inside run_vol_test.sh / run_progressive.sh, which
# means they are NOT set on a login node. That has now cost three compile
# failures that all looked like missing headers but were empty -I flags
# (-I/include, -I$SPERR_PREFIX/include with SPERR_PREFIX unset, ...).
#
# This file is the single place those paths live for interactive use. It does
# NOT set LD_PRELOAD -- source zfp64_preamble.sh for that, and only when you
# actually want the zfp swap and the sperr plugin.

export LP_VIEW=/home/kdougherty/libpressio_cuda/.spack-env/view
export SPERR_PREFIX=/gpfs/fs1/home/kdougherty/spack/opt/spack/linux-zen2/sperr-0.8.2-qsqz7rfwktbmkju43ucbrmtpiyn6wyb6

# lib vs lib64 differs between these two prefixes; resolve rather than guess.
for d in "$LP_VIEW/lib64" "$LP_VIEW/lib"; do
    [ -e "$d/liblibpressio.so" ] && { export LP_LIB="$d"; break; }
done
for d in "$SPERR_PREFIX/lib64" "$SPERR_PREFIX/lib"; do
    ls "$d"/libSPERR.so* >/dev/null 2>&1 && { export SPERR_LIB="$d"; break; }
done

echo "LP_VIEW      = $LP_VIEW"
echo "LP_LIB       = ${LP_LIB:-<NOT FOUND>}"
echo "SPERR_PREFIX = $SPERR_PREFIX"
echo "SPERR_LIB    = ${SPERR_LIB:-<NOT FOUND>}"
[ -f "$SPERR_PREFIX/include/SPERR_C_API.h" ] \
    && echo "SPERR header = $SPERR_PREFIX/include/SPERR_C_API.h" \
    || echo "SPERR header = NOT at \$SPERR_PREFIX/include/SPERR_C_API.h -- run:
    find $SPERR_PREFIX -name 'SPERR_C_API.h'"

cat <<'EOF'

build the probe:
    g++ -O2 -std=c++17 sperr_trunc_probe.cpp -o sperr_trunc_probe \
        -I$SPERR_PREFIX/include -L$SPERR_LIB -lSPERR -Wl,-rpath,$SPERR_LIB

build the option dumper:
    g++ -O2 -std=c++17 dump_opts.cpp -o dump_opts \
        -I$LP_VIEW/include -L$LP_LIB -llibpressio -Wl,-rpath,$LP_LIB
EOF
