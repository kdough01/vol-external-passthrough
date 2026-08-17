set -u
 
SRC="${DUMP_SRC:-dump_opts.cc}"
OUT="${DUMP_OUT:-./dump_opts}"
WHAT="${1:-sperr}"
 
LP_VIEW="${LP_VIEW:-/home/kdougherty/libpressio_cuda/.spack-env/view}"
 
echo "=== paths ==="
echo "LP_VIEW : $LP_VIEW"
[ -d "$LP_VIEW" ] || { echo "ERROR: LP_VIEW is not a directory."; exit 1; }
 
# --- header -----------------------------------------------------------------
INC=""
for d in "$LP_VIEW/include"; do
    [ -f "$d/libpressio/libpressio.h" ] && { INC="$d"; break; }
done
if [ -z "$INC" ]; then
    echo "ERROR: libpressio/libpressio.h not found under $LP_VIEW/include"
    echo "--- what IS there ---"
    find "$LP_VIEW/include" -maxdepth 2 -name 'libpressio*' 2>/dev/null | head -20
    echo
    echo "If this comes up empty the spack view may be stale or the wrong env."
    echo "Try:  ls -l $LP_VIEW ; spack env status"
    exit 1
fi
echo "header  : $INC/libpressio/libpressio.h"
 
# --- library ----------------------------------------------------------------
LIBDIR=""
for d in "$LP_VIEW/lib64" "$LP_VIEW/lib"; do
    if ls "$d"/liblibpressio.so* >/dev/null 2>&1 || \
       ls "$d"/liblibpressio.a   >/dev/null 2>&1; then LIBDIR="$d"; break; fi
done
if [ -z "$LIBDIR" ]; then
    echo "ERROR: liblibpressio.so not found in $LP_VIEW/{lib64,lib}"
    find "$LP_VIEW" -maxdepth 2 -name 'liblibpressio*' 2>/dev/null | head
    exit 1
fi
echo "library : $LIBDIR/$(basename $(ls "$LIBDIR"/liblibpressio.so* 2>/dev/null | head -1))"
 
# --- source -----------------------------------------------------------------
if [ ! -f "$SRC" ]; then
    echo "ERROR: $SRC not found in $(pwd)."
    echo "Note: dump_opts.cpp is the NEW generic dumper, not your existing"
    echo "dump_cuszp_opts.cc. Save it here first, or set DUMP_SRC=<file>."
    exit 1
fi
echo "source  : $SRC"
 
# --- build ------------------------------------------------------------------
echo
echo "=== building ==="
set -x
g++ -O2 -std=c++17 "$SRC" -o "$OUT" \
    -I"$INC" -L"$LIBDIR" -llibpressio \
    -Wl,-rpath,"$LIBDIR"
rc=$?
set +x
[ $rc -eq 0 ] || { echo "build failed rc=$rc"; exit $rc; }
 
echo
echo "=== running: $OUT $WHAT ==="
LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}" "$OUT" "$WHAT"
rc=$?
 
if [ $rc -eq 0 ]; then
    echo
    echo "=== THE ANSWER: does sperr expose chunk dimensions? ==="
    LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}" "$OUT" "$WHAT" \
        | tr ',' '\n' | grep -i chunk
    if [ ${PIPESTATUS[1]} -ne 0 ] 2>/dev/null; then :; fi
    echo "(no output above = no chunk option = Plan B, use the 256^3 crop)"
fi
exit $rc
