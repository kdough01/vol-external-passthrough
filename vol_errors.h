#ifndef VOL_ERRORS_H
#define VOL_ERRORS_H

#include "hdf5.h"

extern hid_t vol_err_class;
extern hid_t maj_compression;
extern hid_t min_compressor_unavail;
extern hid_t min_compress_failed;
extern hid_t min_decompress_failed;

#endif