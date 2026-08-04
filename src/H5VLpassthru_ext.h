/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://support.hdfgroup.org/ftp/HDF5/releases.  *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 * Purpose:	The public header file for the pass-through VOL connector.
 */

#ifndef _H5VLpassthru_ext_H
#define _H5VLpassthru_ext_H

/* Public headers needed by this file */
#include "H5VLpublic.h"        /* Virtual Object Layer                 */
#include <libpressio/libpressio.h>

/* Identifier for the pass-through VOL connector */
#define H5VL_PASSTHRU_EXT	(H5VL_pass_through_ext_register())

/* Public characteristics of the pass-through VOL connector */
#define H5VL_PASSTHRU_EXT_NAME        "pass_through_ext"
#define H5VL_PASSTHRU_EXT_VALUE       517           /* VOL connector ID */

#define VOL_CHUNK_MAGIC     0x564F4C43484B3032ULL   /* "VOLCHK02" */
#define VOL_CHUNK_HDR_WORDS 3                        /* magic, nchunks, chunk_bytes */

#ifndef VOL_NATIVE_MAGIC
#define VOL_NATIVE_MAGIC ((uint64_t)0x564F4C4E41544956ULL)
#endif

/* chunking modes (compression_ctx.chunking_mode) */
#define VOL_CHUNKING_NONE    0
#define VOL_CHUNKING_VOL     1
#define VOL_CHUNKING_PRESSIO 2
#define VOL_CHUNKING_SHARED  3

/* pressio-chunked container: [magic][uint64 csize][uint64 chunk_elems][payload] */
#define VOL_PRESSIO_MAGIC     0x564F4C5052534F31ULL   /* "VOLPRSO1" */
#define VOL_PRESSIO_HDR_WORDS 3

/* Pass-through VOL connector info */
typedef struct H5VL_pass_through_ext_info_t {
    hid_t under_vol_id;         /* VOL ID for under VOL */
    void *under_vol_info;       /* VOL info for under VOL */
} H5VL_pass_through_ext_info_t;

#ifdef __cplusplus
extern "C" {
#endif

/* Technically a private function call, but prototype must be declared here */
extern hid_t H5VL_pass_through_ext_register(void);

#ifdef __cplusplus
}
#endif

#endif /* _H5VLpassthru_H */

