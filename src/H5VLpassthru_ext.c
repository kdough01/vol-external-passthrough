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
 * Purpose:     This is a "pass through" VOL connector, which forwards each
 *              VOL callback to an underlying connector.
 *
 *              It is designed as an example VOL connector for developers to
 *              use when creating new connectors, especially connectors that
 *              are outside of the HDF5 library.  As such, it should _NOT_
 *              include _any_ private HDF5 header files.  This connector should
 *              therefore only make public HDF5 API calls and use standard C /
 *              POSIX calls.
 *
 *              Note that the HDF5 error stack must be preserved on code paths
 *              that could be invoked when the underlying VOL connector's
 *              callback can fail.
 *
 */


/* Header files needed */
/* Do NOT include private HDF5 files here! */
#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Public HDF5 headers */
#include "hdf5.h"

/* This connector's private header */
#include "H5VLpassthru_ext_private.h"
#include <libpressio/libpressio.h>
#include <libpressio_ext/json/pressio_options_json.h>
#include "metadata_structs.h"
#include "vol_errors.h"
#include "vol_timing_sink.h"

/**********/
/* Macros */
/**********/

/* Whether to display log messge when callback is invoked */
/* (Uncomment to enable) */
/* #define ENABLE_EXT_PASSTHRU_LOGGING */
// #define ENABLE_EXT_PASSTHRU_LOGGING

/* Hack for missing va_copy() in old Visual Studio editions
 * (from H5win2_defs.h - used on VS2012 and earlier)
 */
#if defined(_WIN32) && defined(_MSC_VER) && (_MSC_VER < 1800)
#define va_copy(D,S)      ((D) = (S))
#endif

/************/
/* Typedefs */
/************/

/* The pass through VOL connector's object */
typedef struct H5VL_pass_through_ext_t {
    hid_t  under_vol_id;        /* ID for underlying VOL connector */
    void   *under_object;       /* Underlying VOL connector's object */
    void   *custom_data;      /* configuration parameters for the GPU */
} H5VL_pass_through_ext_t;

/* The pass through VOL wrapper context */
typedef struct H5VL_pass_through_ext_wrap_ctx_t {
    hid_t under_vol_id;         /* VOL ID for under VOL */
    void *under_wrap_ctx;       /* Object wrapping context for under VOL */
} H5VL_pass_through_ext_wrap_ctx_t;

/********************* */
/* Function prototypes */
/********************* */

/* Helper routines */
static H5VL_pass_through_ext_t *H5VL_pass_through_ext_new_obj(void *under_obj,
    hid_t under_vol_id);
static herr_t H5VL_pass_through_ext_free_obj(H5VL_pass_through_ext_t *obj);

/* "Management" callbacks */
static herr_t H5VL_pass_through_ext_init(hid_t vipl_id);
static herr_t H5VL_pass_through_ext_term(void);

/* VOL info callbacks */
static void *H5VL_pass_through_ext_info_copy(const void *info);
static herr_t H5VL_pass_through_ext_info_cmp(int *cmp_value, const void *info1, const void *info2);
static herr_t H5VL_pass_through_ext_info_free(void *info);
static herr_t H5VL_pass_through_ext_info_to_str(const void *info, char **str);
static herr_t H5VL_pass_through_ext_str_to_info(const char *str, void **info);

/* VOL object wrap / retrieval callbacks */
static void *H5VL_pass_through_ext_get_object(const void *obj);
static herr_t H5VL_pass_through_ext_get_wrap_ctx(const void *obj, void **wrap_ctx);
static void *H5VL_pass_through_ext_wrap_object(void *obj, H5I_type_t obj_type,
    void *wrap_ctx);
static void *H5VL_pass_through_ext_unwrap_object(void *obj);
static herr_t H5VL_pass_through_ext_free_wrap_ctx(void *obj);

/* Attribute callbacks */
static void *H5VL_pass_through_ext_attr_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t type_id, hid_t space_id, hid_t acpl_id, hid_t aapl_id, hid_t dxpl_id, void **req);
static void *H5VL_pass_through_ext_attr_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t aapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_attr_read(void *attr, hid_t mem_type_id, void *buf, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_attr_write(void *attr, hid_t mem_type_id, const void *buf, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_attr_get(void *obj, H5VL_attr_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_attr_specific(void *obj, const H5VL_loc_params_t *loc_params, H5VL_attr_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_attr_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_attr_close(void *attr, hid_t dxpl_id, void **req);

/* Dataset callbacks */
static void *H5VL_pass_through_ext_dataset_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t lcpl_id, hid_t type_id, hid_t space_id, hid_t dcpl_id, hid_t dapl_id, hid_t dxpl_id, void **req);
static void *H5VL_pass_through_ext_dataset_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t dapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_dataset_read(size_t count, void *dset[],
        hid_t mem_type_id[], hid_t mem_space_id[], hid_t file_space_id[],
        hid_t plist_id, void *buf[], void **req);
static herr_t H5VL_pass_through_ext_dataset_write(size_t count, void *dset[],
        hid_t mem_type_id[], hid_t mem_space_id[], hid_t file_space_id[],
        hid_t plist_id, const void *buf[], void **req);
static herr_t H5VL_pass_through_ext_dataset_get(void *dset, H5VL_dataset_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_dataset_specific(void *obj, H5VL_dataset_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_dataset_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_dataset_close(void *dset, hid_t dxpl_id, void **req);

/* Datatype callbacks */
static void *H5VL_pass_through_ext_datatype_commit(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t type_id, hid_t lcpl_id, hid_t tcpl_id, hid_t tapl_id, hid_t dxpl_id, void **req);
static void *H5VL_pass_through_ext_datatype_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t tapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_datatype_get(void *dt, H5VL_datatype_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_datatype_specific(void *obj, H5VL_datatype_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_datatype_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_datatype_close(void *dt, hid_t dxpl_id, void **req);

/* File callbacks */
static void *H5VL_pass_through_ext_file_create(const char *name, unsigned flags, hid_t fcpl_id, hid_t fapl_id, hid_t dxpl_id, void **req);
static void *H5VL_pass_through_ext_file_open(const char *name, unsigned flags, hid_t fapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_file_get(void *file, H5VL_file_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_file_specific(void *file, H5VL_file_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_file_optional(void *file, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_file_close(void *file, hid_t dxpl_id, void **req);

/* Group callbacks */
static void *H5VL_pass_through_ext_group_create(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id, hid_t dxpl_id, void **req);
static void *H5VL_pass_through_ext_group_open(void *obj, const H5VL_loc_params_t *loc_params, const char *name, hid_t gapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_group_get(void *obj, H5VL_group_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_group_specific(void *obj, H5VL_group_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_group_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_group_close(void *grp, hid_t dxpl_id, void **req);

/* Link callbacks */
static herr_t H5VL_pass_through_ext_link_create(H5VL_link_create_args_t *args, void *obj, const H5VL_loc_params_t *loc_params, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_link_copy(void *src_obj, const H5VL_loc_params_t *loc_params1, void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_link_move(void *src_obj, const H5VL_loc_params_t *loc_params1, void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t lcpl_id, hid_t lapl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_link_get(void *obj, const H5VL_loc_params_t *loc_params, H5VL_link_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_link_specific(void *obj, const H5VL_loc_params_t *loc_params, H5VL_link_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_link_optional(void *obj, const H5VL_loc_params_t *loc_params, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);

/* Object callbacks */
static void *H5VL_pass_through_ext_object_open(void *obj, const H5VL_loc_params_t *loc_params, H5I_type_t *opened_type, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_object_copy(void *src_obj, const H5VL_loc_params_t *src_loc_params, const char *src_name, void *dst_obj, const H5VL_loc_params_t *dst_loc_params, const char *dst_name, hid_t ocpypl_id, hid_t lcpl_id, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_object_get(void *obj, const H5VL_loc_params_t *loc_params, H5VL_object_get_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_object_specific(void *obj, const H5VL_loc_params_t *loc_params, H5VL_object_specific_args_t *args, hid_t dxpl_id, void **req);
static herr_t H5VL_pass_through_ext_object_optional(void *obj, const H5VL_loc_params_t *loc_params, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);

/* Container/connector introspection callbacks */
static herr_t H5VL_pass_through_ext_introspect_get_conn_cls(void *obj, H5VL_get_conn_lvl_t lvl, const H5VL_class_t **conn_cls);
static herr_t H5VL_pass_through_ext_introspect_get_cap_flags(const void *info, uint64_t *cap_flags);
static herr_t H5VL_pass_through_ext_introspect_opt_query(void *obj, H5VL_subclass_t cls, int op_type, uint64_t *flags);

/* Async request callbacks */
static herr_t H5VL_pass_through_ext_request_wait(void *req, uint64_t timeout, H5VL_request_status_t *status);
static herr_t H5VL_pass_through_ext_request_notify(void *obj, H5VL_request_notify_t cb, void *ctx);
static herr_t H5VL_pass_through_ext_request_cancel(void *req, H5VL_request_status_t *status);
static herr_t H5VL_pass_through_ext_request_specific(void *req, H5VL_request_specific_args_t *args);
static herr_t H5VL_pass_through_ext_request_optional(void *req, H5VL_optional_args_t *args);
static herr_t H5VL_pass_through_ext_request_free(void *req);

/* Blob callbacks */
static herr_t H5VL_pass_through_ext_blob_put(void *obj, const void *buf, size_t size, void *blob_id, void *ctx);
static herr_t H5VL_pass_through_ext_blob_get(void *obj, const void *blob_id, void *buf, size_t size, void *ctx);
static herr_t H5VL_pass_through_ext_blob_specific(void *obj, void *blob_id, H5VL_blob_specific_args_t *args);
static herr_t H5VL_pass_through_ext_blob_optional(void *obj, void *blob_id, H5VL_optional_args_t *args);

/* Token callbacks */
static herr_t H5VL_pass_through_ext_token_cmp(void *obj, const H5O_token_t *token1, const H5O_token_t *token2, int *cmp_value);
static herr_t H5VL_pass_through_ext_token_to_str(void *obj, H5I_type_t obj_type, const H5O_token_t *token, char **token_str);
static herr_t H5VL_pass_through_ext_token_from_str(void *obj, H5I_type_t obj_type, const char *token_str, H5O_token_t *token);

/* Generic optional callback */
static herr_t H5VL_pass_through_ext_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req);

/* Compression Functions */
herr_t H5VL_pass_through_ext_transfer_compress(compression_ctx *ctx, const void *data, size_t nbytes);
herr_t H5VL_pass_through_ext_transfer_decompress(compression_ctx *ctx, const void *compressed_data, size_t compressed_size, void *output_buf);
herr_t H5VL_pass_through_ext_transfer_compress_chunk(compression_ctx *ctx, const void *buf, size_t nbytes, void **out_cbuf, uint64_t *out_csize);
herr_t H5VL_pass_through_ext_transfer_decompress_chunk(compression_ctx *ctx, const void *cbuf, size_t csize, void *out, size_t out_bytes);
herr_t H5VL_pass_through_ext_compress_native(compression_ctx *ctx, const void *data, size_t nbytes, size_t hdr_reserve, void **out_cbuf, uint64_t *out_csize);
herr_t H5VL_pass_through_ext_decompress_native(compression_ctx *ctx, const void *cbuf, size_t csize, void *out, size_t out_bytes);

int    H5VL_pass_through_ext_chunking_mode(const compression_ctx *ctx);
size_t H5VL_pass_through_ext_chunk_bytes(const compression_ctx *ctx, size_t total_bytes, size_t dsize);
void   H5VL_pass_through_ext_parse_chunking_opts(compression_ctx *ctx, struct pressio_options *opts);
herr_t H5VL_pass_through_ext_compress_pressio(compression_ctx *ctx, const void *data, size_t nbytes, size_t chunk_bytes_req, void **out_cbuf, uint64_t *out_csize, uint64_t *out_chunk_elems);
herr_t H5VL_pass_through_ext_decompress_pressio(compression_ctx *ctx, const void *cbuf, size_t csize, uint64_t chunk_elems, void *out, size_t out_bytes);


int H5VL_pass_through_ext_compressor_available(const char *compressor_id);
int H5VL_pass_through_ext_buf_is_device(const void *p);

/* Destroy Functions */
void config_params_destroy(config_params *p);

/* Error Handling */
hid_t vol_err_class          = H5I_INVALID_HID;
hid_t maj_compression        = H5I_INVALID_HID;
hid_t min_compressor_unavail = H5I_INVALID_HID;
hid_t min_compress_failed    = H5I_INVALID_HID;
hid_t min_decompress_failed  = H5I_INVALID_HID;
hid_t maj_config             = H5I_INVALID_HID;
hid_t min_config_missing     = H5I_INVALID_HID;

/*******************/
/* Local variables */
/*******************/

/* Pass through VOL connector class struct */
static const H5VL_class_t H5VL_pass_through_ext_g = {
    H5VL_VERSION,                                       /* VOL class struct version */
    (H5VL_class_value_t)H5VL_PASSTHRU_EXT_VALUE,        /* value        */
    H5VL_PASSTHRU_EXT_NAME,                             /* name         */
    H5VL_PASSTHRU_EXT_VERSION,                          /* connector version */
    0,                                                  /* capability flags */
    H5VL_pass_through_ext_init,                         /* initialize   */
    H5VL_pass_through_ext_term,                         /* terminate    */
    {                                           /* info_cls */
        sizeof(H5VL_pass_through_ext_info_t),           /* size    */
        H5VL_pass_through_ext_info_copy,                /* copy    */
        H5VL_pass_through_ext_info_cmp,                 /* compare */
        H5VL_pass_through_ext_info_free,                /* free    */
        H5VL_pass_through_ext_info_to_str,              /* to_str  */
        H5VL_pass_through_ext_str_to_info               /* from_str */
    },
    {                                           /* wrap_cls */
        H5VL_pass_through_ext_get_object,               /* get_object   */
        H5VL_pass_through_ext_get_wrap_ctx,             /* get_wrap_ctx */
        H5VL_pass_through_ext_wrap_object,              /* wrap_object  */
        H5VL_pass_through_ext_unwrap_object,            /* unwrap_object */
        H5VL_pass_through_ext_free_wrap_ctx             /* free_wrap_ctx */
    },
    {                                           /* attribute_cls */
        H5VL_pass_through_ext_attr_create,              /* create */
        H5VL_pass_through_ext_attr_open,                /* open */
        H5VL_pass_through_ext_attr_read,                /* read */
        H5VL_pass_through_ext_attr_write,               /* write */
        H5VL_pass_through_ext_attr_get,                 /* get */
        H5VL_pass_through_ext_attr_specific,            /* specific */
        H5VL_pass_through_ext_attr_optional,            /* optional */
        H5VL_pass_through_ext_attr_close                /* close */
    },
    {                                           /* dataset_cls */
        H5VL_pass_through_ext_dataset_create,           /* create */
        H5VL_pass_through_ext_dataset_open,             /* open */
        H5VL_pass_through_ext_dataset_read,             /* read */
        H5VL_pass_through_ext_dataset_write,            /* write */
        H5VL_pass_through_ext_dataset_get,              /* get */
        H5VL_pass_through_ext_dataset_specific,         /* specific */
        H5VL_pass_through_ext_dataset_optional,         /* optional */
        H5VL_pass_through_ext_dataset_close             /* close */
    },
    {                                           /* datatype_cls */
        H5VL_pass_through_ext_datatype_commit,          /* commit */
        H5VL_pass_through_ext_datatype_open,            /* open */
        H5VL_pass_through_ext_datatype_get,             /* get_size */
        H5VL_pass_through_ext_datatype_specific,        /* specific */
        H5VL_pass_through_ext_datatype_optional,        /* optional */
        H5VL_pass_through_ext_datatype_close            /* close */
    },
    {                                           /* file_cls */
        H5VL_pass_through_ext_file_create,              /* create */
        H5VL_pass_through_ext_file_open,                /* open */
        H5VL_pass_through_ext_file_get,                 /* get */
        H5VL_pass_through_ext_file_specific,            /* specific */
        H5VL_pass_through_ext_file_optional,            /* optional */
        H5VL_pass_through_ext_file_close                /* close */
    },
    {                                           /* group_cls */
        H5VL_pass_through_ext_group_create,             /* create */
        H5VL_pass_through_ext_group_open,               /* open */
        H5VL_pass_through_ext_group_get,                /* get */
        H5VL_pass_through_ext_group_specific,           /* specific */
        H5VL_pass_through_ext_group_optional,           /* optional */
        H5VL_pass_through_ext_group_close               /* close */
    },
    {                                           /* link_cls */
        H5VL_pass_through_ext_link_create,              /* create */
        H5VL_pass_through_ext_link_copy,                /* copy */
        H5VL_pass_through_ext_link_move,                /* move */
        H5VL_pass_through_ext_link_get,                 /* get */
        H5VL_pass_through_ext_link_specific,            /* specific */
        H5VL_pass_through_ext_link_optional             /* optional */
    },
    {                                           /* object_cls */
        H5VL_pass_through_ext_object_open,              /* open */
        H5VL_pass_through_ext_object_copy,              /* copy */
        H5VL_pass_through_ext_object_get,               /* get */
        H5VL_pass_through_ext_object_specific,          /* specific */
        H5VL_pass_through_ext_object_optional           /* optional */
    },
    {                                           /* introspect_cls */
        H5VL_pass_through_ext_introspect_get_conn_cls,  /* get_conn_cls */
        H5VL_pass_through_ext_introspect_get_cap_flags, /* get_cap_flags */
        H5VL_pass_through_ext_introspect_opt_query,     /* opt_query */
    },
    {                                           /* request_cls */
        H5VL_pass_through_ext_request_wait,             /* wait */
        H5VL_pass_through_ext_request_notify,           /* notify */
        H5VL_pass_through_ext_request_cancel,           /* cancel */
        H5VL_pass_through_ext_request_specific,         /* specific */
        H5VL_pass_through_ext_request_optional,         /* optional */
        H5VL_pass_through_ext_request_free              /* free */
    },
    {                                           /* blob_cls */
        H5VL_pass_through_ext_blob_put,                 /* put */
        H5VL_pass_through_ext_blob_get,                 /* get */
        H5VL_pass_through_ext_blob_specific,            /* specific */
        H5VL_pass_through_ext_blob_optional             /* optional */
    },
    {                                           /* token_cls */
        H5VL_pass_through_ext_token_cmp,                /* cmp */
        H5VL_pass_through_ext_token_to_str,             /* to_str */
        H5VL_pass_through_ext_token_from_str              /* from_str */
    },
    H5VL_pass_through_ext_optional                  /* optional */
};

/* The connector identification number, initialized at runtime */
static hid_t H5VL_PASSTHRU_EXT_g = H5I_INVALID_HID;

/* Operation values for new "API" routines */
/* These are initialized in the VOL connector's 'init' callback at runtime.
 *      It's good practice to reset them back to -1 in the 'term' callback.
 */
static int H5VL_passthru_dataset_foo_op_g = -1;
static int H5VL_passthru_dataset_bar_op_g = -1;
static int H5VL_passthru_group_fiddle_op_g = -1;

/* Required shim routines, to enable dynamic loading of shared library */
/* The HDF5 library _must_ find routines with these names and signatures
 *      for a shared library that contains a VOL connector to be detected
 *      and loaded at runtime.
 */
H5PL_type_t H5PLget_plugin_type(void) {return H5PL_TYPE_VOL;}
const void *H5PLget_plugin_info(void) {return &H5VL_pass_through_ext_g;}

/*******************/
/* Wrapper functions for compression metadata */
/*******************/

enum pressio_dtype hdf5_to_pressio_dtype(hid_t type_id) {

    size_t size = H5Tget_size(type_id);
    H5T_class_t cls = H5Tget_class(type_id);

    if (cls == H5T_FLOAT) {
        if (size == 4) return pressio_float_dtype;
        if (size == 8) return pressio_double_dtype;
    } else if (cls == H5T_INTEGER) {
        if (size == 4) return pressio_int32_dtype;
        if (size == 8) return pressio_int64_dtype;
    }

    return pressio_byte_dtype;
}

hid_t pressio_to_hdf5_dtype(enum pressio_dtype dt) {
    switch (dt) {
        case pressio_float_dtype:  return H5Tcopy(H5T_NATIVE_FLOAT);
        case pressio_double_dtype: return H5Tcopy(H5T_NATIVE_DOUBLE);
        case pressio_int32_dtype:  return H5Tcopy(H5T_NATIVE_INT32);
        case pressio_int64_dtype:  return H5Tcopy(H5T_NATIVE_INT64);
        case pressio_uint32_dtype: return H5Tcopy(H5T_NATIVE_UINT32);
        case pressio_uint64_dtype: return H5Tcopy(H5T_NATIVE_UINT64);
        case pressio_int8_dtype:   return H5Tcopy(H5T_NATIVE_INT8);
        case pressio_uint8_dtype:
        case pressio_byte_dtype:   return H5Tcopy(H5T_NATIVE_UCHAR);
        default:                   return H5Tcopy(H5T_NATIVE_UCHAR);
    }
}

config_params *config_params_create(hid_t fapl_id)
{
    (void)fapl_id;

    config_params *p = (config_params*)calloc(1, sizeof(config_params));
    if (!p) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_config, min_config_missing,
                "out of memory allocating config_params");
        return NULL;
    }

    const char *compressor = getenv("HDF5_VOL_PRESSIO_COMPRESSOR");
    const char *level      = getenv("HDF5_VOL_PRESSIO_LEVEL");

    // printf("DEBUG config: compressor='%s' level='%s'\n",
    //    compressor ? compressor : "(not set)",
    //    level      ? level      : "(not set)");

    p->default_compression_id  = strdup(compressor ? compressor : "noop");
    p->compression_level       = level ? (int)strtol(level, NULL, 10) : 1;

    return p;
}

datatype_ctx* datatype_ctx_create(hid_t dataset_id)
{
    datatype_ctx *dt_ctx = (datatype_ctx*)calloc(1, sizeof(datatype_ctx));

    hid_t dtype = H5Dget_type(dataset_id);
    hid_t space = H5Dget_space(dataset_id);

    dt_ctx->type = dtype;
    dt_ctx->rank = H5Sget_simple_extent_ndims(space);

    if (dt_ctx->rank > 0) {
        dt_ctx->dims = (hsize_t*)malloc(dt_ctx->rank * sizeof(hsize_t));
        H5Sget_simple_extent_dims(space, dt_ctx->dims, NULL);
    } else {
        dt_ctx->dims = NULL;
    }

    /* store VOL info if needed later */
    dt_ctx->under_obj = NULL;
    dt_ctx->under_vol = H5I_INVALID_HID;

    H5Sclose(space);
    H5Tclose(dtype);

    return dt_ctx;
}

chunking_ctx* chunking_ctx_create(hid_t dataset_id)
{
    chunking_ctx *chunk_ctx = (chunking_ctx*)calloc(1, sizeof(chunking_ctx));

    hid_t space = H5Dget_space(dataset_id);
    hid_t dcpl = H5Dget_create_plist(dataset_id);

    chunk_ctx->layout = H5Pget_layout(dataset_id);

    if (chunk_ctx->layout == H5D_CHUNKED) {
        H5Pget_chunk(dcpl, 2, chunk_ctx->chunk_dims);
    }

    int rank = H5Sget_simple_extent_ndims(space);
    chunk_ctx->chunk_dims = (hsize_t *)malloc(rank * sizeof(hsize_t));
    H5Sget_simple_extent_dims(space, chunk_ctx->chunk_dims, NULL);

    H5Pclose(dcpl);
    H5Sclose(space);

    return chunk_ctx;
}

void compression_ctx_destroy(compression_ctx *comp_ctx) {
    if (!comp_ctx) return;
    
    if (comp_ctx->compressor_opts) pressio_options_free(comp_ctx->compressor_opts);
    if (comp_ctx->compressor) pressio_compressor_release(comp_ctx->compressor);
    if (comp_ctx->library) pressio_release(comp_ctx->library);

    if (comp_ctx->chunk_wrapper) {
        pressio_compressor_release(comp_ctx->chunk_wrapper);
        comp_ctx->chunk_wrapper       = NULL;
        comp_ctx->chunk_wrapper_elems = 0;
    }
    
    free(comp_ctx->compressor_id);
    free(comp_ctx->dims);
    free(comp_ctx->compressed_buf);
    free(comp_ctx->decomp_buf);
    free(comp_ctx);
}

compression_ctx* compression_ctx_create(int rank, hsize_t *h5dims, enum pressio_dtype dtype, hid_t dcpl_id, config_params *defaults, const char *compressor_override)
{
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH COMPRESSION CTX\n");
#endif


    // printf("DEBUG ctx_create_entry: override='%s'\n", compressor_override ? compressor_override : "(null)");

    compression_ctx *comp_ctx = (compression_ctx*)calloc(1, sizeof(compression_ctx));

    if (compressor_override && compressor_override[0] != '\0') {
        comp_ctx->compressor_id = strdup(compressor_override);
    } else if (H5Pexist(dcpl_id, "pressio:compressor") > 0) {
        size_t len;
        H5Pget_size(dcpl_id, "pressio:compressor", &len);
        comp_ctx->compressor_id = (char*)malloc(len + 1);
        H5Pget(dcpl_id, "pressio:compressor", comp_ctx->compressor_id);
        comp_ctx->compressor_id[len] = '\0';
    } else {
        comp_ctx->compressor_id = strdup(defaults->default_compression_id);
    }

    char json_opts[4096] = "";
    if (H5Pexist(dcpl_id, "vol:options_json") > 0) {
        H5Pget(dcpl_id, "vol:options_json", json_opts);
    }

    comp_ctx->ndims = (size_t)rank;
    comp_ctx->dims = (size_t*)malloc(rank * sizeof(size_t));
        for (int i=0;i<rank;i++) {
        comp_ctx->dims[i] = (size_t)h5dims[i];
    }
    comp_ctx->dtype = dtype;

    // most of this follows the "basics.c" file in the libpressio tutorial with a few modifications like error handling
    // get the compressor
    comp_ctx->library = pressio_instance();
    comp_ctx->compressor = pressio_get_compressor(comp_ctx->library, comp_ctx->compressor_id);

    // printf("DEBUG ctx_create: id='%s' compressor=%p\n", comp_ctx->compressor_id, (void*)comp_ctx->compressor);

    // make sure the compressor specified exists and is known by libpressio
    if (!comp_ctx->compressor) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compressor_unavail,
                "compressor '%s' not found in libpressio registry: %s",
                comp_ctx->compressor_id, pressio_error_msg(comp_ctx->library));
        compression_ctx_destroy(comp_ctx);
        return NULL;
    }

    /* Configure metrics if requested */
    if (getenv("HDF5_VOL_PRESSIO_METRICS")) {
        struct pressio_options *metrics_opts = pressio_options_new();
        pressio_options_set_string(metrics_opts, "pressio:metric", "composite");
        
        const char *plugins[] = {"size", "time"};
        pressio_options_set_strings(metrics_opts, "composite:plugins", 2, plugins);
        
        pressio_compressor_set_options(comp_ctx->compressor, metrics_opts);
        pressio_options_free(metrics_opts);
    }

    // configure metrics for the compressor - this is default and overidden if JSON is present
    comp_ctx->compressor_opts = pressio_options_new();
    char level_key[128];
    snprintf(level_key, sizeof(level_key), "%s:compression_level", comp_ctx->compressor_id);
    pressio_options_set_integer(comp_ctx->compressor_opts, level_key, defaults->compression_level);

    if(pressio_compressor_set_options(comp_ctx->compressor, comp_ctx->compressor_opts)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "pressio_compressor_set_options failed for '%s': %s",
                comp_ctx->compressor_id,
                pressio_compressor_error_msg(comp_ctx->compressor));
        compression_ctx_destroy(comp_ctx);
        return NULL;
    }

    // JSON
    if (json_opts[0] != '\0') {
        struct pressio_options *opts = pressio_options_new_json(comp_ctx->library, json_opts);
        if (!opts) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "failed to parse JSON options for '%s': %s",
                    comp_ctx->compressor_id,
                    pressio_error_msg(comp_ctx->library));
            compression_ctx_destroy(comp_ctx);
            return NULL;
        }
        H5VL_pass_through_ext_parse_chunking_opts(comp_ctx, opts);
        pressio_compressor_set_options(comp_ctx->compressor, opts);
        pressio_options_free(opts);
    }

    comp_ctx->compressed_buf = NULL;
    comp_ctx->compressed_chunk_size = 0;

    comp_ctx->stage_buf    = NULL;
    comp_ctx->stage_total  = 0;
    comp_ctx->stage_filled = 0;

    return comp_ctx;
}

gpu_vol_dataset_t *gpu_vol_dataset_wrap(void *under_dataset,
                                        int rank, hsize_t *h5dims,
                                        hid_t type_id,
                                        enum pressio_dtype pressio_dt,
                                        hid_t dcpl_id,
                                        hid_t under_vol_id,
                                        gpu_vol_file_t *file_ctx,
                                        const char *compressor_override)
{
    gpu_vol_dataset_t *gpu_dataset_ctx =
        (gpu_vol_dataset_t *)calloc(1, sizeof(gpu_vol_dataset_t));
    gpu_dataset_ctx->under_dataset = under_dataset;
    gpu_dataset_ctx->under_vol_id  = under_vol_id;
    gpu_dataset_ctx->file_ctx      = file_ctx;
    gpu_dataset_ctx->comp_ctx = compression_ctx_create(rank, h5dims, pressio_dt,
                                                       dcpl_id,
                                                       file_ctx->config_params,
                                                       compressor_override);
    return gpu_dataset_ctx;
}

void gpu_vol_file_destroy(gpu_vol_file_t *file_ctx) {
    if (!file_ctx) return;

    config_params_destroy(file_ctx->config_params);

    free(file_ctx);
}

void gpu_vol_dataset_destroy(gpu_vol_dataset_t *ds_ctx)
{
    if (!ds_ctx) return;

    if (ds_ctx->comp_ctx)
        compression_ctx_destroy(ds_ctx->comp_ctx);

    free(ds_ctx);
}

void config_params_destroy(config_params *p) {
    if (!p) return;
    free(p->default_compression_id);
    free(p);
}

gpu_vol_file_t* gpu_vol_file_wrap(hid_t fapl_id, hid_t under_vol_id, void *under_file)
{
    gpu_vol_file_t *ctx = (gpu_vol_file_t*)calloc(1, sizeof(gpu_vol_file_t));
    if (!ctx) return NULL;

    ctx->under_file   = under_file;
    ctx->under_vol_id = under_vol_id;
    ctx->config_params = config_params_create(fapl_id);
    if (!ctx->config_params) { free(ctx); return NULL; }

    return ctx;
}

static herr_t
vol_scatter_cb(const void **data_out, size_t *len_out, void *op_data)
{
    vol_scatter_ctx *s = (vol_scatter_ctx *)op_data;
    size_t remaining = s->nbytes - s->off;
    *data_out = s->buf + s->off;
    *len_out  = remaining;
    s->off    = s->nbytes;
    return 0;
}

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


/*-------------------------------------------------------------------------
 * Function:    H5VL__pass_through_new_obj
 *
 * Purpose:     Create a new pass through object for an underlying object
 *
 * Return:      Success:    Pointer to the new pass through object
 *              Failure:    NULL
 *
 * Programmer:  Quincey Koziol
 *              Monday, December 3, 2018
 *
 *-------------------------------------------------------------------------
 */
static H5VL_pass_through_ext_t *
H5VL_pass_through_ext_new_obj(void *under_obj, hid_t under_vol_id)
{
    H5VL_pass_through_ext_t *new_obj;

    new_obj = (H5VL_pass_through_ext_t *)calloc(1, sizeof(H5VL_pass_through_ext_t));
    new_obj->under_object = under_obj;
    new_obj->under_vol_id = under_vol_id;
    H5Iinc_ref(new_obj->under_vol_id);

    return new_obj;
} /* end H5VL__pass_through_new_obj() */


/*-------------------------------------------------------------------------
 * Function:    H5VL__pass_through_free_obj
 *
 * Purpose:     Release a pass through object
 *
 * Note:	Take care to preserve the current HDF5 error stack
 *		when calling HDF5 API calls.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 * Programmer:  Quincey Koziol
 *              Monday, December 3, 2018
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_free_obj(H5VL_pass_through_ext_t *obj)
{
    hid_t err_id;

    err_id = H5Eget_current_stack();

    H5Idec_ref(obj->under_vol_id);

    H5Eset_current_stack(err_id);

    free(obj);

    return 0;
} /* end H5VL__pass_through_free_obj() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_register
 *
 * Purpose:     Register the pass-through VOL connector and retrieve an ID
 *              for it.
 *
 * Return:      Success:    The ID for the pass-through VOL connector
 *              Failure:    -1
 *
 * Programmer:  Quincey Koziol
 *              Wednesday, November 28, 2018
 *
 *-------------------------------------------------------------------------
 */
hid_t
H5VL_pass_through_ext_register(void)
{
    /* Singleton register the pass-through VOL connector ID */
    if(H5VL_PASSTHRU_EXT_g < 0)
        H5VL_PASSTHRU_EXT_g = H5VLregister_connector(&H5VL_pass_through_ext_g, H5P_DEFAULT);

    return H5VL_PASSTHRU_EXT_g;
} /* end H5VL_pass_through_ext_register() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_init
 *
 * Purpose:     Initialize this VOL connector, performing any necessary
 *              operations for the connector that will apply to all containers
 *              accessed with the connector.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_init(hid_t vipl_id)
{
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL INIT\n");
#endif

    /* Shut compiler up about unused parameter */
    (void)vipl_id;

    /* Acquire operation values for new "API" routines to use */
    assert(-1 == H5VL_passthru_dataset_foo_op_g);
    if(H5VLregister_opt_operation(H5VL_SUBCLS_DATASET, H5VL_PASSTHRU_EXT_DYN_FOO, &H5VL_passthru_dataset_foo_op_g) < 0)
        return(-1);
    assert(-1 != H5VL_passthru_dataset_foo_op_g);
    assert(-1 == H5VL_passthru_dataset_bar_op_g);
    if(H5VLregister_opt_operation(H5VL_SUBCLS_DATASET, H5VL_PASSTHRU_EXT_DYN_BAR, &H5VL_passthru_dataset_bar_op_g) < 0)
        return(-1);
    assert(-1 != H5VL_passthru_dataset_bar_op_g);
    assert(-1 == H5VL_passthru_group_fiddle_op_g);
    if(H5VLregister_opt_operation(H5VL_SUBCLS_GROUP, H5VL_PASSTHRU_EXT_DYN_FIDDLE, &H5VL_passthru_group_fiddle_op_g) < 0)
        return(-1);
    assert(-1 != H5VL_passthru_group_fiddle_op_g);

    /* Register error class and codes for this VOL connector */
    vol_err_class          = H5Eregister_class("VOL Passthrough", "vol_passthrough", "1.0");
    maj_compression        = H5Ecreate_msg(vol_err_class, H5E_MAJOR, "Compression");
    min_compressor_unavail = H5Ecreate_msg(vol_err_class, H5E_MINOR, "Compressor not available");
    min_compress_failed    = H5Ecreate_msg(vol_err_class, H5E_MINOR, "Compression failed");
    min_decompress_failed  = H5Ecreate_msg(vol_err_class, H5E_MINOR, "Decompression failed");
    min_config_missing = H5Ecreate_msg(vol_err_class, H5E_MINOR, "Configuration missing or invalid");
    maj_config = H5Ecreate_msg(vol_err_class, H5E_MAJOR, "Configuration");
    if(vol_err_class < 0 || maj_compression < 0 || min_compressor_unavail < 0 ||
        min_compress_failed < 0 || min_decompress_failed < 0)
        return(-1);
        
    // H5Eset_auto(H5E_DEFAULT, (H5E_auto2_t)H5Eprint, stderr);
    return 0;
} /* end H5VL_pass_through_ext_init() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_term
 *
 * Purpose:     Terminate this VOL connector, performing any necessary
 *              operations for the connector that release connector-wide
 *              resources (usually created / initialized with the 'init'
 *              callback).
 *
 * Return:      Success:    0
 *              Failure:    (Can't fail)
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_term(void)
{
#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL TERM\n");
#endif

    /* Reset VOL ID */
    H5VL_PASSTHRU_EXT_g = H5I_INVALID_HID;

    /* Reset operation values for new "API" routines */
    if(-1 != H5VL_passthru_dataset_foo_op_g) {
        if(H5VLunregister_opt_operation(H5VL_SUBCLS_DATASET, H5VL_PASSTHRU_EXT_DYN_FOO) < 0)
            return(-1);
        H5VL_passthru_dataset_foo_op_g = (-1);
    } /* end if */
    if(-1 != H5VL_passthru_dataset_bar_op_g) {
        if(H5VLunregister_opt_operation(H5VL_SUBCLS_DATASET, H5VL_PASSTHRU_EXT_DYN_BAR) < 0)
            return(-1);
        H5VL_passthru_dataset_bar_op_g = (-1);
    } /* end if */
    if(-1 != H5VL_passthru_group_fiddle_op_g) {
        if(H5VLunregister_opt_operation(H5VL_SUBCLS_GROUP, H5VL_PASSTHRU_EXT_DYN_FIDDLE) < 0)
            return(-1);
        H5VL_passthru_group_fiddle_op_g = (-1);
    } /* end if */

    /* Clean up error class and codes */
    // if(H5I_INVALID_HID != min_decompress_failed) {
    //     H5Eclose_msg(min_decompress_failed);
    //     min_decompress_failed = H5I_INVALID_HID;
    // }
    // if(H5I_INVALID_HID != min_compress_failed) {
    //     H5Eclose_msg(min_compress_failed);
    //     min_compress_failed = H5I_INVALID_HID;
    // }
    // if(H5I_INVALID_HID != min_compressor_unavail) {
    //     H5Eclose_msg(min_compressor_unavail);
    //     min_compressor_unavail = H5I_INVALID_HID;
    // }
    // if(H5I_INVALID_HID != maj_compression) {
    //     H5Eclose_msg(maj_compression);
    //     maj_compression = H5I_INVALID_HID;
    // }
    // if(H5I_INVALID_HID != vol_err_class) {
    //     H5Eunregister_class(vol_err_class);
    //     vol_err_class = H5I_INVALID_HID;
    // }

    // if(H5I_INVALID_HID != min_config_missing) {
    //     H5Eclose_msg(min_config_missing);
    //     min_config_missing = H5I_INVALID_HID;
    // }

    // H5Eset_auto(H5E_DEFAULT, NULL, NULL);

    return 0;
} /* end H5VL_pass_through_ext_term() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_info_copy
 *
 * Purpose:     Duplicate the connector's info object.
 *
 * Returns:     Success:    New connector info object
 *              Failure:    NULL
 *
 *---------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_info_copy(const void *_info)
{
    const H5VL_pass_through_ext_info_t *info = (const H5VL_pass_through_ext_info_t *)_info;
    H5VL_pass_through_ext_info_t *new_info;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL INFO Copy\n");
#endif

    /* Allocate new VOL info struct for the pass through connector */
    new_info = (H5VL_pass_through_ext_info_t *)calloc(1, sizeof(H5VL_pass_through_ext_info_t));

    /* Increment reference count on underlying VOL ID, and copy the VOL info */
    new_info->under_vol_id = info->under_vol_id;
    H5Iinc_ref(new_info->under_vol_id);
    if(info->under_vol_info)
        H5VLcopy_connector_info(new_info->under_vol_id, &(new_info->under_vol_info), info->under_vol_info);

    return new_info;
} /* end H5VL_pass_through_ext_info_copy() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_info_cmp
 *
 * Purpose:     Compare two of the connector's info objects, setting *cmp_value,
 *              following the same rules as strcmp().
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_info_cmp(int *cmp_value, const void *_info1, const void *_info2)
{
    const H5VL_pass_through_ext_info_t *info1 = (const H5VL_pass_through_ext_info_t *)_info1;
    const H5VL_pass_through_ext_info_t *info2 = (const H5VL_pass_through_ext_info_t *)_info2;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL INFO Compare\n");
#endif

    /* Sanity checks */
    assert(info1);
    assert(info2);

    /* Initialize comparison value */
    *cmp_value = 0;

    /* Compare under VOL connector classes */
    H5VLcmp_connector_cls(cmp_value, info1->under_vol_id, info2->under_vol_id);
    if(*cmp_value != 0)
        return 0;

    /* Compare under VOL connector info objects */
    H5VLcmp_connector_info(cmp_value, info1->under_vol_id, info1->under_vol_info, info2->under_vol_info);
    if(*cmp_value != 0)
        return 0;

    return 0;
} /* end H5VL_pass_through_ext_info_cmp() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_info_free
 *
 * Purpose:     Release an info object for the connector.
 *
 * Note:	Take care to preserve the current HDF5 error stack
 *		when calling HDF5 API calls.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_info_free(void *_info)
{
    H5VL_pass_through_ext_info_t *info = (H5VL_pass_through_ext_info_t *)_info;
    hid_t err_id;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL INFO Free\n");
#endif

    err_id = H5Eget_current_stack();

    /* Release underlying VOL ID and info */
    if(info->under_vol_info)
        H5VLfree_connector_info(info->under_vol_id, info->under_vol_info);
    H5Idec_ref(info->under_vol_id);

    H5Eset_current_stack(err_id);

    /* Free pass through info object itself */
    free(info);

    return 0;
} /* end H5VL_pass_through_ext_info_free() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_info_to_str
 *
 * Purpose:     Serialize an info object for this connector into a string
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_info_to_str(const void *_info, char **str)
{
    const H5VL_pass_through_ext_info_t *info = (const H5VL_pass_through_ext_info_t *)_info;
    H5VL_class_value_t under_value = (H5VL_class_value_t)-1;
    char *under_vol_string = NULL;
    size_t under_vol_str_len = 0;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL INFO To String\n");
#endif

    /* Get value and string for underlying VOL connector */
    H5VLget_value(info->under_vol_id, &under_value);
    H5VLconnector_info_to_str(info->under_vol_info, info->under_vol_id, &under_vol_string);

    /* Determine length of underlying VOL info string */
    if(under_vol_string)
        under_vol_str_len = strlen(under_vol_string);

    /* Allocate space for our info */
    *str = (char *)H5allocate_memory(32 + under_vol_str_len, (hbool_t)0);
    assert(*str);

    /* Encode our info
     * Normally we'd use snprintf() here for a little extra safety, but that
     * call had problems on Windows until recently. So, to be as platform-independent
     * as we can, we're using sprintf() instead.
     */
    sprintf(*str, "under_vol=%u;under_info={%s}", (unsigned)under_value, (under_vol_string ? under_vol_string : ""));

    /* Release under VOL info string, if there is one */
    if(under_vol_string)
        H5free_memory(under_vol_string);

    return 0;
} /* end H5VL_pass_through_ext_info_to_str() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_str_to_info
 *
 * Purpose:     Deserialize a string into an info object for this connector.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_str_to_info(const char *str, void **_info)
{
    H5VL_pass_through_ext_info_t *info;
    unsigned under_vol_value;
    const char *under_vol_info_start, *under_vol_info_end;
    hid_t under_vol_id;
    void *under_vol_info = NULL;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL INFO String To Info\n");
#endif

    /* Retrieve the underlying VOL connector value and info */
    sscanf(str, "under_vol=%u;", &under_vol_value);
    under_vol_id = H5VLregister_connector_by_value((H5VL_class_value_t)under_vol_value, H5P_DEFAULT);
    under_vol_info_start = strchr(str, '{');
    under_vol_info_end = strrchr(str, '}');
    assert(under_vol_info_end > under_vol_info_start);
    if(under_vol_info_end != (under_vol_info_start + 1)) {
        char *under_vol_info_str;

        under_vol_info_str = (char *)malloc((size_t)(under_vol_info_end - under_vol_info_start));
        memcpy(under_vol_info_str, under_vol_info_start + 1, (size_t)((under_vol_info_end - under_vol_info_start) - 1));
        *(under_vol_info_str + (under_vol_info_end - under_vol_info_start)) = '\0';

        H5VLconnector_str_to_info(under_vol_info_str, under_vol_id, &under_vol_info);

        free(under_vol_info_str);
    } /* end else */

    /* Allocate new pass-through VOL connector info and set its fields */
    info = (H5VL_pass_through_ext_info_t *)calloc(1, sizeof(H5VL_pass_through_ext_info_t));
    info->under_vol_id = under_vol_id;
    info->under_vol_info = under_vol_info;

    /* Set return value */
    *_info = info;

    return 0;
} /* end H5VL_pass_through_ext_str_to_info() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_get_object
 *
 * Purpose:     Retrieve the 'data' for a VOL object.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_get_object(const void *obj)
{
    const H5VL_pass_through_ext_t *o = (const H5VL_pass_through_ext_t *)obj;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL Get object\n");
#endif

    return H5VLget_object(o->under_object, o->under_vol_id);
} /* end H5VL_pass_through_ext_get_object() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_get_wrap_ctx
 *
 * Purpose:     Retrieve a "wrapper context" for an object
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_get_wrap_ctx(const void *obj, void **wrap_ctx)
{
    const H5VL_pass_through_ext_t *o = (const H5VL_pass_through_ext_t *)obj;
    H5VL_pass_through_ext_wrap_ctx_t *new_wrap_ctx;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL WRAP CTX Get\n");
#endif

    /* Allocate new VOL object wrapping context for the pass through connector */
    new_wrap_ctx = (H5VL_pass_through_ext_wrap_ctx_t *)calloc(1, sizeof(H5VL_pass_through_ext_wrap_ctx_t));

    /* Increment reference count on underlying VOL ID, and copy the VOL info */
    new_wrap_ctx->under_vol_id = o->under_vol_id;
    H5Iinc_ref(new_wrap_ctx->under_vol_id);
    H5VLget_wrap_ctx(o->under_object, o->under_vol_id, &new_wrap_ctx->under_wrap_ctx);

    /* Set wrap context to return */
    *wrap_ctx = new_wrap_ctx;

    return 0;
} /* end H5VL_pass_through_ext_get_wrap_ctx() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_wrap_object
 *
 * Purpose:     Use a "wrapper context" to wrap a data object
 *
 * Return:      Success:    Pointer to wrapped object
 *              Failure:    NULL
 *
 *---------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_wrap_object(void *obj, H5I_type_t obj_type, void *_wrap_ctx)
{
    H5VL_pass_through_ext_wrap_ctx_t *wrap_ctx = (H5VL_pass_through_ext_wrap_ctx_t *)_wrap_ctx;
    H5VL_pass_through_ext_t *new_obj;
    void *under;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL WRAP Object\n");
#endif

    /* Wrap the object with the underlying VOL */
    under = H5VLwrap_object(obj, obj_type, wrap_ctx->under_vol_id, wrap_ctx->under_wrap_ctx);
    if(under)
        new_obj = H5VL_pass_through_ext_new_obj(under, wrap_ctx->under_vol_id);
    else
        new_obj = NULL;

    return new_obj;
} /* end H5VL_pass_through_ext_wrap_object() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_unwrap_object
 *
 * Purpose:     Unwrap a wrapped object, discarding the wrapper, but returning
 *		underlying object.
 *
 * Return:      Success:    Pointer to unwrapped object
 *              Failure:    NULL
 *
 *---------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_unwrap_object(void *obj)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    void *under;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL UNWRAP Object\n");
#endif

    /* Unrap the object with the underlying VOL */
    under = H5VLunwrap_object(o->under_object, o->under_vol_id);

    if(under)
        H5VL_pass_through_ext_free_obj(o);

    return under;
} /* end H5VL_pass_through_ext_unwrap_object() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_free_wrap_ctx
 *
 * Purpose:     Release a "wrapper context" for an object
 *
 * Note:	Take care to preserve the current HDF5 error stack
 *		when calling HDF5 API calls.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_free_wrap_ctx(void *_wrap_ctx)
{
    H5VL_pass_through_ext_wrap_ctx_t *wrap_ctx = (H5VL_pass_through_ext_wrap_ctx_t *)_wrap_ctx;
    hid_t err_id;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL WRAP CTX Free\n");
#endif

    err_id = H5Eget_current_stack();

    /* Release underlying VOL ID and wrap context */
    if(wrap_ctx->under_wrap_ctx)
        H5VLfree_wrap_ctx(wrap_ctx->under_wrap_ctx, wrap_ctx->under_vol_id);
    H5Idec_ref(wrap_ctx->under_vol_id);

    H5Eset_current_stack(err_id);

    /* Free pass through wrap context object itself */
    free(wrap_ctx);

    return 0;
} /* end H5VL_pass_through_ext_free_wrap_ctx() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_attr_create
 *
 * Purpose:     Creates an attribute on an object.
 *
 * Return:      Success:    Pointer to attribute object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_attr_create(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t type_id, hid_t space_id, hid_t acpl_id,
    hid_t aapl_id, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *attr;
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    void *under;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL ATTRIBUTE Create\n");
#endif

    under = H5VLattr_create(o->under_object, loc_params, o->under_vol_id, name, type_id, space_id, acpl_id, aapl_id, dxpl_id, req);
    if(under) {
        attr = H5VL_pass_through_ext_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        attr = NULL;

    return (void*)attr;
} /* end H5VL_pass_through_ext_attr_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_attr_open
 *
 * Purpose:     Opens an attribute on an object.
 *
 * Return:      Success:    Pointer to attribute object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_attr_open(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t aapl_id, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *attr;
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    void *under;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL ATTRIBUTE Open\n");
#endif

    under = H5VLattr_open(o->under_object, loc_params, o->under_vol_id, name, aapl_id, dxpl_id, req);
    if(under) {
        attr = H5VL_pass_through_ext_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        attr = NULL;

    return (void *)attr;
} /* end H5VL_pass_through_ext_attr_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_attr_read
 *
 * Purpose:     Reads data from attribute.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_attr_read(void *attr, hid_t mem_type_id, void *buf,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)attr;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL ATTRIBUTE Read\n");
#endif

    ret_value = H5VLattr_read(o->under_object, o->under_vol_id, mem_type_id, buf, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_attr_read() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_attr_write
 *
 * Purpose:     Writes data to attribute.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_attr_write(void *attr, hid_t mem_type_id, const void *buf,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)attr;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL ATTRIBUTE Write\n");
#endif

    ret_value = H5VLattr_write(o->under_object, o->under_vol_id, mem_type_id, buf, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_attr_write() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_attr_get
 *
 * Purpose:     Gets information about an attribute
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_attr_get(void *obj, H5VL_attr_get_args_t *args, hid_t dxpl_id,
    void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL ATTRIBUTE Get\n");
#endif

    ret_value = H5VLattr_get(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_attr_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_attr_specific
 *
 * Purpose:     Specific operation on attribute
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_attr_specific(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_attr_specific_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL ATTRIBUTE Specific\n");
#endif

    ret_value = H5VLattr_specific(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_attr_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_attr_optional
 *
 * Purpose:     Perform a connector-specific operation on an attribute
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_attr_optional(void *obj, H5VL_optional_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL ATTRIBUTE Optional\n");
#endif

    ret_value = H5VLattr_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_attr_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_attr_close
 *
 * Purpose:     Closes an attribute.
 *
 * Return:      Success:    0
 *              Failure:    -1, attr not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_attr_close(void *attr, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)attr;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL ATTRIBUTE Close\n");
#endif

    ret_value = H5VLattr_close(o->under_object, o->under_vol_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    /* Release our wrapper, if underlying attribute was closed */
    if(ret_value >= 0)
        H5VL_pass_through_ext_free_obj(o);

    return ret_value;
} /* end H5VL_pass_through_ext_attr_close() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_dataset_create
 *
 * Purpose:     Creates a dataset in a container
 *
 * Return:      Success:    Pointer to a dataset object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_dataset_create(void *obj,
    const H5VL_loc_params_t *loc_params,
    const char *name,
    hid_t lcpl_id,
    hid_t type_id,
    hid_t space_id,
    hid_t dcpl_id,
    hid_t dapl_id,
    hid_t dxpl_id,
    void **req)
{
    H5VL_pass_through_ext_t *dset;
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    void *under;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL DATASET Create\n");
#endif

    gpu_vol_file_t *file_ctx = (gpu_vol_file_t*)o->custom_data;
    config_params *config_ctx = file_ctx ? file_ctx->config_params : NULL;

    /* Extract original N-dimensional shape info */
    int rank = H5Sget_simple_extent_ndims(space_id);
    hsize_t *h5dims = NULL;
    if (rank > 0) {
        h5dims = (hsize_t *)malloc(rank * sizeof(hsize_t));
        H5Sget_simple_extent_dims(space_id, h5dims, NULL);
    }
    enum pressio_dtype pressio_dt = hdf5_to_pressio_dtype(type_id);

    hid_t underlying_space_id = space_id;
    hid_t underlying_dcpl_id = dcpl_id;
    hid_t underlying_type_id = type_id;

    int do_compress = (config_ctx && file_ctx && file_ctx->compress_on_write);

    /* If compression is active, rewrite the creation parameters to 1D bytes */
    if (do_compress) {
        /* Create 1D array with unlimited space for compressed bytes */
        hsize_t byte_dims[1] = {0};
        hsize_t max_byte_dims[1] = {H5S_UNLIMITED};
        underlying_space_id = H5Screate_simple(1, byte_dims, max_byte_dims);

        /* Create a new DCPL and force chunking to allow extent resizing */
        hid_t temp_dcpl = (dcpl_id == H5P_DEFAULT) ? H5Pcreate(H5P_DATASET_CREATE) : dcpl_id;
        underlying_dcpl_id = H5Pcopy(temp_dcpl);
        if (dcpl_id == H5P_DEFAULT) {
            H5Pclose(temp_dcpl); 
        }

        hsize_t chunk_size[1] = { 1048576 }; 
        H5Pset_chunk(underlying_dcpl_id, 1, chunk_size);
        underlying_type_id = H5T_NATIVE_UCHAR;
    }

    /* Create the underlying dataset */
    under = H5VLdataset_create(
        o->under_object, loc_params, o->under_vol_id, name,
        lcpl_id, underlying_type_id, underlying_space_id,
        underlying_dcpl_id, dapl_id, dxpl_id, req
    );

    if (under) {
        dset = H5VL_pass_through_ext_new_obj(under, o->under_vol_id);

        if (do_compress) {
            /* Store the hidden attributes as metadata for when we want to open the dataset */
            H5VL_loc_params_t attr_loc;
            attr_loc.type = H5VL_OBJECT_BY_SELF;
            attr_loc.obj_type = H5I_DATASET;

            hid_t scalar_space = H5Screate(H5S_SCALAR);

            hid_t acpl_id = H5Pcreate(H5P_ATTRIBUTE_CREATE);
            hid_t aapl_id = H5Pcreate(H5P_ATTRIBUTE_ACCESS);

            void *attr_rank = H5VLattr_create(under, &attr_loc, o->under_vol_id,
                "_VOL_ORIG_RANK", H5T_NATIVE_INT, scalar_space,
                acpl_id, aapl_id, dxpl_id, NULL);
            if (attr_rank) {
                H5VLattr_write(attr_rank, o->under_vol_id, H5T_NATIVE_INT, &rank, dxpl_id, NULL);
                H5VLattr_close(attr_rank, o->under_vol_id, dxpl_id, NULL);
            }

            hsize_t dim_space_sz[1] = { (hsize_t)rank };
            hid_t dim_space = H5Screate_simple(1, dim_space_sz, NULL);

            void *attr_dims = H5VLattr_create(under, &attr_loc, o->under_vol_id,
                "_VOL_ORIG_DIMS", H5T_NATIVE_HSIZE, dim_space,
                acpl_id, aapl_id, dxpl_id, NULL);
            if (attr_dims) {
                H5VLattr_write(attr_dims, o->under_vol_id, H5T_NATIVE_HSIZE, h5dims, dxpl_id, NULL);
                H5VLattr_close(attr_dims, o->under_vol_id, dxpl_id, NULL);
            }

            int p_dt = (int)pressio_dt;
            void *attr_dt = H5VLattr_create(under, &attr_loc, o->under_vol_id,
                "_VOL_ORIG_TYPE", H5T_NATIVE_INT, scalar_space,
                acpl_id, aapl_id, dxpl_id, NULL);
            if (attr_dt) {
                H5VLattr_write(attr_dt, o->under_vol_id, H5T_NATIVE_INT, &p_dt, dxpl_id, NULL);
                H5VLattr_close(attr_dt, o->under_vol_id, dxpl_id, NULL);
            }

            char comp_name[64] = "";
            if (H5Pexist(dcpl_id, "pressio:compressor") > 0)
                H5Pget(dcpl_id, "pressio:compressor", comp_name);
            else
                strncpy(comp_name, config_ctx->default_compression_id, sizeof(comp_name)-1);

            hid_t str_type = H5Tcopy(H5T_C_S1);
            H5Tset_size(str_type, 64);
            void *attr_comp = H5VLattr_create(under, &attr_loc, o->under_vol_id,
                "_VOL_COMPRESSOR", str_type, scalar_space,
                acpl_id, aapl_id, dxpl_id, NULL);
            if (attr_comp) {
                H5VLattr_write(attr_comp, o->under_vol_id, str_type, comp_name, dxpl_id, NULL);
                H5VLattr_close(attr_comp, o->under_vol_id, dxpl_id, NULL);
            }

            H5Tclose(str_type);
            H5Pclose(acpl_id);
            H5Pclose(aapl_id);
            H5Sclose(scalar_space);
            H5Sclose(dim_space);

            /* Store original metadata in the context so write/read know what to do */
            dset->custom_data = gpu_vol_dataset_wrap(under, rank, h5dims, type_id, pressio_dt, dcpl_id, o->under_vol_id, file_ctx, NULL);

            gpu_vol_dataset_t *ds_ctx = (gpu_vol_dataset_t*)dset->custom_data;

            // add in dataset label for timing output
            if (ds_ctx && ds_ctx->comp_ctx && name) {
                strncpy(ds_ctx->comp_ctx->dataset_name, name,
                        sizeof(ds_ctx->comp_ctx->dataset_name) - 1);
                ds_ctx->comp_ctx->dataset_name[sizeof(ds_ctx->comp_ctx->dataset_name) - 1] = '\0';
            }

            /* Flag whether compression was explicitly requested */
            if (ds_ctx) {
                ds_ctx->compression_requested =
                    (comp_name[0] != '\0' && strcmp(comp_name, "noop") != 0) ? 1 : 0;
            }

            if (ds_ctx && ds_ctx->compression_requested && !ds_ctx->comp_ctx) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_compressor_unavail,
                        "dataset '%s' creation aborted: compressor '%s' could not be initialized",
                        name, comp_name);
                        
                H5VLdataset_close(under, o->under_vol_id, dxpl_id, NULL);
                H5VL_pass_through_ext_free_obj(dset);
                gpu_vol_dataset_destroy(ds_ctx);
                free(h5dims);
                
                H5Sclose(underlying_space_id);
                H5Pclose(underlying_dcpl_id);
                return NULL;
            }

            /* Snapshot compressor options as a hidden attribute for replay at dataset_open */
            if (ds_ctx && ds_ctx->comp_ctx && ds_ctx->comp_ctx->compressor) {
                struct pressio_options *opts = pressio_compressor_get_options(ds_ctx->comp_ctx->compressor);
                if (opts) {
                    /* get_options won't include vol:* keys (the compressor ignored them);
                    * re-inject from the ctx so the mode survives into _VOL_OPTIONS_JSON */
                    if (ds_ctx->comp_ctx->chunking_mode == VOL_CHUNKING_VOL)
                        pressio_options_set_string(opts, "vol:chunking_mode", "vol");
                    else if (ds_ctx->comp_ctx->chunking_mode == VOL_CHUNKING_PRESSIO)
                        pressio_options_set_string(opts, "vol:chunking_mode", "pressio");
                    else if (ds_ctx->comp_ctx->chunking_mode == VOL_CHUNKING_SHARED)
                        pressio_options_set_string(opts, "vol:chunking_mode", "shared");
                    if (ds_ctx->comp_ctx->chunk_n > 0)
                        pressio_options_set_uinteger64(opts, "vol:chunk_n", ds_ctx->comp_ctx->chunk_n);
                    char *json = pressio_options_to_json(ds_ctx->comp_ctx->library, opts);
                    if (json) {
                        hid_t json_str_type = H5Tcopy(H5T_C_S1);
                        H5Tset_size(json_str_type, strlen(json) + 1);
                        H5Tset_strpad(json_str_type, H5T_STR_NULLTERM);
                        hid_t json_space = H5Screate(H5S_SCALAR);
                        hid_t json_acpl = H5Pcreate(H5P_ATTRIBUTE_CREATE);
                        hid_t json_aapl = H5Pcreate(H5P_ATTRIBUTE_ACCESS);
                        void *attr_json = H5VLattr_create(under, &attr_loc, o->under_vol_id,
                            "_VOL_OPTIONS_JSON", json_str_type, json_space,
                            json_acpl, json_aapl, dxpl_id, NULL);
                        if (attr_json) {
                            H5VLattr_write(attr_json, o->under_vol_id,
                                        json_str_type, json, dxpl_id, NULL);
                            H5VLattr_close(attr_json, o->under_vol_id, dxpl_id, NULL);
                        }
                        H5Pclose(json_acpl);
                        H5Pclose(json_aapl);
                        H5Sclose(json_space);
                        H5Tclose(json_str_type);
                        free(json);
                    }
                    pressio_options_free(opts);
                }
            }

            H5Sclose(underlying_space_id);
            H5Pclose(underlying_dcpl_id);
        } else {
            dset->custom_data = NULL;
        }

        if (req && *req) {
            *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);
        }
    } else {
        dset = NULL;
    }

    free(h5dims);
    return (void *)dset;
} /* end H5VL_pass_through_ext_dataset_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_dataset_open
 *
 * Purpose:     Opens a dataset in a container
 *
 * Return:      Success:    Pointer to a dataset object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_dataset_open(void *obj,
    const H5VL_loc_params_t *loc_params,
    const char *name,
    hid_t dapl_id,
    hid_t dxpl_id,
    void **req)
{
    H5VL_pass_through_ext_t *dset;
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    void *under;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL DATASET Open\n");
#endif

    under = H5VLdataset_open(
        o->under_object, loc_params, o->under_vol_id, name,
        dapl_id, dxpl_id, req
    );

    if (!under)
        return NULL;

    dset = H5VL_pass_through_ext_new_obj(under, o->under_vol_id);
    gpu_vol_file_t *file_ctx = (gpu_vol_file_t*)o->custom_data;
    config_params *config_ctx = file_ctx ? file_ctx->config_params : NULL;

    if (config_ctx) {
        /* 1. Get underlying dataset's DCPL handle */
        H5VL_dataset_get_args_t get_args;
        get_args.op_type = H5VL_DATASET_GET_DCPL;
        H5VLdataset_get(under, o->under_vol_id, &get_args, dapl_id, NULL);
        hid_t real_dcpl_id = get_args.args.get_dcpl.dcpl_id;

        /* Read Metadata */
        H5VL_loc_params_t attr_loc;
        attr_loc.type = H5VL_OBJECT_BY_SELF;
        attr_loc.obj_type = H5I_DATASET;

        int recovered_rank = 0;
        hsize_t *recovered_dims = NULL;
        int recovered_dt = 0;

        hid_t aapl_id = H5Pcreate(H5P_ATTRIBUTE_ACCESS);

        void *attr_rank = H5VLattr_open(under, &attr_loc, o->under_vol_id, "_VOL_ORIG_RANK", aapl_id, dxpl_id, NULL);
        if (attr_rank) {
            H5VLattr_read(attr_rank, o->under_vol_id, H5T_NATIVE_INT, &recovered_rank, dxpl_id, NULL);
            H5VLattr_close(attr_rank, o->under_vol_id, dxpl_id, NULL);
        }

        if (recovered_rank > 0) {
            recovered_dims = (hsize_t *)malloc(recovered_rank * sizeof(hsize_t));
            void *attr_dims = H5VLattr_open(under, &attr_loc, o->under_vol_id, "_VOL_ORIG_DIMS", aapl_id, dxpl_id, NULL);
            if (attr_dims) {
                H5VLattr_read(attr_dims, o->under_vol_id, H5T_NATIVE_HSIZE, recovered_dims, dxpl_id, NULL);
                H5VLattr_close(attr_dims, o->under_vol_id, dxpl_id, NULL);
            }
        }

        void *attr_dt = H5VLattr_open(under, &attr_loc, o->under_vol_id, "_VOL_ORIG_TYPE", aapl_id, dxpl_id, NULL);
        if (attr_dt) {
            H5VLattr_read(attr_dt, o->under_vol_id, H5T_NATIVE_INT, &recovered_dt, dxpl_id, NULL);
            H5VLattr_close(attr_dt, o->under_vol_id, dxpl_id, NULL);
        }

        enum pressio_dtype real_pressio_dt = (enum pressio_dtype)recovered_dt;

        get_args.op_type = H5VL_DATASET_GET_TYPE;
        get_args.args.get_type.type_id = H5I_INVALID_HID;
        H5VLdataset_get(under, o->under_vol_id, &get_args, dapl_id, NULL);
        hid_t real_type_id = get_args.args.get_type.type_id;

        char recovered_comp[64] = "";
        hid_t str_type = H5Tcopy(H5T_C_S1);
        H5Tset_size(str_type, 64);
        void *attr_comp = H5VLattr_open(under, &attr_loc, o->under_vol_id,
            "_VOL_COMPRESSOR", aapl_id, dxpl_id, NULL);
        if (attr_comp) {
            H5VLattr_read(attr_comp, o->under_vol_id, str_type, recovered_comp, dxpl_id, NULL);
            H5VLattr_close(attr_comp, o->under_vol_id, dxpl_id, NULL);
        }
        H5Tclose(str_type);
        H5Pclose(aapl_id);

        /* Read back the options JSON snapshot written at dataset_create */
        char *json_buf = NULL;
        hid_t json_aapl = H5Pcreate(H5P_ATTRIBUTE_ACCESS);
        void *attr_json = H5VLattr_open(under, &attr_loc, o->under_vol_id,
            "_VOL_OPTIONS_JSON", json_aapl, dxpl_id, NULL);
        if (attr_json) {
            H5VL_attr_get_args_t ag;
            ag.op_type = H5VL_ATTR_GET_TYPE;
            ag.args.get_type.type_id = H5I_INVALID_HID;
            H5VLattr_get(attr_json, o->under_vol_id, &ag, dxpl_id, NULL);
            hid_t json_type = ag.args.get_type.type_id;
            size_t jlen = H5Tget_size(json_type);
            json_buf = malloc(jlen);
            H5VLattr_read(attr_json, o->under_vol_id, json_type, json_buf, dxpl_id, NULL);
            H5Tclose(json_type);
            H5VLattr_close(attr_json, o->under_vol_id, dxpl_id, NULL);
        }
        H5Pclose(json_aapl);

        dset->custom_data = gpu_vol_dataset_wrap(under, recovered_rank, recovered_dims,
                                          real_type_id, real_pressio_dt,
                                          real_dcpl_id, o->under_vol_id, file_ctx,
                                          recovered_comp);

        /* Replay options into the compressor handle, then re-apply the CUDA stream */
        gpu_vol_dataset_t *ds_ctx = (gpu_vol_dataset_t*)dset->custom_data;

        if (ds_ctx && ds_ctx->comp_ctx && name) {
            strncpy(ds_ctx->comp_ctx->dataset_name, name,
                    sizeof(ds_ctx->comp_ctx->dataset_name) - 1);
            ds_ctx->comp_ctx->dataset_name[sizeof(ds_ctx->comp_ctx->dataset_name) - 1] = '\0';
        }

        if (json_buf && ds_ctx && ds_ctx->comp_ctx && ds_ctx->comp_ctx->compressor) {
            struct pressio_options *opts = pressio_options_new_json(ds_ctx->comp_ctx->library, json_buf);
            if (opts) {
                H5VL_pass_through_ext_parse_chunking_opts(ds_ctx->comp_ctx, opts);
                pressio_compressor_set_options(ds_ctx->comp_ctx->compressor, opts);
                pressio_options_free(opts);
            }
            json_buf = NULL;
        }
        free(json_buf);

        H5Pclose(real_dcpl_id);
        if (recovered_dims) free(recovered_dims);
    } else {
        dset->custom_data = NULL;
    }

    if (req && *req) {
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);
    }

    return (void *)dset;
} /* end H5VL_pass_through_ext_dataset_open() */

/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_dataset_read
 *
 * Purpose:     Reads data elements from a dataset into a buffer.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
typedef struct {
    H5VL_pass_through_ext_t *d;
    void                    *under;
    compression_ctx         *ctx;
    hid_t                    plist_id;
    hid_t                    file_space_id;   /* for spatial planning     */
    const unsigned char     *cbuf;            /* fetched container        */
    size_t                   cont_bytes;
    void                    *dst;             /* == ctx->decomp_buf       */
    size_t                   total_bytes;     /* logical size             */
    int                      want_layers;     /* progressive, 0 = all     */
    const char              *dset_name;
    size_t                   u;
} vol_read_req_t;

typedef struct { double io_ms, decomp_ms; } vol_read_timing_t;

/* =========================================================================
 * PATH 1 -- NATIVE:  [magic][csize][payload]
 * ========================================================================= */
static herr_t
vol_read_native(const vol_read_req_t *req, vol_read_timing_t *t)
{
    const size_t hdr_bytes = 2 * sizeof(uint64_t);
    uint64_t csize;

    if (req->cont_bytes < hdr_bytes) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "truncated native header for dataset %zu", req->u);
        return -1;
    }
    memcpy(&csize, req->cbuf + sizeof(uint64_t), sizeof(uint64_t));

    if (csize == 0 || hdr_bytes + csize > req->cont_bytes) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "native payload [%zu,%llu) exceeds container size %zu for "
                "dataset %zu (container corrupt?)",
                hdr_bytes, (unsigned long long)(hdr_bytes + csize),
                req->cont_bytes, req->u);
        return -1;
    }

    double _c0 = bench_now_ms();
    herr_t rc = H5VL_pass_through_ext_decompress_native(
                    req->ctx, req->cbuf + hdr_bytes, (size_t)csize,
                    req->dst, req->total_bytes);
    t->decomp_ms += bench_now_ms() - _c0;
    return rc;
}

/* =========================================================================
 * PATH 2 -- PRESSIO:  [magic][csize][chunk_elems][payload]
 *
 * One self-framed blob from libpressio's 'chunking' meta-compressor.
 * ========================================================================= */
static herr_t
vol_read_pressio(const vol_read_req_t *req, vol_read_timing_t *t)
{
    const size_t hdr_bytes = VOL_PRESSIO_HDR_WORDS * sizeof(uint64_t);
    uint64_t csize, chunk_elems;

    if (req->cont_bytes < hdr_bytes) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "truncated pressio-chunked header for dataset %zu", req->u);
        return -1;
    }
    memcpy(&csize,       req->cbuf + 8,  sizeof(uint64_t));
    memcpy(&chunk_elems, req->cbuf + 16, sizeof(uint64_t));

    if (chunk_elems == 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "corrupt pressio-chunked header for dataset %zu "
                "(chunk_elems=0)", req->u);
        return -1;
    }
    if (csize == 0 || hdr_bytes + csize > req->cont_bytes) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "pressio payload exceeds container size %zu for dataset %zu",
                req->cont_bytes, req->u);
        return -1;
    }

    double _c0 = bench_now_ms();
    herr_t rc = H5VL_pass_through_ext_decompress_pressio(
                    req->ctx, req->cbuf + hdr_bytes, (size_t)csize,
                    chunk_elems, req->dst, req->total_bytes);
    t->decomp_ms += bench_now_ms() - _c0;
    return rc;
}

/* =========================================================================
 * PATH 3 -- VOL-LEVEL CHUNKED:
 *   [magic][nchunks][chunk_bytes][csize table][payloads]
 * ========================================================================= */
static herr_t
vol_read_vol(const vol_read_req_t *req, vol_read_timing_t *t)
{
    uint64_t nc, cbytes;

    if (req->cont_bytes < VOL_CHUNK_HDR_WORDS * sizeof(uint64_t)) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "truncated chunk header for dataset %zu", req->u);
        return -1;
    }
    memcpy(&nc,     req->cbuf + 8,  sizeof(uint64_t));
    memcpy(&cbytes, req->cbuf + 16, sizeof(uint64_t));

    const size_t nchunks     = (size_t)nc;
    const size_t chunk_bytes = (size_t)cbytes;
    const size_t payload_off =
        (VOL_CHUNK_HDR_WORDS + nchunks) * sizeof(uint64_t);
    const unsigned char *table =
        req->cbuf + VOL_CHUNK_HDR_WORDS * sizeof(uint64_t);

    if (nchunks == 0 || chunk_bytes == 0 ||
        payload_off > req->cont_bytes ||
        (nchunks - 1) * chunk_bytes >= req->total_bytes ||
        nchunks * chunk_bytes < req->total_bytes) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "inconsistent chunk header for dataset %zu "
                "(nchunks=%zu chunk_bytes=%zu total=%zu)",
                req->u, nchunks, chunk_bytes, req->total_bytes);
        return -1;
    }

    double _c0 = bench_now_ms();
    herr_t rc  = 0;
    size_t poff = payload_off;

    for (size_t k = 0; k < nchunks && rc >= 0; k++) {
        uint64_t csize_k;
        memcpy(&csize_k, table + k * sizeof(uint64_t), sizeof(uint64_t));

        if (csize_k == 0 || poff + (size_t)csize_k > req->cont_bytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "chunk %zu payload [%zu,%zu) exceeds container size %zu "
                    "for dataset %zu (container corrupt?)",
                    k, poff, poff + (size_t)csize_k, req->cont_bytes, req->u);
            rc = -1;
            break;
        }

        size_t doff = k * chunk_bytes;
        size_t dlen = (req->total_bytes - doff < chunk_bytes)
                          ? (req->total_bytes - doff) : chunk_bytes;

        rc = H5VL_pass_through_ext_transfer_decompress_chunk(
                 req->ctx, req->cbuf + poff, (size_t)csize_k,
                 (char *)req->dst + doff, dlen);

        poff += (size_t)csize_k;
    }
    t->decomp_ms += bench_now_ms() - _c0;
    return rc;
}

/* =========================================================================
 * PATH 4 -- SHARED METADATA
 * ========================================================================= */
static herr_t
vol_read_shared(const vol_read_req_t *req, vol_read_timing_t *t)
{
    double _c0 = bench_now_ms();
    herr_t rc = H5VL_pass_through_ext_decompress_shared(
                    req->ctx, req->cbuf, req->cont_bytes,
                    req->dst, req->total_bytes);
    t->decomp_ms += bench_now_ms() - _c0;
    return rc;
}

/* =========================================================================
 * PATH 5 -- PROGRESSIVE
 *
 * Decodes and sums the first `want_layers` residual layers. The value MUST be
 * the same one the read plan used: if the plan fetched only n layers and the
 * decoder is asked for more, it would sum zeroed memory and silently return a
 * wrong field. The dispatcher computes it once and passes it to both.
 * ========================================================================= */
static herr_t
vol_read_progressive(const vol_read_req_t *req, vol_read_timing_t *t)
{
    double achieved = 0.0;

    double _c0 = bench_now_ms();
    herr_t rc = H5VL_pass_through_ext_decompress_progressive(
                    req->ctx, req->cbuf, req->cont_bytes,
                    req->dst, req->total_bytes,
                    req->want_layers, &achieved);
    t->decomp_ms += bench_now_ms() - _c0;

    if (rc >= 0 && req->want_layers > 0)
        fprintf(stderr, "[VOL] dataset '%s' read at REDUCED fidelity: "
                        "%d layer(s), bound ~%.3e\n",
                req->dset_name, req->want_layers, achieved);
    return rc;
}

/* =========================================================================
 * PATH 0 -- NO COMPRESSION
 * ========================================================================= */
static herr_t
vol_read_passthrough(H5VL_pass_through_ext_t *d, void *under,
                     hid_t mem_type_id, hid_t mem_space_id,
                     hid_t file_space_id, hid_t plist_id, void *buf,
                     size_t u, vol_read_timing_t *t)
{
    void *rbufs[] = { buf };
    double _io0 = bench_now_ms();
    herr_t rc = H5VLdataset_read(1, &under, d->under_vol_id, &mem_type_id,
                                 &mem_space_id, &file_space_id, plist_id,
                                 rbufs, NULL);
    t->io_ms += bench_now_ms() - _io0;

    if (rc < 0)
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_decompress_failed,
                "passthrough dataset_read failed for dataset %zu", u);
    return rc;
}

/* =========================================================================
 * SERVE -- hand this read's bytes out of the decompressed buffer.
 *
 * KNOWN LIMITATION, PRESERVED AS-IS: this uses mem_space only for the SIZE
 * and a running cursor for the POSITION -- file_space is not consulted. A
 * hyperslab read of the middle of the array therefore returns the FIRST bytes,
 * and a re-read after the cursor is exhausted returns nothing, both silently.
 * Correct for the one-H5S_ALL-read-per-open pattern the benchmarks use.
 * Fix this before claiming partial spatial reads work end to end.
 * ========================================================================= */
static herr_t
vol_read_serve(compression_ctx *ctx, hid_t mem_space_id, void *dst, size_t u)
{
    size_t want;

    if (mem_space_id == H5S_ALL) {
        want = ctx->decomp_size - ctx->read_served;
    } else {
        hssize_t sel_pts = H5Sget_select_npoints(mem_space_id);
        if (sel_pts < 0) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_decompress_failed,
                    "could not query memory selection for dataset %zu", u);
            return -1;
        }
        want = (size_t)sel_pts * pressio_dtype_size(ctx->dtype);
    }

    if (ctx->read_served >= ctx->decomp_size) want = 0;
    else if (ctx->read_served + want > ctx->decomp_size)
        want = ctx->decomp_size - ctx->read_served;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    fprintf(stderr, "SERVE: want=%zu served=%zu/%zu dest=%p\n",
            want, ctx->read_served, ctx->decomp_size, dst);
    fflush(stderr);
#endif

    if (want > 0) {
        memcpy(dst, (char *)ctx->decomp_buf + ctx->read_served, want);
        ctx->read_served += want;
    }
    return 0;
}

/* =========================================================================
 * DISPATCHER
 * ========================================================================= */
static herr_t
H5VL_pass_through_ext_dataset_read(
    size_t count, void *dset[], hid_t mem_type_id[], hid_t mem_space_id[],
    hid_t file_space_id[], hid_t plist_id, void *buf[], void **req_out)
{
    herr_t ret_val = 0;

    for (size_t u = 0; u < count; u++) {
        vol_read_timing_t t = {0.0, 0.0};
        double _d0 = bench_now_ms();

        H5VL_pass_through_ext_t *d = (H5VL_pass_through_ext_t *)dset[u];
        void *under = d->under_object;
        gpu_vol_dataset_t *ds_ctx = (gpu_vol_dataset_t *)d->custom_data;
        compression_ctx   *ctx    = ds_ctx ? ds_ctx->comp_ctx : NULL;
        const char *dset_name =
            (ctx && ctx->dataset_name[0]) ? ctx->dataset_name : "none";

        /* ---------------- no compression ---------------- */
        if (!ctx) {
            if (ds_ctx && ds_ctx->compression_requested) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_compressor_unavail,
                        "compression was requested but no compression context "
                        "is available -- refusing to fall through to "
                        "uncompressed storage");
                ret_val = -1;
                continue;
            }
            if (vol_read_passthrough(d, under, mem_type_id[u], mem_space_id[u],
                                     file_space_id[u], plist_id, buf[u],
                                     u, &t) < 0)
                ret_val = -1;

            vol_timing_emit(dset_name, "none", "read",
                            bench_now_ms() - _d0, 0.0, t.io_ms);
            continue;
        }

        const size_t total_bytes = vol_logical_nbytes(ctx);

        /* ---------------- load + decode, once per open ---------------- */
        if (!ctx->decomp_buf) {
            vol_read_plan_t plan;
            unsigned char  *cbuf = NULL;
            const int want_layers =
                H5VL_pass_through_ext_progressive_want(plist_id);

            /* PHASE 1: header + directory, then decide which byte ranges this
             * request actually needs. PHASE 2: read only those. */
            double _io0 = bench_now_ms();
            if (vol_container_plan(under, d->under_vol_id, plist_id, ctx,
                                   file_space_id[u], want_layers, &plan) < 0) {
                t.io_ms += bench_now_ms() - _io0;
                ret_val = -1;
                continue;
            }
            if (vol_container_fetch(under, d->under_vol_id, plist_id,
                                    &plan, &cbuf) < 0) {
                t.io_ms += bench_now_ms() - _io0;
                vol_read_plan_free(&plan);
                ret_val = -1;
                continue;
            }
            t.io_ms += bench_now_ms() - _io0;

            if (plan.partial)
                fprintf(stderr, "[VOL] partial container read: %llu of %llu "
                        "bytes (%.1f%%)\n",
                        (unsigned long long)plan.bytes_needed,
                        (unsigned long long)plan.cont_bytes,
                        100.0 * (double)plan.bytes_needed /
                                (double)plan.cont_bytes);

            ctx->decomp_buf = malloc(total_bytes);
            if (!ctx->decomp_buf) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_decompress_failed,
                        "out of memory allocating %zu byte decompress buffer "
                        "for dataset %zu", total_bytes, u);
                free(cbuf);
                vol_read_plan_free(&plan);
                ret_val = -1;
                continue;
            }

            vol_read_req_t rreq;
            rreq.d             = d;
            rreq.under         = under;
            rreq.ctx           = ctx;
            rreq.plist_id      = plist_id;
            rreq.file_space_id = file_space_id[u];
            rreq.cbuf          = cbuf;
            rreq.cont_bytes    = (size_t)plan.cont_bytes;
            rreq.dst           = ctx->decomp_buf;
            rreq.total_bytes   = total_bytes;
            rreq.want_layers   = want_layers;
            rreq.dset_name     = dset_name;
            rreq.u             = u;

            herr_t rc;
            switch (plan.kind) {
            case VOL_CONT_PRESSIO:     rc = vol_read_pressio(&rreq, &t);     break;
            case VOL_CONT_CHUNKED:     rc = vol_read_vol(&rreq, &t);         break;
            case VOL_CONT_SHARED:      rc = vol_read_shared(&rreq, &t);      break;
            case VOL_CONT_PROGRESSIVE: rc = vol_read_progressive(&rreq, &t); break;
            case VOL_CONT_NATIVE:      rc = vol_read_native(&rreq, &t);      break;
            default:
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_decompress_failed,
                        "unrecognized container magic 0x%016llx for dataset "
                        "%zu (file written by an incompatible VOL version?)",
                        (unsigned long long)plan.magic, u);
                rc = -1;
                break;
            }

            free(cbuf);
            vol_read_plan_free(&plan);

            if (rc < 0) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_decompress_failed,
                        "decompression failed for dataset %zu compressor='%s'",
                        u, ctx->compressor_id);
                free(ctx->decomp_buf);
                ctx->decomp_buf = NULL;
                ret_val = -1;
                continue;
            }

            ctx->decomp_size = total_bytes;
            ctx->read_served = 0;
        }

        /* ---------------- serve ---------------- */
        if (vol_read_serve(ctx, mem_space_id[u], buf[u], u) < 0) {
            ret_val = -1;
            continue;
        }

        vol_timing_emit(dset_name, ctx->compressor_id, "read",
                        bench_now_ms() - _d0, t.decomp_ms, t.io_ms);
    }

    return ret_val;
} /* end H5VL_pass_through_ext_dataset_read() */

/* =========================================================================
 *   H5VL_pass_through_ext_dataset_write   dispatcher: resolves the context,
 *                                         locates the write in the logical
 *                                         array, stages partial strips, then
 *                                         switches on the chunking mode
 *     vol_write_native                    VOL_CHUNKING_NONE
 *     vol_write_pressio                   VOL_CHUNKING_PRESSIO
 *     vol_write_vol                       VOL_CHUNKING_VOL
 *     vol_write_shared                    VOL_CHUNKING_SHARED
 *     vol_write_progressive               VOL_CHUNKING_PROGRESSIVE
 * ========================================================================= */

/* -------------------------------------------------------------------------
 * Per-write request, resolved once by the dispatcher.
 * ---------------------------------------------------------------------- */
typedef struct {
    H5VL_pass_through_ext_t *d;
    void                    *under;
    compression_ctx         *ctx;
    hid_t                    plist_id;
    const void              *comp_src;     /* whole logical buffer          */
    size_t                   total_bytes;  /* logical size                  */
    size_t                   dsize;        /* element size                  */
    const char              *dset_name;
    size_t                   u;            /* index, for error messages     */
} vol_write_req_t;

/* Resize the underlying 1-D byte extent to hold `total` bytes. */
static herr_t
vol_write_set_extent(const vol_write_req_t *req, hsize_t total,
                     vol_write_timing_t *t)
{
    hsize_t new_size[H5S_MAX_RANK] = {0};
    H5VL_dataset_specific_args_t sargs;

    new_size[0] = total;
    sargs.op_type = H5VL_DATASET_SET_EXTENT;
    sargs.args.set_extent.size = new_size;

    double _x0 = bench_now_ms();
    herr_t rc = H5VLdataset_specific(req->under, req->d->under_vol_id, &sargs,
                                     req->plist_id, NULL);
    t->container_ms += bench_now_ms() - _x0;

    if (rc < 0)
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "failed to resize dataset %zu to %llu bytes",
                req->u, (unsigned long long)total);
    return rc;
}

/* One write covering the whole resized extent. H5S_ALL is exact here, so no
 * dataspace objects are needed at all. */
static herr_t
vol_write_whole(const vol_write_req_t *req, const void *blob, hsize_t total,
                vol_write_timing_t *t)
{
    if (vol_write_set_extent(req, total, t) < 0) return -1;

    const void *wbufs[] = { blob };
    hid_t mspace = H5S_ALL, fspace = H5S_ALL;

    double _io0 = bench_now_ms();
    herr_t rc = H5VLdataset_write(1, &req->under, req->d->under_vol_id,
                                  (hid_t[]){H5T_NATIVE_UCHAR},
                                  &mspace, &fspace, req->plist_id, wbufs, NULL);
    t->io_ms += bench_now_ms() - _io0;

    if (rc < 0)
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "container write failed for dataset %zu", req->u);
    return rc;
}

/* Write `len` bytes at byte offset `off` within a container of `total` bytes. */
static herr_t
vol_write_at(const vol_write_req_t *req, const void *buf,
             hsize_t off, hsize_t len, hsize_t total, vol_write_timing_t *t)
{
    if (len == 0) return 0;

    double _p0 = bench_now_ms();
    hid_t mspace = H5Screate_simple(1, &len, NULL);
    hid_t fspace = H5Screate_simple(1, &total, NULL);
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &off, NULL, &len, NULL);
    t->container_ms += bench_now_ms() - _p0;

    const void *bufs[] = { buf };
    double _io0 = bench_now_ms();
    herr_t rc = H5VLdataset_write(1, &req->under, req->d->under_vol_id,
                                  (hid_t[]){H5T_NATIVE_UCHAR},
                                  &mspace, &fspace, req->plist_id, bufs, NULL);
    t->io_ms += bench_now_ms() - _io0;

    double _p1 = bench_now_ms();
    H5Sclose(mspace);
    H5Sclose(fspace);
    t->container_ms += bench_now_ms() - _p1;

    if (rc < 0)
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "write at offset %llu failed for dataset %zu",
                (unsigned long long)off, req->u);
    return rc;
}

/* Reset the accumulators the compress helpers add into. */
static void
vol_write_reset_timers(compression_ctx *ctx, vol_write_timing_t *t)
{
    ctx->device_ms       = 0.0;
    ctx->pressio_call_ms = 0.0;
    (void)t;
}

static void
vol_write_capture_timers(compression_ctx *ctx, vol_write_timing_t *t,
                         double c0)
{
    t->compress_ms     = bench_now_ms() - c0;
    t->pressio_call_ms = ctx->pressio_call_ms;
    t->device_ms       = ctx->device_ms;
}

/* =========================================================================
 * PATH 1 -- NATIVE (default)
 *
 * One compress call into a single self-describing blob, then ONE write of
 * [VOL_NATIVE_MAGIC][csize][payload]. Whatever chunking happens is the
 * codec's own internal blocking; the VOL imposes none.
 * ========================================================================= */
static herr_t
vol_write_native(const vol_write_req_t *req, vol_write_timing_t *t)
{
    compression_ctx *ctx = req->ctx;
    const size_t hdr_bytes = 2 * sizeof(uint64_t);   /* magic + csize */
    void    *blob = NULL;
    uint64_t clen = 0;
    herr_t   rc;

    vol_write_reset_timers(ctx, t);
    double _c0 = bench_now_ms();
    rc = H5VL_pass_through_ext_compress_native(ctx, req->comp_src,
                                               req->total_bytes, hdr_bytes,
                                               &blob, &clen);
    vol_write_capture_timers(ctx, t, _c0);

    if (rc < 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "native compression failed for dataset %zu compressor='%s'",
                req->u, ctx->compressor_id);
        free(blob);
        return -1;
    }

    {
        double _h0 = bench_now_ms();
        uint64_t *hdr = (uint64_t *)blob;
        hdr[0] = VOL_NATIVE_MAGIC;
        hdr[1] = clen;
        t->container_ms += bench_now_ms() - _h0;
    }

    rc = vol_write_whole(req, blob, (hsize_t)(hdr_bytes + clen), t);
    free(blob);
    return rc;
}

/* =========================================================================
 * PATH 2 -- PRESSIO
 *
 * One compress through libpressio's 'chunking' meta-compressor, producing a
 * single self-framed blob. Its internal chunk table lives inside the payload,
 * so the container only records the size and the chunk geometry needed to
 * rebuild the same wrapper at read time:
 *   [VOL_PRESSIO_MAGIC][csize][chunk_elems][payload]
 *
 * The header cannot be fused with the payload here (libpressio owns the
 * output allocation), so this is header write + payload write.
 * ========================================================================= */
static herr_t
vol_write_pressio(const vol_write_req_t *req, vol_write_timing_t *t)
{
    compression_ctx *ctx = req->ctx;
    uint64_t phdr[VOL_PRESSIO_HDR_WORDS];
    const size_t hdr_bytes = sizeof(phdr);
    void    *blob        = NULL;
    uint64_t clen        = 0;
    uint64_t chunk_elems = 0;
    herr_t   rc;

    const size_t chunk_bytes_req =
        H5VL_pass_through_ext_chunk_bytes(ctx, req->total_bytes, req->dsize);

    vol_write_reset_timers(ctx, t);
    double _c0 = bench_now_ms();
    rc = H5VL_pass_through_ext_compress_pressio(ctx, req->comp_src,
                                                req->total_bytes,
                                                chunk_bytes_req,
                                                &blob, &clen, &chunk_elems);
    vol_write_capture_timers(ctx, t, _c0);

    if (rc < 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "pressio-chunked compression failed for dataset %zu "
                "compressor='%s'", req->u, ctx->compressor_id);
        free(blob);
        return -1;
    }

    const hsize_t total = (hsize_t)(hdr_bytes + clen);

    if (vol_write_set_extent(req, total, t) < 0) { free(blob); return -1; }

    double _h0 = bench_now_ms();
    phdr[0] = VOL_PRESSIO_MAGIC;
    phdr[1] = clen;
    phdr[2] = chunk_elems;
    t->container_ms += bench_now_ms() - _h0;

    if (vol_write_at(req, phdr, 0, (hsize_t)hdr_bytes, total, t) < 0) {
        free(blob);
        return -1;
    }
    rc = vol_write_at(req, blob, (hsize_t)hdr_bytes, (hsize_t)clen, total, t);

    free(blob);
    return rc;
}

/* =========================================================================
 * PATH 3 -- VOL-LEVEL CHUNKING
 *
 * The VOL slices the buffer and compresses each slice independently, then
 * writes [magic][nchunks][chunk_bytes][csize table][payloads].
 *
 * The header cannot be fused with the payloads: the csize table is not known
 * until every chunk has been compressed. So this is one header write followed
 * by N payload writes.
 * ========================================================================= */
static herr_t
vol_write_vol(const vol_write_req_t *req, vol_write_timing_t *t)
{
    compression_ctx *ctx = req->ctx;
    const size_t chunk_bytes =
        H5VL_pass_through_ext_chunk_bytes(ctx, req->total_bytes, req->dsize);
    const size_t nchunks    = (req->total_bytes + chunk_bytes - 1) / chunk_bytes;
    const size_t tail_bytes = req->total_bytes - (nchunks - 1) * chunk_bytes;

    void    **chunk_bufs = NULL;
    uint64_t *csizes     = NULL;
    uint64_t *hdr        = NULL;
    herr_t    rc         = 0;

    /* RAGGED TAIL IS ALWAYS REPORTED. */
    if (tail_bytes != chunk_bytes) {
        fprintf(stderr,
            "[VOL RAGGED] dataset %zu '%s': %zu chunks of %zu B + a FINAL "
            "chunk of %zu B (%.1f%% of a full chunk). total=%zu elem=%zu. "
            "The last chunk has a different length from the rest -- rule it "
            "out before blaming the codec.\n",
            req->u, ctx->compressor_id, nchunks - 1, chunk_bytes, tail_bytes,
            100.0 * (double)tail_bytes / (double)chunk_bytes,
            req->total_bytes, req->dsize);
    } else if (getenv("VOL_COMP_CHUNK_LOG")) {
        fprintf(stderr,
            "[VOL RAGGED] dataset %zu '%s': clean split, %zu x %zu B, "
            "no ragged tail\n", req->u, ctx->compressor_id, nchunks, chunk_bytes);
    }

    chunk_bufs = (void **)calloc(nchunks, sizeof(void *));
    csizes     = (uint64_t *)malloc(nchunks * sizeof(uint64_t));
    if (!chunk_bufs || !csizes) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "out of memory allocating chunk table (%zu chunks) for "
                "dataset %zu", nchunks, req->u);
        free(chunk_bufs); free(csizes);
        return -1;
    }

    vol_write_reset_timers(ctx, t);
    double _c0 = bench_now_ms();
    for (size_t k = 0; k < nchunks && rc >= 0; k++) {
        size_t coff = k * chunk_bytes;
        size_t clen = (req->total_bytes - coff < chunk_bytes)
                          ? (req->total_bytes - coff) : chunk_bytes;
        rc = H5VL_pass_through_ext_transfer_compress_chunk(
                 ctx, (const char *)req->comp_src + coff, clen,
                 &chunk_bufs[k], &csizes[k]);
    }
    vol_write_capture_timers(ctx, t, _c0);

    if (rc < 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "chunked compression failed for dataset %zu compressor='%s'",
                req->u, ctx->compressor_id);
        goto done;
    }

    {
        const size_t hdr_words = VOL_CHUNK_HDR_WORDS + nchunks;
        const size_t hdr_bytes = hdr_words * sizeof(uint64_t);
        uint64_t payload_bytes = 0;
        for (size_t k = 0; k < nchunks; k++) payload_bytes += csizes[k];
        const hsize_t total = (hsize_t)(hdr_bytes + payload_bytes);

        if (vol_write_set_extent(req, total, t) < 0) { rc = -1; goto done; }

        double _h0 = bench_now_ms();
        hdr = (uint64_t *)malloc(hdr_bytes);
        if (!hdr) {
            t->container_ms += bench_now_ms() - _h0;
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "out of memory for %zu byte header, dataset %zu",
                    hdr_bytes, req->u);
            rc = -1; goto done;
        }
        hdr[0] = VOL_CHUNK_MAGIC;
        hdr[1] = (uint64_t)nchunks;
        hdr[2] = (uint64_t)chunk_bytes;
        memcpy(&hdr[VOL_CHUNK_HDR_WORDS], csizes, nchunks * sizeof(uint64_t));
        t->container_ms += bench_now_ms() - _h0;

        if (vol_write_at(req, hdr, 0, (hsize_t)hdr_bytes, total, t) < 0) {
            rc = -1; goto done;
        }

        hsize_t poff = (hsize_t)hdr_bytes;
        for (size_t k = 0; k < nchunks; k++) {
            if (vol_write_at(req, chunk_bufs[k], poff, (hsize_t)csizes[k],
                             total, t) < 0) {
                rc = -1;
                break;
            }
            free(chunk_bufs[k]);
            chunk_bufs[k] = NULL;
            poff += (hsize_t)csizes[k];
        }
    }

done:
    if (chunk_bufs) for (size_t k = 0; k < nchunks; k++) free(chunk_bufs[k]);
    free(chunk_bufs);
    free(csizes);
    free(hdr);
    return rc;
}

/* =========================================================================
 * PATH 4 -- SHARED METADATA
 *
 * Chunked payloads that all reference ONE per-dataset metadata region. An 
 * H5Z filter cannot do this: each chunk callback sees only its own bytes
 * and must emit a standalone result, so shared state would have to be 
 * duplicated per chunk.
 * ========================================================================= */
static herr_t
vol_write_shared(const vol_write_req_t *req, vol_write_timing_t *t)
{
    compression_ctx *ctx = req->ctx;
    void    *container = NULL;
    uint64_t total     = 0;
    herr_t   rc;

    const size_t chunk_bytes_req =
        H5VL_pass_through_ext_chunk_bytes(ctx, req->total_bytes, req->dsize);

    vol_write_reset_timers(ctx, t);
    double _c0 = bench_now_ms();
    rc = H5VL_pass_through_ext_compress_shared(ctx, req->comp_src,
                                               req->total_bytes,
                                               chunk_bytes_req,
                                               &container, &total);
    vol_write_capture_timers(ctx, t, _c0);

    if (rc < 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "shared-metadata compression failed for dataset %zu "
                "compressor='%s'", req->u, ctx->compressor_id);
        free(container);
        return -1;
    }

    rc = vol_write_whole(req, container, (hsize_t)total, t);
    free(container);
    return rc;
}

/* =========================================================================
 * PATH 5 -- PROGRESSIVE
 *
 * L residual layers, coarse to fine. Reading n < L layers reconstructs the
 * full extent at reduced fidelity. Also a single write: the layers are
 * assembled into one container.
 * ========================================================================= */
static herr_t
vol_write_progressive(const vol_write_req_t *req, vol_write_timing_t *t)
{
    compression_ctx *ctx = req->ctx;
    void    *container = NULL;
    uint64_t total     = 0;
    herr_t   rc;

    const int    nlayers = H5VL_pass_through_ext_progressive_layers(ctx);
    const double ratio   = H5VL_pass_through_ext_progressive_ratio(ctx);

    vol_write_reset_timers(ctx, t);
    double _c0 = bench_now_ms();
    rc = H5VL_pass_through_ext_compress_progressive(ctx, req->comp_src,
                                                    req->total_bytes,
                                                    nlayers, ratio,
                                                    &container, &total);
    vol_write_capture_timers(ctx, t, _c0);

    if (rc < 0) {
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "progressive compression failed for dataset %zu "
                "compressor='%s'", req->u, ctx->compressor_id);
        free(container);
        return -1;
    }

    rc = vol_write_whole(req, container, (hsize_t)total, t);
    free(container);
    return rc;
}

/* =========================================================================
 * PATH 0 -- NO COMPRESSION
 *
 * Straight passthrough. Reached only when no compression context exists and
 * none was requested; a missing context after an explicit request is an error
 * raised by the dispatcher, never a silent fall-through to uncompressed
 * storage.
 * ========================================================================= */
static herr_t
vol_write_passthrough(H5VL_pass_through_ext_t *d, void *under,
                      hid_t mem_type_id, hid_t mem_space_id,
                      hid_t file_space_id, hid_t plist_id, const void *buf,
                      size_t u, vol_write_timing_t *t)
{
    const void *wbufs[] = { buf };
    double _io0 = bench_now_ms();
    herr_t rc = H5VLdataset_write(1, &under, d->under_vol_id, &mem_type_id,
                                  &mem_space_id, &file_space_id, plist_id,
                                  wbufs, NULL);
    t->io_ms += bench_now_ms() - _io0;

    if (rc < 0)
        H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                vol_err_class, maj_compression, min_compress_failed,
                "passthrough dataset_write failed for dataset %zu", u);
    return rc;
}

/* =========================================================================
 * H5VL_pass_through_ext_dataset_write DISPATCHER
 * ========================================================================= */
static herr_t
H5VL_pass_through_ext_dataset_write(
    size_t count, void *dset[], hid_t mem_type_id[], hid_t mem_space_id[],
    hid_t file_space_id[], hid_t plist_id, const void *buf[], void **req_out)
{
    herr_t ret_val = 0;

    for (size_t u = 0; u < count; u++) {
        vol_write_timing_t t;
        double _d0 = bench_now_ms();
        memset(&t, 0, sizeof(t));

        H5VL_pass_through_ext_t *d = (H5VL_pass_through_ext_t *)dset[u];
        void *under = d->under_object;
        gpu_vol_dataset_t *ds_ctx = (gpu_vol_dataset_t *)d->custom_data;
        compression_ctx   *ctx    = ds_ctx ? ds_ctx->comp_ctx : NULL;
        const char *dset_name =
            (ctx && ctx->dataset_name[0]) ? ctx->dataset_name : "none";

        /* ---------------- no compression ---------------- */
        if (!ctx) {
            if (ds_ctx && ds_ctx->compression_requested) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_compressor_unavail,
                        "compression was requested but no compression context "
                        "is available -- refusing to fall through to "
                        "uncompressed storage");
                ret_val = -1;
                continue;
            }
            if (vol_write_passthrough(d, under, mem_type_id[u], mem_space_id[u],
                                      file_space_id[u], plist_id, buf[u],
                                      u, &t) < 0)
                ret_val = -1;

            t.total_ms = bench_now_ms() - _d0;
            vol_timing_finalize(&t, dset_name, "none", 0.03);
            vol_timing_emit_ex(dset_name, "none", "write", &t);
            continue;
        }

        /* ---------------- locate this write ---------------- */
        const size_t dsize = pressio_dtype_size(ctx->dtype);
        size_t total_elems = 1;
        for (int i = 0; i < (int)ctx->ndims; i++) total_elems *= ctx->dims[i];
        const size_t total_bytes = total_elems * dsize;

        size_t off_bytes, len_bytes;
        if (file_space_id[u] == H5S_ALL || ctx->ndims == 0) {
            off_bytes = 0;
            len_bytes = total_bytes;
        } else {
            hssize_t np = H5Sget_select_npoints(file_space_id[u]);
            if (np < 0) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_compress_failed,
                        "could not query file selection for dataset %zu", u);
                ret_val = -1; continue;
            }

            hsize_t start[H5S_MAX_RANK], end[H5S_MAX_RANK];
            if (H5Sget_select_bounds(file_space_id[u], start, end) < 0) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_compress_failed,
                        "could not get selection bounds for dataset %zu", u);
                ret_val = -1; continue;
            }

            int contiguous = 1;
            hsize_t block_elems = 1;
            for (int i = 0; i < (int)ctx->ndims; i++) {
                block_elems *= (end[i] - start[i] + 1);
                if (i >= 1 && (start[i] != 0 || end[i] != ctx->dims[i] - 1))
                    contiguous = 0;
            }
            if (!contiguous || block_elems != (hsize_t)np) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_compress_failed,
                        "dataset %zu: non-contiguous partial write not "
                        "supported by the compression VOL", u);
                ret_val = -1; continue;
            }

            hsize_t lin = 0, stride = 1;
            for (int i = (int)ctx->ndims - 1; i >= 0; i--) {
                lin += start[i] * stride;
                stride *= ctx->dims[i];
            }
            off_bytes = (size_t)lin * dsize;
            len_bytes = (size_t)np  * dsize;
        }

        if (off_bytes + len_bytes > total_bytes) {
            H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                    vol_err_class, maj_compression, min_compress_failed,
                    "dataset %zu: write [%zu,%zu) exceeds logical size %zu",
                    u, off_bytes, off_bytes + len_bytes, total_bytes);
            ret_val = -1; continue;
        }

        /* ---------------- staging ----------------
         * Whole writes compress directly from the caller buffer (host OR
         * device -- that is what keeps the GPU path free of a host bounce).
         * Strips are staged on the host until the array is complete. */
        const void *comp_src = NULL;
        const int is_whole_write = (off_bytes == 0 && len_bytes == total_bytes);

        if (is_whole_write) {
            if (ctx->stage_buf) {          /* discard stale partial staging */
                free(ctx->stage_buf);
                ctx->stage_buf    = NULL;
                ctx->stage_filled = 0;
                ctx->stage_total  = 0;
            }
            comp_src = buf[u];
        } else {
            if (H5VL_pass_through_ext_buf_is_device(buf[u])) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_compress_failed,
                        "dataset %zu: device-resident buffers are only "
                        "supported for whole-dataset writes, not partial "
                        "strips", u);
                ret_val = -1; continue;
            }

            if (!ctx->stage_buf) {
                ctx->stage_buf = malloc(total_bytes);
                if (!ctx->stage_buf) {
                    H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                            vol_err_class, maj_compression, min_compress_failed,
                            "out of memory staging %zu bytes for dataset %zu",
                            total_bytes, u);
                    ret_val = -1; continue;
                }
                ctx->stage_total  = total_bytes;
                ctx->stage_filled = 0;
            }

            double _s0 = bench_now_ms();
            memcpy((char *)ctx->stage_buf + off_bytes, buf[u], len_bytes);
            t.stage_ms += bench_now_ms() - _s0;
            ctx->stage_filled += len_bytes;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
            printf("------- DATASET Write strip: off=%zu len=%zu filled=%zu/%zu\n",
                   off_bytes, len_bytes, ctx->stage_filled, ctx->stage_total);
#endif

            if (ctx->stage_filled < ctx->stage_total) {
                /* Partial strip accepted; nothing compressed or written yet. */
                t.total_ms = bench_now_ms() - _d0;
                vol_timing_finalize(&t, dset_name, ctx->compressor_id, 0.03);
                vol_timing_emit_ex(dset_name, ctx->compressor_id,
                                   "write_strip", &t);
                continue;
            }
            comp_src = ctx->stage_buf;
        }

        /* ---------------- dispatch ---------------- */
        vol_write_req_t wreq;
        wreq.d           = d;
        wreq.under       = under;
        wreq.ctx         = ctx;
        wreq.plist_id    = plist_id;
        wreq.comp_src    = comp_src;
        wreq.total_bytes = total_bytes;
        wreq.dsize       = dsize;
        wreq.dset_name   = dset_name;
        wreq.u           = u;

        herr_t rc;
        switch (H5VL_pass_through_ext_chunking_mode(ctx)) {
        case VOL_CHUNKING_PRESSIO:     rc = vol_write_pressio(&wreq, &t);     break;
        case VOL_CHUNKING_VOL:         rc = vol_write_vol(&wreq, &t);         break;
        case VOL_CHUNKING_SHARED:      rc = vol_write_shared(&wreq, &t);      break;
        case VOL_CHUNKING_PROGRESSIVE: rc = vol_write_progressive(&wreq, &t); break;
        case VOL_CHUNKING_NONE:
        default:                       rc = vol_write_native(&wreq, &t);      break;
        }

        /* Staging is consumed by every path; free it in exactly one place.
         * NULL on the whole-write path, so this is unconditional. */
        free(ctx->stage_buf);
        ctx->stage_buf    = NULL;
        ctx->stage_filled = 0;
        ctx->stage_total  = 0;

        if (rc < 0) ret_val = -1;

        t.total_ms = bench_now_ms() - _d0;
        vol_timing_finalize(&t, dset_name, ctx->compressor_id, 0.03);
        vol_timing_emit_ex(dset_name, ctx->compressor_id,
                           (rc < 0) ? "write_failed" : "write", &t);
    }

    return ret_val;
} /* end H5VL_pass_through_ext_dataset_write() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_dataset_get
 *
 * Purpose:     Gets information about a dataset
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_dataset_get(void *dset, H5VL_dataset_get_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)dset;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL DATASET Get\n");
#endif

    gpu_vol_dataset_t *ds_ctx = (gpu_vol_dataset_t *)o->custom_data;
    compression_ctx   *ctx    = ds_ctx ? ds_ctx->comp_ctx : NULL;

    if (ctx && ctx->ndims > 0 && ctx->dims) {
        if (args->op_type == H5VL_DATASET_GET_SPACE) {
            hsize_t dims[H5S_MAX_RANK];
            for (size_t i = 0; i < ctx->ndims; i++)
                dims[i] = (hsize_t)ctx->dims[i];
            hid_t sid = H5Screate_simple((int)ctx->ndims, dims, NULL);
            if (sid < 0) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_decompress_failed,
                        "could not build logical dataspace for compressed dataset");
                return -1;
            }
            args->args.get_space.space_id = sid;
            return 0;
        }
        if (args->op_type == H5VL_DATASET_GET_TYPE) {
            hid_t tid = pressio_to_hdf5_dtype(ctx->dtype);
            if (tid < 0) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_decompress_failed,
                        "could not build logical datatype for compressed dataset");
                return -1;
            }
            args->args.get_type.type_id = tid;
            return 0;
        }

        if (args->op_type == H5VL_DATASET_GET_DCPL) {
            hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
            if (dcpl < 0) {
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_decompress_failed,
                        "could not build logical DCPL for compressed dataset");
                return -1;
            }
            hsize_t chunk[H5S_MAX_RANK];
            for (size_t i = 0; i < ctx->ndims; i++)
                chunk[i] = (hsize_t)ctx->dims[i];
            if (H5Pset_chunk(dcpl, (int)ctx->ndims, chunk) < 0) {
                H5Pclose(dcpl);
                H5Epush(H5E_DEFAULT, __FILE__, __func__, __LINE__,
                        vol_err_class, maj_compression, min_decompress_failed,
                        "could not set chunk on logical DCPL for compressed dataset");
                return -1;
            }
            args->args.get_dcpl.dcpl_id = dcpl;
            return 0;
        }
        if (args->op_type == H5VL_DATASET_GET_STORAGE_SIZE) {
            gpu_vol_dataset_t *ds_ctx = (gpu_vol_dataset_t *)o->custom_data;

            if (ds_ctx && ds_ctx->comp_ctx && args->args.get_storage_size.storage_size) {
                H5VL_dataset_get_args_t sargs;
                hssize_t npts;

                sargs.op_type = H5VL_DATASET_GET_SPACE;
                sargs.args.get_space.space_id = H5I_INVALID_HID;

                if (H5VLdataset_get(o->under_object, o->under_vol_id, &sargs,
                                    dxpl_id, NULL) < 0)
                    return -1;

                npts = H5Sget_simple_extent_npoints(sargs.args.get_space.space_id);
                H5Sclose(sargs.args.get_space.space_id);
                if (npts < 0)
                    return -1;

                *args->args.get_storage_size.storage_size = (hsize_t)npts;
                return 0;
            }
        }

    }

    ret_value = H5VLdataset_get(o->under_object, o->under_vol_id, args, dxpl_id, req);

    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;

    ret_value = H5VLdataset_get(o->under_object, o->under_vol_id, args, dxpl_id, req);

    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_dataset_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_dataset_specific
 *
 * Purpose:     Specific operation on a dataset
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_dataset_specific(void *obj, H5VL_dataset_specific_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    hid_t under_vol_id;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL H5Dspecific\n");
#endif

    // Save copy of underlying VOL connector ID and prov helper, in case of
    // refresh destroying the current object
    under_vol_id = o->under_vol_id;

    ret_value = H5VLdataset_specific(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_dataset_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_dataset_optional
 *
 * Purpose:     Perform a connector-specific operation on a dataset
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_dataset_optional(void *obj, H5VL_optional_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL DATASET Optional\n");
#endif

    /* Sanity check */
    assert(-1 != H5VL_passthru_dataset_foo_op_g);
    assert(-1 != H5VL_passthru_dataset_bar_op_g);

    /* Capture and perform connector-specific 'foo' and 'bar' operations */
    if(args->op_type == H5VL_passthru_dataset_foo_op_g) {
        H5VL_passthru_ext_dataset_foo_args_t *foo_args;      /* Parameters for 'foo' operation */

        /* Set up access to parameters for 'foo' operation */
        foo_args = (H5VL_passthru_ext_dataset_foo_args_t *)args->args;
printf("foo: foo_args->i = %d, foo_args->d = %f\n", foo_args->i, foo_args->d);

        /* <do 'foo', with 'i' and 'd'> */

        /* Set return value */
        ret_value = 0;

    } else if(args->op_type == H5VL_passthru_dataset_bar_op_g) {
        H5VL_passthru_ext_dataset_bar_args_t *bar_args;      /* Parameters for 'bar' operation */

        /* Set up access to parameters for 'bar' operation */
        bar_args = (H5VL_passthru_ext_dataset_bar_args_t *)args->args;
printf("bar: bar_args->dp = %p, bar_args->up = %p\n", bar_args->dp, bar_args->up);

        /* <do 'bar', possibly with 'dp' and 'up'> */

        /* Set values to return to application in parameters */
        if(bar_args->dp)
            *bar_args->dp = 3.14159;
        if(bar_args->up)
            *bar_args->up = 42;

        /* Set return value */
        ret_value = 0;

    } else
        ret_value = H5VLdataset_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_dataset_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_dataset_close
 *
 * Purpose:     Closes a dataset.
 *
 * Return:      Success:    0
 *              Failure:    -1, dataset not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_dataset_close(void *dset, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)dset;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL DATASET Close\n");
#endif

    gpu_vol_dataset_t *ds_ctx = (gpu_vol_dataset_t *)o->custom_data;

    ret_value = H5VLdataset_close(o->under_object, o->under_vol_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    /* Release our wrapper, if underlying dataset was closed */
    if(ret_value >= 0) {
        if (ds_ctx) gpu_vol_dataset_destroy(ds_ctx);
        H5VL_pass_through_ext_free_obj(o);
    }

    return ret_value;
} /* end H5VL_pass_through_ext_dataset_close() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_datatype_commit
 *
 * Purpose:     Commits a datatype inside a container.
 *
 * Return:      Success:    Pointer to datatype object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_datatype_commit(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t type_id, hid_t lcpl_id, hid_t tcpl_id, hid_t tapl_id,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *dt;
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    void *under;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL DATATYPE Commit\n");
#endif

    under = H5VLdatatype_commit(o->under_object, loc_params, o->under_vol_id, name, type_id, lcpl_id, tcpl_id, tapl_id, dxpl_id, req);
    if(under) {
        dt = H5VL_pass_through_ext_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        dt = NULL;

    return (void *)dt;
} /* end H5VL_pass_through_ext_datatype_commit() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_datatype_open
 *
 * Purpose:     Opens a named datatype inside a container.
 *
 * Return:      Success:    Pointer to datatype object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_datatype_open(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t tapl_id, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *dt;
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    void *under;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL DATATYPE Open\n");
#endif

    under = H5VLdatatype_open(o->under_object, loc_params, o->under_vol_id, name, tapl_id, dxpl_id, req);
    if(under) {
        dt = H5VL_pass_through_ext_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        dt = NULL;

    return (void *)dt;
} /* end H5VL_pass_through_ext_datatype_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_datatype_get
 *
 * Purpose:     Get information about a datatype
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_datatype_get(void *dt, H5VL_datatype_get_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)dt;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL DATATYPE Get\n");
#endif

    ret_value = H5VLdatatype_get(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_datatype_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_datatype_specific
 *
 * Purpose:     Specific operations for datatypes
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_datatype_specific(void *obj, H5VL_datatype_specific_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    hid_t under_vol_id;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL DATATYPE Specific\n");
#endif

    // Save copy of underlying VOL connector ID and prov helper, in case of
    // refresh destroying the current object
    under_vol_id = o->under_vol_id;

    ret_value = H5VLdatatype_specific(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_datatype_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_datatype_optional
 *
 * Purpose:     Perform a connector-specific operation on a datatype
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_datatype_optional(void *obj, H5VL_optional_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL DATATYPE Optional\n");
#endif

    ret_value = H5VLdatatype_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_datatype_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_datatype_close
 *
 * Purpose:     Closes a datatype.
 *
 * Return:      Success:    0
 *              Failure:    -1, datatype not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_datatype_close(void *dt, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)dt;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL DATATYPE Close\n");
#endif

    assert(o->under_object);

    ret_value = H5VLdatatype_close(o->under_object, o->under_vol_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    /* Release our wrapper, if underlying datatype was closed */
    if(ret_value >= 0)
        H5VL_pass_through_ext_free_obj(o);

    return ret_value;
} /* end H5VL_pass_through_ext_datatype_close() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_file_create
 *
 * Purpose:     Creates a container using this connector
 *
 * Return:      Success:    Pointer to a file object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_file_create(const char *name, unsigned flags, hid_t fcpl_id,
    hid_t fapl_id, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_info_t *info;
    H5VL_pass_through_ext_t *file;
    hid_t under_fapl_id;
    void *under;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL FILE Create\n");
#endif

    /* Get copy of our VOL info from FAPL */
    H5Pget_vol_info(fapl_id, (void **)&info);

    printf("Info: %p\n", (void *)info);

    /* Make sure we have info about the underlying VOL to be used */
    if (!info)
        return NULL;

    /* Copy the FAPL */
    under_fapl_id = H5Pcopy(fapl_id);

    /* Set the VOL ID and info for the underlying FAPL */
    H5Pset_vol(under_fapl_id, info->under_vol_id, info->under_vol_info);

    /* Open the file with the underlying VOL connector */
    under = H5VLfile_create(name, flags, fcpl_id, under_fapl_id, dxpl_id, req);
    if(under) {
        file = H5VL_pass_through_ext_new_obj(under, info->under_vol_id);

        /* Set the config params */
        file->custom_data = gpu_vol_file_wrap(under_fapl_id, info->under_vol_id, under);
        ((gpu_vol_file_t*)file->custom_data)->compress_on_write = 1;

        /* Check for async request */
        if(req && *req)
            *req = H5VL_pass_through_ext_new_obj(*req, info->under_vol_id);
    } /* end if */
    else
        file = NULL;

    /* Close underlying FAPL */
    H5Pclose(under_fapl_id);

    /* Release copy of our VOL info */
    H5VL_pass_through_ext_info_free(info);

    return (void *)file;
} /* end H5VL_pass_through_ext_file_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_file_open
 *
 * Purpose:     Opens a container created with this connector
 *
 * Return:      Success:    Pointer to a file object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_file_open(const char *name, unsigned flags, hid_t fapl_id,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_info_t *info;
    H5VL_pass_through_ext_t *file;
    hid_t under_fapl_id;
    void *under;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL FILE Open\n");
#endif

    /* Get copy of our VOL info from FAPL */
    H5Pget_vol_info(fapl_id, (void **)&info);

    /* Make sure we have info about the underlying VOL to be used */
    if (!info)
        return NULL;

    /* Copy the FAPL */
    under_fapl_id = H5Pcopy(fapl_id);

    /* Set the VOL ID and info for the underlying FAPL */
    H5Pset_vol(under_fapl_id, info->under_vol_id, info->under_vol_info);

    /* Open the file with the underlying VOL connector */
    under = H5VLfile_open(name, flags, under_fapl_id, dxpl_id, req);
    if(under) {
        file = H5VL_pass_through_ext_new_obj(under, info->under_vol_id);

        file->custom_data = gpu_vol_file_wrap(under_fapl_id, info->under_vol_id, under);
        ((gpu_vol_file_t*)file->custom_data)->compress_on_write = 0;

        /* Check for async request */
        if(req && *req)
            *req = H5VL_pass_through_ext_new_obj(*req, info->under_vol_id);
    } /* end if */
    else
        file = NULL;

    /* Close underlying FAPL */
    H5Pclose(under_fapl_id);

    /* Release copy of our VOL info */
    H5VL_pass_through_ext_info_free(info);

    return (void *)file;
} /* end H5VL_pass_through_ext_file_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_file_get
 *
 * Purpose:     Get info about a file
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_file_get(void *file, H5VL_file_get_args_t *args, hid_t dxpl_id,
    void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)file;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL FILE Get\n");
#endif

    ret_value = H5VLfile_get(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_file_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_file_specific
 *
 * Purpose:     Specific operation on file
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_file_specific(void *file, H5VL_file_specific_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)file;
    H5VL_pass_through_ext_t *new_o;
    H5VL_file_specific_args_t my_args;
    H5VL_file_specific_args_t *new_args;
    H5VL_pass_through_ext_info_t *info;
    hid_t under_vol_id = -1;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL FILE Specific\n");
#endif

    /* Check for 'is accessible' operation */
    if(args->op_type == H5VL_FILE_IS_ACCESSIBLE) {
        /* Make a (shallow) copy of the arguments */
        memcpy(&my_args, args, sizeof(my_args));

        /* Set up the new FAPL for the updated arguments */

        /* Get copy of our VOL info from FAPL */
        H5Pget_vol_info(args->args.is_accessible.fapl_id, (void **)&info);

        /* Make sure we have info about the underlying VOL to be used */
        if (!info)
            return (-1);

        /* Keep the correct underlying VOL ID for later */
        under_vol_id = info->under_vol_id;

        /* Copy the FAPL */
        my_args.args.is_accessible.fapl_id = H5Pcopy(args->args.is_accessible.fapl_id);

        /* Set the VOL ID and info for the underlying FAPL */
        H5Pset_vol(my_args.args.is_accessible.fapl_id, info->under_vol_id, info->under_vol_info);

        /* Set argument pointer to new arguments */
        new_args = &my_args;

        /* Set object pointer for operation */
        new_o = NULL;
    } /* end else-if */
    /* Check for 'delete' operation */
    else if(args->op_type == H5VL_FILE_DELETE) {
        /* Make a (shallow) copy of the arguments */
        memcpy(&my_args, args, sizeof(my_args));

        /* Set up the new FAPL for the updated arguments */

        /* Get copy of our VOL info from FAPL */
        H5Pget_vol_info(args->args.del.fapl_id, (void **)&info);

        /* Make sure we have info about the underlying VOL to be used */
        if (!info)
            return (-1);

        /* Keep the correct underlying VOL ID for later */
        under_vol_id = info->under_vol_id;

        /* Copy the FAPL */
        my_args.args.del.fapl_id = H5Pcopy(args->args.del.fapl_id);

        /* Set the VOL ID and info for the underlying FAPL */
        H5Pset_vol(my_args.args.del.fapl_id, info->under_vol_id, info->under_vol_info);

        /* Set argument pointer to new arguments */
        new_args = &my_args;

        /* Set object pointer for operation */
        new_o = NULL;
    } /* end else-if */
    else {
        /* Keep the correct underlying VOL ID for later */
        under_vol_id = o->under_vol_id;

        /* Set argument pointer to current arguments */
        new_args = args;

        /* Set object pointer for operation */
        new_o = o->under_object;
    } /* end else */

    ret_value = H5VLfile_specific(new_o, under_vol_id, new_args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, under_vol_id);

    /* Check for 'is accessible' operation */
    if(args->op_type == H5VL_FILE_IS_ACCESSIBLE) {
        /* Close underlying FAPL */
        H5Pclose(my_args.args.is_accessible.fapl_id);

        /* Release copy of our VOL info */
        H5VL_pass_through_ext_info_free(info);
    } /* end else-if */
    /* Check for 'delete' operation */
    else if(args->op_type == H5VL_FILE_DELETE) {
        /* Close underlying FAPL */
        H5Pclose(my_args.args.del.fapl_id);

        /* Release copy of our VOL info */
        H5VL_pass_through_ext_info_free(info);
    } /* end else-if */
    else if(args->op_type == H5VL_FILE_REOPEN) {
        /* Wrap reopened file struct pointer, if we reopened one */
        if(ret_value >= 0 && args->args.reopen.file)
            *args->args.reopen.file = H5VL_pass_through_ext_new_obj(*args->args.reopen.file, o->under_vol_id);
    } /* end else */

    return ret_value;
} /* end H5VL_pass_through_ext_file_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_file_optional
 *
 * Purpose:     Perform a connector-specific operation on a file
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_file_optional(void *file, H5VL_optional_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)file;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL File Optional\n");
#endif

    ret_value = H5VLfile_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_file_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_file_close
 *
 * Purpose:     Closes a file.
 *
 * Return:      Success:    0
 *              Failure:    -1, file not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_file_close(void *file, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)file;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL FILE Close\n");
#endif

    ret_value = H5VLfile_close(o->under_object, o->under_vol_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    /* Release our wrapper, if underlying file was closed */
    if(ret_value >= 0) {
        gpu_vol_file_destroy((gpu_vol_file_t*)o->custom_data);
        H5VL_pass_through_ext_free_obj(o);
    }

    return ret_value;
} /* end H5VL_pass_through_ext_file_close() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_group_create
 *
 * Purpose:     Creates a group inside a container
 *
 * Return:      Success:    Pointer to a group object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_group_create(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t lcpl_id, hid_t gcpl_id, hid_t gapl_id,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *group;
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    void *under;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL GROUP Create\n");
#endif

    under = H5VLgroup_create(o->under_object, loc_params, o->under_vol_id, name, lcpl_id, gcpl_id,  gapl_id, dxpl_id, req);
    if(under) {
        group = H5VL_pass_through_ext_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        group = NULL;

    return (void *)group;
} /* end H5VL_pass_through_ext_group_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_group_open
 *
 * Purpose:     Opens a group inside a container
 *
 * Return:      Success:    Pointer to a group object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_group_open(void *obj, const H5VL_loc_params_t *loc_params,
    const char *name, hid_t gapl_id, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *group;
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    void *under;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL GROUP Open\n");
#endif

    under = H5VLgroup_open(o->under_object, loc_params, o->under_vol_id, name, gapl_id, dxpl_id, req);
    if(under) {
        group = H5VL_pass_through_ext_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        group = NULL;

    return (void *)group;
} /* end H5VL_pass_through_ext_group_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_group_get
 *
 * Purpose:     Get info about a group
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_group_get(void *obj, H5VL_group_get_args_t *args, hid_t dxpl_id,
    void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL GROUP Get\n");
#endif

    ret_value = H5VLgroup_get(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_group_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_group_specific
 *
 * Purpose:     Specific operation on a group
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_group_specific(void *obj, H5VL_group_specific_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    H5VL_group_specific_args_t my_args;
    H5VL_group_specific_args_t *new_args;
    hid_t under_vol_id;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL GROUP Specific\n");
#endif

    // Save copy of underlying VOL connector ID and prov helper, in case of
    // refresh destroying the current object
    under_vol_id = o->under_vol_id;

    /* Unpack arguments to get at the child file pointer when mounting a file */
    if(args->op_type == H5VL_GROUP_MOUNT) {

        /* Make a (shallow) copy of the arguments */
        memcpy(&my_args, args, sizeof(my_args));

        /* Set the object for the child file */
        my_args.args.mount.child_file = ((H5VL_pass_through_ext_t *)args->args.mount.child_file)->under_object;

        /* Point to modified arguments */
        new_args = &my_args;
    } /* end if */
    else
        new_args = args;

    ret_value = H5VLgroup_specific(o->under_object, under_vol_id, new_args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_group_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_group_optional
 *
 * Purpose:     Perform a connector-specific operation on a group
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_group_optional(void *obj, H5VL_optional_args_t *args,
    hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL GROUP Optional\n");
#endif

    /* Sanity check */
    assert(-1 != H5VL_passthru_group_fiddle_op_g);

    /* Capture and perform connector-specific 'fiddle' operation */
    if(args->op_type == H5VL_passthru_group_fiddle_op_g) {
        /* No args for 'fiddle' operation */

printf("fiddle\n");

        /* <do 'fiddle'> */

        /* Set return value */
        ret_value = 0;

    } else
        ret_value = H5VLgroup_optional(o->under_object, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_group_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_group_close
 *
 * Purpose:     Closes a group.
 *
 * Return:      Success:    0
 *              Failure:    -1, group not closed.
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_group_close(void *grp, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)grp;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL GROUP Close\n");
#endif

    ret_value = H5VLgroup_close(o->under_object, o->under_vol_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    /* Release our wrapper, if underlying file was closed */
    if(ret_value >= 0)
        H5VL_pass_through_ext_free_obj(o);

    return ret_value;
} /* end H5VL_pass_through_ext_group_close() */

/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_link_create
 *
 * Purpose:     Creates a hard / soft / UD / external link.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_link_create(H5VL_link_create_args_t *args, void *obj,
    const H5VL_loc_params_t *loc_params, hid_t lcpl_id, hid_t lapl_id,
    hid_t dxpl_id, void **req)
{
    H5VL_link_create_args_t my_args;
    H5VL_link_create_args_t *new_args;
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    hid_t under_vol_id = -1;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL LINK Create\n");
#endif

    /* Try to retrieve the "under" VOL id */
    if(o)
        under_vol_id = o->under_vol_id;

    /* Fix up the link target object for hard link creation */
    if(H5VL_LINK_CREATE_HARD == args->op_type) {
        /* If it's a non-NULL pointer, find the 'under object' and re-set the args */
        if(args->args.hard.curr_obj) {
            /* Make a (shallow) copy of the arguments */
            memcpy(&my_args, args, sizeof(my_args));

            /* Check if we still need the "under" VOL ID */
            if(under_vol_id < 0)
                under_vol_id = ((H5VL_pass_through_ext_t *)args->args.hard.curr_obj)->under_vol_id;

            /* Set the object for the link target */
            my_args.args.hard.curr_obj = ((H5VL_pass_through_ext_t *)args->args.hard.curr_obj)->under_object;

            /* Set argument pointer to modified parameters */
            new_args = &my_args;
        } /* end if */
        else
            new_args = args;
    } /* end if */
    else
        new_args = args;

    /* Re-issue 'link create' call, possibly using the unwrapped pieces */
    ret_value = H5VLlink_create(new_args, (o ? o->under_object : NULL), loc_params, under_vol_id, lcpl_id, lapl_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_link_create() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_link_copy
 *
 * Purpose:     Renames an object within an HDF5 container and copies it to a new
 *              group.  The original name SRC is unlinked from the group graph
 *              and then inserted with the new name DST (which can specify a
 *              new path for the object) as an atomic operation. The names
 *              are interpreted relative to SRC_LOC_ID and
 *              DST_LOC_ID, which are either file IDs or group ID.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_link_copy(void *src_obj, const H5VL_loc_params_t *loc_params1,
    void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t lcpl_id,
    hid_t lapl_id, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o_src = (H5VL_pass_through_ext_t *)src_obj;
    H5VL_pass_through_ext_t *o_dst = (H5VL_pass_through_ext_t *)dst_obj;
    hid_t under_vol_id = -1;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL LINK Copy\n");
#endif

    /* Retrieve the "under" VOL id */
    if(o_src)
        under_vol_id = o_src->under_vol_id;
    else if(o_dst)
        under_vol_id = o_dst->under_vol_id;
    assert(under_vol_id > 0);

    ret_value = H5VLlink_copy((o_src ? o_src->under_object : NULL), loc_params1, (o_dst ? o_dst->under_object : NULL), loc_params2, under_vol_id, lcpl_id, lapl_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_link_copy() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_link_move
 *
 * Purpose:     Moves a link within an HDF5 file to a new group.  The original
 *              name SRC is unlinked from the group graph
 *              and then inserted with the new name DST (which can specify a
 *              new path for the object) as an atomic operation. The names
 *              are interpreted relative to SRC_LOC_ID and
 *              DST_LOC_ID, which are either file IDs or group ID.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_link_move(void *src_obj, const H5VL_loc_params_t *loc_params1,
    void *dst_obj, const H5VL_loc_params_t *loc_params2, hid_t lcpl_id,
    hid_t lapl_id, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o_src = (H5VL_pass_through_ext_t *)src_obj;
    H5VL_pass_through_ext_t *o_dst = (H5VL_pass_through_ext_t *)dst_obj;
    hid_t under_vol_id = -1;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL LINK Move\n");
#endif

    /* Retrieve the "under" VOL id */
    if(o_src)
        under_vol_id = o_src->under_vol_id;
    else if(o_dst)
        under_vol_id = o_dst->under_vol_id;
    assert(under_vol_id > 0);

    ret_value = H5VLlink_move((o_src ? o_src->under_object : NULL), loc_params1, (o_dst ? o_dst->under_object : NULL), loc_params2, under_vol_id, lcpl_id, lapl_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_link_move() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_link_get
 *
 * Purpose:     Get info about a link
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_link_get(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_link_get_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL LINK Get\n");
#endif

    ret_value = H5VLlink_get(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_link_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_link_specific
 *
 * Purpose:     Specific operation on a link
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_link_specific(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_link_specific_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL LINK Specific\n");
#endif

    ret_value = H5VLlink_specific(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_link_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_link_optional
 *
 * Purpose:     Perform a connector-specific operation on a link
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_link_optional(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL LINK Optional\n");
#endif

    ret_value = H5VLlink_optional(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_link_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_object_open
 *
 * Purpose:     Opens an object inside a container.
 *
 * Return:      Success:    Pointer to object
 *              Failure:    NULL
 *
 *-------------------------------------------------------------------------
 */
static void *
H5VL_pass_through_ext_object_open(void *obj, const H5VL_loc_params_t *loc_params,
    H5I_type_t *opened_type, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *new_obj;
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    void *under;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL OBJECT Open\n");
#endif

    under = H5VLobject_open(o->under_object, loc_params, o->under_vol_id, opened_type, dxpl_id, req);
    if(under) {
        new_obj = H5VL_pass_through_ext_new_obj(under, o->under_vol_id);

        /* Check for async request */
        if(req && *req)
            *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);
    } /* end if */
    else
        new_obj = NULL;

    return (void *)new_obj;
} /* end H5VL_pass_through_ext_object_open() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_object_copy
 *
 * Purpose:     Copies an object inside a container.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_object_copy(void *src_obj, const H5VL_loc_params_t *src_loc_params,
    const char *src_name, void *dst_obj, const H5VL_loc_params_t *dst_loc_params,
    const char *dst_name, hid_t ocpypl_id, hid_t lcpl_id, hid_t dxpl_id,
    void **req)
{
    H5VL_pass_through_ext_t *o_src = (H5VL_pass_through_ext_t *)src_obj;
    H5VL_pass_through_ext_t *o_dst = (H5VL_pass_through_ext_t *)dst_obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL OBJECT Copy\n");
#endif

    ret_value = H5VLobject_copy(o_src->under_object, src_loc_params, src_name, o_dst->under_object, dst_loc_params, dst_name, o_src->under_vol_id, ocpypl_id, lcpl_id, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o_src->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_object_copy() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_object_get
 *
 * Purpose:     Get info about an object
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_object_get(void *obj, const H5VL_loc_params_t *loc_params, H5VL_object_get_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL OBJECT Get\n");
#endif

    ret_value = H5VLobject_get(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_object_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_object_specific
 *
 * Purpose:     Specific operation on an object
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_object_specific(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_object_specific_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    hid_t under_vol_id;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL OBJECT Specific\n");
#endif

    // Save copy of underlying VOL connector ID and prov helper, in case of
    // refresh destroying the current object
    under_vol_id = o->under_vol_id;

    ret_value = H5VLobject_specific(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_object_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_object_optional
 *
 * Purpose:     Perform a connector-specific operation for an object
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_object_optional(void *obj, const H5VL_loc_params_t *loc_params,
    H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL OBJECT Optional\n");
#endif

    ret_value = H5VLobject_optional(o->under_object, loc_params, o->under_vol_id, args, dxpl_id, req);

    /* Check for async request */
    if(req && *req)
        *req = H5VL_pass_through_ext_new_obj(*req, o->under_vol_id);

    return ret_value;
} /* end H5VL_pass_through_ext_object_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_introspect_get_conn_clss
 *
 * Purpose:     Query the connector class.
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_pass_through_ext_introspect_get_conn_cls(void *obj, H5VL_get_conn_lvl_t lvl,
    const H5VL_class_t **conn_cls)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL INTROSPECT GetConnCls\n");
#endif

    /* Check for querying this connector's class */
    if(H5VL_GET_CONN_LVL_CURR == lvl) {
        *conn_cls = &H5VL_pass_through_ext_g;
        ret_value = 0;
    } /* end if */
    else
        ret_value = H5VLintrospect_get_conn_cls(o->under_object, o->under_vol_id,
            lvl, conn_cls);

    return ret_value;
} /* end H5VL_pass_through_ext_introspect_get_conn_cls() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_introspect_get_cap_flags
 *
 * Purpose:     Query the capability flags for this connector and any
 *              underlying connector(s).
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_pass_through_ext_introspect_get_cap_flags(const void *_info, uint64_t *cap_flags)
{
    const H5VL_pass_through_ext_info_t *info = (const H5VL_pass_through_ext_info_t *)_info;
    herr_t                          ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL INTROSPECT GetCapFlags\n");
#endif

    /* Invoke the query on the underlying VOL connector */
    ret_value = H5VLintrospect_get_cap_flags(info->under_vol_info, info->under_vol_id, cap_flags);

    /* Bitwise OR our capability flags in */
    if (ret_value >= 0)
        *cap_flags |= H5VL_pass_through_ext_g.cap_flags;

    return ret_value;
} /* end H5VL_pass_through_introspect_ext_get_cap_flags() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_introspect_opt_query
 *
 * Purpose:     Query if an optional operation is supported by this connector
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_pass_through_ext_introspect_opt_query(void *obj, H5VL_subclass_t cls,
    int op_type, uint64_t *flags)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL INTROSPECT OptQuery\n");
#endif

    ret_value = H5VLintrospect_opt_query(o->under_object, o->under_vol_id, cls,
        op_type, flags);

    return ret_value;
} /* end H5VL_pass_through_ext_introspect_opt_query() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_request_wait
 *
 * Purpose:     Wait (with a timeout) for an async operation to complete
 *
 * Note:        Releases the request if the operation has completed and the
 *              connector callback succeeds
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_request_wait(void *obj, uint64_t timeout,
    H5VL_request_status_t *status)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL REQUEST Wait\n");
#endif

    ret_value = H5VLrequest_wait(o->under_object, o->under_vol_id, timeout, status);

    return ret_value;
} /* end H5VL_pass_through_ext_request_wait() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_request_notify
 *
 * Purpose:     Registers a user callback to be invoked when an asynchronous
 *              operation completes
 *
 * Note:        Releases the request, if connector callback succeeds
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_request_notify(void *obj, H5VL_request_notify_t cb, void *ctx)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL REQUEST Notify\n");
#endif

    ret_value = H5VLrequest_notify(o->under_object, o->under_vol_id, cb, ctx);

    return ret_value;
} /* end H5VL_pass_through_ext_request_notify() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_request_cancel
 *
 * Purpose:     Cancels an asynchronous operation
 *
 * Note:        Releases the request, if connector callback succeeds
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_request_cancel(void *obj, H5VL_request_status_t *status)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL REQUEST Cancel\n");
#endif

    ret_value = H5VLrequest_cancel(o->under_object, o->under_vol_id, status);

    return ret_value;
} /* end H5VL_pass_through_ext_request_cancel() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_request_specific
 *
 * Purpose:     Specific operation on a request
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_request_specific(void *obj, H5VL_request_specific_args_t *args)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value = -1;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL REQUEST Specific\n");
#endif

    ret_value = H5VLrequest_specific(o->under_object, o->under_vol_id, args);

    return ret_value;
} /* end H5VL_pass_through_ext_request_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_request_optional
 *
 * Purpose:     Perform a connector-specific operation for a request
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_request_optional(void *obj, H5VL_optional_args_t *args)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL REQUEST Optional\n");
#endif

    ret_value = H5VLrequest_optional(o->under_object, o->under_vol_id, args);

    return ret_value;
} /* end H5VL_pass_through_ext_request_optional() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_request_free
 *
 * Purpose:     Releases a request, allowing the operation to complete without
 *              application tracking
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *-------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_request_free(void *obj)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL REQUEST Free\n");
#endif

    ret_value = H5VLrequest_free(o->under_object, o->under_vol_id);

    if(ret_value >= 0)
        H5VL_pass_through_ext_free_obj(o);

    return ret_value;
} /* end H5VL_pass_through_ext_request_free() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_blob_put
 *
 * Purpose:     Handles the blob 'put' callback
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_pass_through_ext_blob_put(void *obj, const void *buf, size_t size,
    void *blob_id, void *ctx)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL BLOB Put\n");
#endif

    ret_value = H5VLblob_put(o->under_object, o->under_vol_id, buf, size,
        blob_id, ctx);

    return ret_value;
} /* end H5VL_pass_through_ext_blob_put() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_blob_get
 *
 * Purpose:     Handles the blob 'get' callback
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_pass_through_ext_blob_get(void *obj, const void *blob_id, void *buf,
    size_t size, void *ctx)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL BLOB Get\n");
#endif

    ret_value = H5VLblob_get(o->under_object, o->under_vol_id, blob_id, buf,
        size, ctx);

    return ret_value;
} /* end H5VL_pass_through_ext_blob_get() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_blob_specific
 *
 * Purpose:     Handles the blob 'specific' callback
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_pass_through_ext_blob_specific(void *obj, void *blob_id,
    H5VL_blob_specific_args_t *args)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL BLOB Specific\n");
#endif

    ret_value = H5VLblob_specific(o->under_object, o->under_vol_id, blob_id, args);

    return ret_value;
} /* end H5VL_pass_through_ext_blob_specific() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_blob_optional
 *
 * Purpose:     Handles the blob 'optional' callback
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_pass_through_ext_blob_optional(void *obj, void *blob_id, H5VL_optional_args_t *args)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL BLOB Optional\n");
#endif

    ret_value = H5VLblob_optional(o->under_object, o->under_vol_id, blob_id, args);

    return ret_value;
} /* end H5VL_pass_through_ext_blob_optional() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_token_cmp
 *
 * Purpose:     Compare two of the connector's object tokens, setting
 *              *cmp_value, following the same rules as strcmp().
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_token_cmp(void *obj, const H5O_token_t *token1,
    const H5O_token_t *token2, int *cmp_value)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL TOKEN Compare\n");
#endif

    /* Sanity checks */
    assert(obj);
    assert(token1);
    assert(token2);
    assert(cmp_value);

    ret_value = H5VLtoken_cmp(o->under_object, o->under_vol_id, token1, token2, cmp_value);

    return ret_value;
} /* end H5VL_pass_through_ext_token_cmp() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_token_to_str
 *
 * Purpose:     Serialize the connector's object token into a string.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_token_to_str(void *obj, H5I_type_t obj_type,
    const H5O_token_t *token, char **token_str)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL TOKEN To string\n");
#endif

    /* Sanity checks */
    assert(obj);
    assert(token);
    assert(token_str);

    ret_value = H5VLtoken_to_str(o->under_object, obj_type, o->under_vol_id, token, token_str);

    return ret_value;
} /* end H5VL_pass_through_ext_token_to_str() */


/*---------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_token_from_str
 *
 * Purpose:     Deserialize the connector's object token from a string.
 *
 * Return:      Success:    0
 *              Failure:    -1
 *
 *---------------------------------------------------------------------------
 */
static herr_t
H5VL_pass_through_ext_token_from_str(void *obj, H5I_type_t obj_type,
    const char *token_str, H5O_token_t *token)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL TOKEN From string\n");
#endif

    /* Sanity checks */
    assert(obj);
    assert(token);
    assert(token_str);

    ret_value = H5VLtoken_from_str(o->under_object, obj_type, o->under_vol_id, token_str, token);

    return ret_value;
} /* end H5VL_pass_through_ext_token_from_str() */


/*-------------------------------------------------------------------------
 * Function:    H5VL_pass_through_ext_optional
 *
 * Purpose:     Handles the generic 'optional' callback
 *
 * Return:      SUCCEED / FAIL
 *
 *-------------------------------------------------------------------------
 */
herr_t
H5VL_pass_through_ext_optional(void *obj, H5VL_optional_args_t *args, hid_t dxpl_id, void **req)
{
    H5VL_pass_through_ext_t *o = (H5VL_pass_through_ext_t *)obj;
    herr_t ret_value;

#ifdef ENABLE_EXT_PASSTHRU_LOGGING
    printf("------- EXT PASS THROUGH VOL generic Optional\n");
#endif

    ret_value = H5VLoptional(o->under_object, o->under_vol_id, args, dxpl_id, req);

    return ret_value;
} /* end H5VL_pass_through_ext_optional() */

