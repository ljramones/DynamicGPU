#ifndef DYNAMIS_METAL_H
#define DYNAMIS_METAL_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct DGDevice DGDevice;
typedef struct DGBuffer DGBuffer;
typedef struct DGPipeline DGPipeline;
typedef struct DGBatch DGBatch;
uint32_t dg_abi(void);
const char *dg_error(void); /* Thread-local, copy before next native call. */
int dg_device_count(void);
DGDevice *dg_device_open(int index);
const char *dg_device_name(DGDevice *device);
int dg_device_caps(DGDevice *device, uint64_t *out); /* max buffer, max threads x/y/z */
/* ABI 2: GPU families, bits 0-4: Apple7, Apple8, Apple9, Mac2, Metal3. */
int dg_device_features(DGDevice *device, uint64_t *out);
void dg_device_release(DGDevice *device);
DGBuffer *dg_buffer_create(DGDevice *device, uint64_t bytes, int privateStorage);
void *dg_buffer_contents(DGBuffer *buffer);
void dg_buffer_release(DGBuffer *buffer);
DGPipeline *dg_pipeline_create(DGDevice *device, const void *bytes, uint64_t size, int source, const char *entry);
uint64_t dg_pipeline_threads(DGPipeline *pipeline);
void dg_pipeline_release(DGPipeline *pipeline);
DGBatch *dg_batch_create(DGDevice *device);
int dg_copy(DGBatch *batch, DGBuffer *src, uint64_t srcOffset, DGBuffer *dst, uint64_t dstOffset, uint64_t bytes);
int dg_upload(DGBatch *batch, DGBuffer *dst, uint64_t offset, const void *data, uint64_t bytes);
int dg_fill(DGBatch *batch, DGBuffer *dst, uint64_t offset, uint64_t bytes, uint32_t pattern);
int dg_compute_begin(DGBatch *batch, DGPipeline *pipeline);
int dg_compute_buffer(DGBatch *batch, DGBuffer *buffer, uint64_t offset, uint32_t slot);
int dg_compute_parameters(DGBatch *batch, const void *data, uint64_t bytes, uint32_t slot);
int dg_compute_dispatch(DGBatch *batch, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t lx, uint32_t ly, uint32_t lz);
int dg_commit(DGBatch *batch);
int dg_status(DGBatch *batch); /* 0 pending, 1 complete, -1 error */
void dg_batch_release(DGBatch *batch);
#ifdef __cplusplus
}
#endif
#endif
