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
/* ABI 3: GPU execution time of a completed batch in seconds (GPUEndTime - GPUStartTime), or -1 if unknown. */
double dg_gpu_seconds(DGBatch *batch);
/* ABI 4: block until a committed batch completes or nanos pass: 1 completed, 0 timed out, -1 error. */
int dg_wait(DGBatch *batch, int64_t nanos);
/* ABI 4: recommendedMaxWorkingSetSize and currentAllocatedSize. */
int dg_device_memory(DGDevice *device, uint64_t *out);
/* ABI 4: per-dispatch timing. Profile before the first dispatch; read the nanoseconds after completion (-1 if unsampled). */
int dg_batch_profile(DGBatch *batch, uint32_t dispatches);
int dg_dispatch_times(DGBatch *batch, int64_t *out, uint32_t count);
/* ABI 5: a debug group around the next dispatch. */
int dg_compute_label(DGBatch *batch, const char *label);
/* ABI 5: timeline fences (MTLSharedEvent). */
typedef struct DGEvent DGEvent;
DGEvent *dg_event_create(DGDevice *device);
uint64_t dg_event_value(DGEvent *event);
int dg_event_signal(DGEvent *event, uint64_t value);
int dg_event_wait(DGEvent *event, uint64_t value, int64_t nanos); /* 1 reached, 0 timed out, -1 error */
void dg_event_release(DGEvent *event);
int dg_batch_wait_event(DGBatch *batch, DGEvent *event, uint64_t value);
int dg_batch_signal_event(DGBatch *batch, DGEvent *event, uint64_t value);
/* ABI 5: a batch on queue q: 0 compute, 1 transfer (a second command queue). */
DGBatch *dg_batch_create_on(DGDevice *device, int q);
/* ABI 5: an indirect dispatch: threadgroup counts read from args at offset. */
int dg_compute_dispatch_indirect(DGBatch *batch, DGBuffer *args, uint64_t offset, uint32_t lx, uint32_t ly, uint32_t lz);
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
