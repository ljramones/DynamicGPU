#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "dynamis_metal.h"
#include <stdio.h>
#include <string.h>

struct DGDevice { id<MTLDevice> device; id<MTLCommandQueue> queue; id<MTLCommandQueue> transfer; };
struct DGBuffer { id<MTLBuffer> buffer; };
struct DGPipeline { id<MTLComputePipelineState> pipeline; };
// ABI 5: timeline fences as MTLSharedEvents. Host waits use a shared event listener and a semaphore, not polling.
typedef struct DGEvent { id<MTLSharedEvent> event; } DGEvent;
static MTLSharedEventListener *listener(void) {
    static MTLSharedEventListener *shared = nil; static dispatch_once_t once;
    dispatch_once(&once, ^{ shared = [[MTLSharedEventListener alloc] init]; });
    return shared;
}
struct DGBatch {
    DGDevice *device; id<MTLCommandBuffer> command; id<MTLComputeCommandEncoder> encoder; dispatch_semaphore_t done; int completed;
    // per-dispatch timing (dg_batch_profile): a counter sample pair per dispatch, and CPU/GPU clock samples at commit
    id<MTLCounterSampleBuffer> samples; uint32_t sampleDispatches, nextDispatch; MTLTimestamp cpu0, gpu0;
    int labelled;   // a debug group is open for the current dispatch
};
static _Thread_local char errorText[4096];
static _Thread_local char deviceName[1024];
static void error(NSString *message) { snprintf(errorText, sizeof(errorText), "%s", message.UTF8String ?: "Unknown Metal failure"); }
#define START @autoreleasepool { @try { errorText[0] = 0;
#define END(value) } @catch (NSException *e) { error(e.reason); return value; } }
uint32_t dg_abi(void) { return 5; }
const char *dg_error(void) { return errorText; }
int dg_device_count(void) { START NSArray *a = MTLCopyAllDevices(); int n = (int)a.count; [a release]; return n; END(-1) }
DGDevice *dg_device_open(int index) { START
    NSArray *a = MTLCopyAllDevices();
    if (index < 0 || (NSUInteger)index >= a.count) { error(@"Metal device index unavailable"); [a release]; return NULL; }
    DGDevice *d = calloc(1, sizeof(*d)); if (!d) { [a release]; error(@"Device wrapper allocation failed"); return NULL; }
    d->device = [a[index] retain]; [a release]; d->queue = [d->device newCommandQueue];
    if (!d->queue) { error(@"newCommandQueue failed"); [d->device release]; free(d); return NULL; } return d;
END(NULL) }
const char *dg_device_name(DGDevice *d) { START snprintf(deviceName, sizeof(deviceName), "%s", d->device.name.UTF8String); return deviceName; END(NULL) }
int dg_device_caps(DGDevice *d, uint64_t *out) { START
    MTLSize t = d->device.maxThreadsPerThreadgroup; out[0] = d->device.maxBufferLength; out[1] = t.width; out[2] = t.height; out[3] = t.depth; return 0;
END(-1) }
int dg_device_features(DGDevice *d, uint64_t *out) { START
    uint64_t f = 0; id<MTLDevice> m = d->device;
    if ([m supportsFamily:MTLGPUFamilyApple7]) f |= 1;
    if ([m supportsFamily:MTLGPUFamilyApple8]) f |= 2;
    if ([m supportsFamily:MTLGPUFamilyApple9]) f |= 4;
    if ([m supportsFamily:MTLGPUFamilyMac2]) f |= 8;
    if ([m supportsFamily:MTLGPUFamilyMetal3]) f |= 16;
    out[0] = f; return 0;
END(-1) }
int dg_device_memory(DGDevice *d, uint64_t *out) { START
    out[0] = d->device.recommendedMaxWorkingSetSize; out[1] = d->device.currentAllocatedSize; return 0;
END(-1) }
DGEvent *dg_event_create(DGDevice *d) { START
    DGEvent *e = calloc(1, sizeof(*e)); if (!e) { error(@"event allocation failed"); return NULL; }
    e->event = [d->device newSharedEvent]; if (!e->event) { free(e); error(@"newSharedEvent failed"); return NULL; }
    e->event.signaledValue = 0; return e;
END(NULL) }
uint64_t dg_event_value(DGEvent *e) { START return e->event.signaledValue; END(0) }
int dg_event_signal(DGEvent *e, uint64_t value) { START e->event.signaledValue = value; return 0; END(-1) }
int dg_event_wait(DGEvent *e, uint64_t value, int64_t nanos) { START
    if (e->event.signaledValue >= value) return 1;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0); dispatch_retain(sem);   // one reference for the handler
    [e->event notifyListener:listener() atValue:value block:^(id<MTLSharedEvent> ev, uint64_t v) { (void) ev; (void) v; dispatch_semaphore_signal(sem); dispatch_release(sem); }];
    long r = dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, nanos)); dispatch_release(sem);
    return r == 0 || e->event.signaledValue >= value ? 1 : 0;
END(-1) }
void dg_event_release(DGEvent *e) { START if (e) { [e->event release]; free(e); } END() }
void dg_device_release(DGDevice *d) { START if (d) { [d->queue release]; [d->transfer release]; [d->device release]; free(d); } END() }
DGBuffer *dg_buffer_create(DGDevice *d, uint64_t bytes, int privateStorage) { START
    id<MTLBuffer> b = [d->device newBufferWithLength:bytes options:privateStorage ? MTLResourceStorageModePrivate : MTLResourceStorageModeShared];
    if (!b) { error(@"Metal buffer allocation failed (size/storage unsupported or memory exhausted)"); return NULL; }
    DGBuffer *v = calloc(1, sizeof(*v)); if (!v) { [b release]; error(@"Buffer wrapper allocation failed"); return NULL; } v->buffer = b; return v;
END(NULL) }
void *dg_buffer_contents(DGBuffer *b) { START return b->buffer.contents; END(NULL) }
void dg_buffer_release(DGBuffer *b) { START if (b) { [b->buffer release]; free(b); } END() }
DGPipeline *dg_pipeline_create(DGDevice *d, const void *bytes, uint64_t size, int source, const char *entry) { START
    NSError *e = nil; id<MTLLibrary> library;
    if (source) {
        NSString *text = [[NSString alloc] initWithBytes:bytes length:size encoding:NSUTF8StringEncoding];
        if (!text) { error(@"Invalid UTF-8 MSL source"); return NULL; }
        // IEEE 754 by default. MTLMathModeSafe alone still fuses a*b + c into one rounding (measured on macOS 27),
        // so the source is compiled with fp contraction off; a kernel may opt back in with its own
        // `#pragma METAL fp contract(fast)`. #line 1 keeps compiler diagnostics on the caller's line numbers.
        MTLCompileOptions *options = [MTLCompileOptions new]; NSString *compiled = text;
        if (@available(macOS 15.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            compiled = [@"#pragma METAL fp contract(off)\n#line 1\n" stringByAppendingString:text];
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        library = [d->device newLibraryWithSource:compiled options:options error:&e]; [options release]; [text release];
    } else {
        dispatch_data_t data = dispatch_data_create(bytes, size, dispatch_get_global_queue(0, 0), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        library = [d->device newLibraryWithData:data error:&e]; dispatch_release(data);
    }
    if (!library) { error(e.localizedDescription); return NULL; }
    id<MTLFunction> function = [library newFunctionWithName:[NSString stringWithUTF8String:entry]]; [library release];
    if (!function) { error([NSString stringWithFormat:@"Metal entry point not found: %s", entry]); return NULL; }
    id<MTLComputePipelineState> p = [d->device newComputePipelineStateWithFunction:function error:&e]; [function release];
    if (!p) { error(e.localizedDescription); return NULL; }
    DGPipeline *v = calloc(1, sizeof(*v)); if (!v) { [p release]; error(@"Pipeline wrapper allocation failed"); return NULL; } v->pipeline = p; return v;
END(NULL) }
uint64_t dg_pipeline_threads(DGPipeline *p) { START return p->pipeline.maxTotalThreadsPerThreadgroup; END(0) }
void dg_pipeline_release(DGPipeline *p) { START if (p) { [p->pipeline release]; free(p); } END() }
// ABI 5: a batch on queue q (0 compute, 1 transfer). The transfer queue is a second command queue, made on first use;
// Metal runs command queues concurrently, and blit work can use the copy engine.
DGBatch *dg_batch_create_on(DGDevice *d, int q) { START
    id<MTLCommandQueue> target = d->queue;
    if (q == 1) {
        if (!d->transfer) { d->transfer = [d->device newCommandQueue]; if (!d->transfer) { error(@"newCommandQueue failed"); return NULL; } }
        target = d->transfer;
    }
    DGBatch *b = calloc(1, sizeof(*b)); if (!b) { error(@"Batch allocation failed"); return NULL; }
    b->device = d; b->command = [[target commandBuffer] retain];
    if (!b->command) { error(@"Metal command buffer allocation failed"); free(b); return NULL; } return b;
END(NULL) }
DGBatch *dg_batch_create(DGDevice *d) { START
    DGBatch *b = calloc(1, sizeof(*b)); if (!b) { error(@"Batch allocation failed"); return NULL; }
    b->device = d; b->command = [[d->queue commandBuffer] retain];
    if (!b->command) { error(@"Metal command buffer allocation failed"); free(b); return NULL; } return b;
END(NULL) }
// Consecutive dispatches share one compute encoder (serial dispatch: each completes, and its writes are visible,
// before the next begins); it is ended before a blit and at commit. One encoder per dispatch cost the GPU ~4 us each.
static void end_compute(DGBatch *b) { if (b->encoder) { [b->encoder endEncoding]; [b->encoder release]; b->encoder = nil; } }
// Waits are encoded before the batch's first command; signals just before commit.
int dg_batch_wait_event(DGBatch *b, DGEvent *e, uint64_t value) { START end_compute(b); [b->command encodeWaitForEvent:e->event value:value]; return 0; END(-1) }
int dg_batch_signal_event(DGBatch *b, DGEvent *e, uint64_t value) { START end_compute(b); [b->command encodeSignalEvent:e->event value:value]; return 0; END(-1) }
int dg_copy(DGBatch *b, DGBuffer *src, uint64_t so, DGBuffer *dst, uint64_t off, uint64_t bytes) { START
    end_compute(b);
    id<MTLBlitCommandEncoder> e = [b->command blitCommandEncoder]; if (!e) { error(@"Blit encoder unavailable"); return -1; }
    [e copyFromBuffer:src->buffer sourceOffset:so toBuffer:dst->buffer destinationOffset:off size:bytes]; [e endEncoding]; return 0;
END(-1) }
int dg_upload(DGBatch *b, DGBuffer *dst, uint64_t offset, const void *data, uint64_t bytes) { START
    id<MTLBuffer> stage = [b->device->device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
    if (!stage) { error(@"Upload staging allocation failed"); return -1; }
    DGBuffer source = {stage}; int result = dg_copy(b, &source, 0, dst, offset, bytes); [stage release]; return result;
END(-1) }
int dg_fill(DGBatch *b, DGBuffer *dst, uint64_t offset, uint64_t bytes, uint32_t pattern) { START
    id<MTLBuffer> stage = [b->device->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!stage) { error(@"Fill staging allocation failed"); return -1; }
    uint32_t *words = stage.contents; for (uint64_t i = 0; i < bytes / 4; ++i) words[i] = pattern;
    DGBuffer source = {stage}; int result = dg_copy(b, &source, 0, dst, offset, bytes); [stage release]; return result;
END(-1) }
// Profiling: Apple GPUs sample counters at stage (encoder) boundaries only, so each dispatch gets its own encoder
// with a start and end timestamp sample.
int dg_batch_profile(DGBatch *b, uint32_t dispatches) { START
    if (dispatches == 0) return 0;
    id<MTLDevice> m = b->device->device;
    if (![m supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) { error(@"no stage-boundary counter sampling"); return -1; }
    id<MTLCounterSet> timestamps = nil;
    for (id<MTLCounterSet> set in m.counterSets) if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) timestamps = set;
    if (!timestamps) { error(@"no timestamp counter set"); return -1; }
    MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
    d.counterSet = timestamps; d.storageMode = MTLStorageModeShared; d.sampleCount = 2 * dispatches;
    NSError *e = nil; b->samples = [m newCounterSampleBufferWithDescriptor:d error:&e]; [d release];
    if (!b->samples) { error(e ? e.localizedDescription : @"counter sample buffer allocation failed"); return -1; }
    b->sampleDispatches = dispatches; b->nextDispatch = 0; return 0;
END(-1) }
int dg_compute_begin(DGBatch *b, DGPipeline *p) { START
    if (b->samples && b->nextDispatch < b->sampleDispatches) {
        end_compute(b);
        MTLComputePassDescriptor *pd = [MTLComputePassDescriptor computePassDescriptor];
        pd.sampleBufferAttachments[0].sampleBuffer = b->samples;
        pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * b->nextDispatch;
        pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * b->nextDispatch + 1;
        b->encoder = [[b->command computeCommandEncoderWithDescriptor:pd] retain];
        if (!b->encoder) { error(@"Compute encoder unavailable"); return -1; }
    } else if (!b->encoder) {
        b->encoder = [[b->command computeCommandEncoder] retain]; if (!b->encoder) { error(@"Compute encoder unavailable"); return -1; }
    }
    [b->encoder setComputePipelineState:p->pipeline]; return 0;
END(-1) }
// ABI 5: a debug group around the next dispatch (Xcode GPU captures show it); dg_compute_dispatch pops it.
int dg_compute_label(DGBatch *b, const char *label) { START
    if (!b->encoder) { error(@"no compute encoder"); return -1; }
    [b->encoder pushDebugGroup:[NSString stringWithUTF8String:label]]; b->labelled = 1; return 0;
END(-1) }
int dg_compute_buffer(DGBatch *b, DGBuffer *buffer, uint64_t offset, uint32_t slot) { START [b->encoder setBuffer:buffer->buffer offset:offset atIndex:slot]; return 0; END(-1) }
int dg_compute_parameters(DGBatch *b, const void *data, uint64_t bytes, uint32_t slot) { START [b->encoder setBytes:data length:bytes atIndex:slot]; return 0; END(-1) }
int dg_compute_dispatch(DGBatch *b, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t lx, uint32_t ly, uint32_t lz) { START
    [b->encoder dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(lx, ly, lz)];
    if (b->labelled) { [b->encoder popDebugGroup]; b->labelled = 0; }
    if (b->samples && b->nextDispatch < b->sampleDispatches) { end_compute(b); b->nextDispatch++; }
    return 0;
END(-1) }
int dg_commit(DGBatch *b) { START
    end_compute(b);
    if (b->samples) [b->device->device sampleTimestamps:&b->cpu0 gpuTimestamp:&b->gpu0];
    // A completion handler signals a semaphore so dg_wait can block instead of polling. The handler holds its own
    // reference to the semaphore: it may run after dg_batch_release.
    b->done = dispatch_semaphore_create(0);
    dispatch_semaphore_t sem = b->done; dispatch_retain(sem);
    [b->command addCompletedHandler:^(id<MTLCommandBuffer> cb) { (void) cb; dispatch_semaphore_signal(sem); dispatch_release(sem); }];
    [b->command commit]; return 0;
END(-1) }
int dg_wait(DGBatch *b, int64_t nanos) { START
    if (b->completed) return 1;
    if (!b->done) { error(@"batch not committed"); return -1; }
    if (dispatch_semaphore_wait(b->done, dispatch_time(DISPATCH_TIME_NOW, nanos)) != 0) return 0;
    b->completed = 1; return 1;
END(-1) }
int dg_status(DGBatch *b) { START
    if (b->command.status == MTLCommandBufferStatusError) { error(b->command.error.localizedDescription); return -1; }
    return b->command.status == MTLCommandBufferStatusCompleted ? 1 : 0;
END(-1) }
// The profiled dispatches' GPU nanoseconds (ABI 4). GPU ticks become nanoseconds through the CPU clock: the CPU and
// GPU timestamps sampled at commit and now give the ratio. sampleTimestamps reports the CPU timestamp in
// nanoseconds (measured on an M4 Max: converting it again as mach ticks made every time 125/3 too large).
int dg_dispatch_times(DGBatch *b, int64_t *out, uint32_t count) { START
    if (!b->samples || count > b->sampleDispatches) { error(@"batch was not profiled for that many dispatches"); return -1; }
    NSData *data = [b->samples resolveCounterRange:NSMakeRange(0, 2 * count)];
    if (!data) { error(@"counter samples unavailable"); return -1; }
    const MTLCounterResultTimestamp *t = (const MTLCounterResultTimestamp *) data.bytes;
    MTLTimestamp cpu1, gpu1; [b->device->device sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
    double cpuNanos = (double) (cpu1 - b->cpu0), gpuTicks = (double) (gpu1 - b->gpu0);
    double nanosPerTick = gpuTicks > 0 && cpuNanos > 0 ? cpuNanos / gpuTicks : 1.0;
    for (uint32_t k = 0; k < count; k++) {
        uint64_t s = t[2 * k].timestamp, e = t[2 * k + 1].timestamp;
        out[k] = s == MTLCounterErrorValue || e == MTLCounterErrorValue || e < s ? -1 : (int64_t) ((e - s) * nanosPerTick + 0.5);
    }
    return 0;
END(-1) }
double dg_gpu_seconds(DGBatch *b) { START
    CFTimeInterval start = b->command.GPUStartTime, end = b->command.GPUEndTime;
    return end > start ? end - start : -1.0;
END(-1.0) }
void dg_batch_release(DGBatch *b) { START if (b) { if (b->encoder) { [b->encoder endEncoding]; [b->encoder release]; } [b->command release]; if (b->done) dispatch_release(b->done); [b->samples release]; free(b); } END() }
