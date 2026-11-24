#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "dynamis_metal.h"
#include <stdio.h>
#include <string.h>

struct DGDevice { id<MTLDevice> device; id<MTLCommandQueue> queue; };
struct DGBuffer { id<MTLBuffer> buffer; };
struct DGPipeline { id<MTLComputePipelineState> pipeline; };
struct DGBatch { DGDevice *device; id<MTLCommandBuffer> command; id<MTLComputeCommandEncoder> encoder; };
static _Thread_local char errorText[4096];
static _Thread_local char deviceName[1024];
static void error(NSString *message) { snprintf(errorText, sizeof(errorText), "%s", message.UTF8String ?: "Unknown Metal failure"); }
#define START @autoreleasepool { @try { errorText[0] = 0;
#define END(value) } @catch (NSException *e) { error(e.reason); return value; } }
uint32_t dg_abi(void) { return 3; }
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
void dg_device_release(DGDevice *d) { START if (d) { [d->queue release]; [d->device release]; free(d); } END() }
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
DGBatch *dg_batch_create(DGDevice *d) { START
    DGBatch *b = calloc(1, sizeof(*b)); if (!b) { error(@"Batch allocation failed"); return NULL; }
    b->device = d; b->command = [[d->queue commandBuffer] retain];
    if (!b->command) { error(@"Metal command buffer allocation failed"); free(b); return NULL; } return b;
END(NULL) }
int dg_copy(DGBatch *b, DGBuffer *src, uint64_t so, DGBuffer *dst, uint64_t off, uint64_t bytes) { START
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
int dg_compute_begin(DGBatch *b, DGPipeline *p) { START
    b->encoder = [[b->command computeCommandEncoder] retain]; if (!b->encoder) { error(@"Compute encoder unavailable"); return -1; }
    [b->encoder setComputePipelineState:p->pipeline]; return 0;
END(-1) }
int dg_compute_buffer(DGBatch *b, DGBuffer *buffer, uint64_t offset, uint32_t slot) { START [b->encoder setBuffer:buffer->buffer offset:offset atIndex:slot]; return 0; END(-1) }
int dg_compute_parameters(DGBatch *b, const void *data, uint64_t bytes, uint32_t slot) { START [b->encoder setBytes:data length:bytes atIndex:slot]; return 0; END(-1) }
int dg_compute_dispatch(DGBatch *b, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t lx, uint32_t ly, uint32_t lz) { START
    [b->encoder dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(lx, ly, lz)];
    [b->encoder endEncoding]; [b->encoder release]; b->encoder = nil; return 0;
END(-1) }
int dg_commit(DGBatch *b) { START [b->command commit]; return 0; END(-1) }
int dg_status(DGBatch *b) { START
    if (b->command.status == MTLCommandBufferStatusError) { error(b->command.error.localizedDescription); return -1; }
    return b->command.status == MTLCommandBufferStatusCompleted ? 1 : 0;
END(-1) }
double dg_gpu_seconds(DGBatch *b) { START
    CFTimeInterval start = b->command.GPUStartTime, end = b->command.GPUEndTime;
    return end > start ? end - start : -1.0;
END(-1.0) }
void dg_batch_release(DGBatch *b) { START if (b) { if (b->encoder) { [b->encoder endEncoding]; [b->encoder release]; } [b->command release]; free(b); } END() }
