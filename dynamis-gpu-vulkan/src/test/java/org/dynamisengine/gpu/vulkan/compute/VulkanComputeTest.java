package org.dynamisengine.gpu.vulkan.compute;
import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.test.compute.ComputeConformance;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import java.util.Map;
@EnabledIfSystemProperty(named = "compute.vulkan.required", matches = "true")
class VulkanComputeTest extends org.dynamisengine.gpu.test.compute.CapabilityTierConformance {
    @org.junit.jupiter.api.Test void malformedShaderRejectedBeforeDriver() {
        PipelineDescriptor good; try (var p = pipeline(64)) { good = p.descriptor(); }
        var bad = new KernelBundle("invalid", "1", Map.of(Backend.VULKAN, new KernelBundle.Executable(KernelBundle.Format.SPIRV, new byte[24])));
        var descriptor = new PipelineDescriptor(bad, "vector_add", good.layout(), good.localSize(), Map.of(), java.util.Set.of(), java.util.Set.of());
        org.junit.jupiter.api.Assertions.assertEquals(ComputeException.Code.INVALID_ARGUMENT,
            org.junit.jupiter.api.Assertions.assertThrows(ComputeException.class, () -> device.pipeline(descriptor)).code());
    }
    @Override protected KernelBundle strictKernel() {
        try (var stream = ComputeConformance.class.getResourceAsStream("/compute/fp_strict.spv")) {
            return new KernelBundle("fp-strict", "1", Map.of(Backend.VULKAN, new KernelBundle.Executable(KernelBundle.Format.SPIRV, stream.readAllBytes())));
        } catch (java.io.IOException e) { throw new AssertionError(e); }
    }
    /** MoltenVK (Apple's vendor ID, 0x106B) runs with fast math off; other drivers may reassociate unmarked code. */
    @Override protected boolean strictWithoutPreciseMarks() { return device.adapter().id().contains(":" + 0x106B + ":"); }
    @Override protected KernelBundle errorKernel() {
        try (var stream = ComputeConformance.class.getResourceAsStream("/compute/error_report.spv")) {
            return new KernelBundle("error-report", "1", Map.of(Backend.VULKAN, new KernelBundle.Executable(KernelBundle.Format.SPIRV, stream.readAllBytes())));
        } catch (java.io.IOException e) { throw new AssertionError(e); }
    }
    /** MoltenVK on Apple GPUs (vendor 0x106B) fuses fma; Mesa's lavapipe (vendor 0x10005) rounds twice. */
    @Override protected Boolean expectedFusedFma() {
        String id = device.adapter().id();
        if (id.contains(":" + 0x106B + ":")) return true;
        if (id.contains(":" + 0x10005 + ":")) return false;
        return null;
    }
    /** MoltenVK on Apple GPUs and Mesa's lavapipe; lavapipe adds FP64 and 64-bit atomics. */
    @Override protected java.util.Set<String> expectedTiers() {
        String id = device.adapter().id(); var apple = java.util.Set.of(GpuCapabilities.FP16, GpuCapabilities.INT16, GpuCapabilities.INT8, GpuCapabilities.INT64,
                GpuCapabilities.SUBGROUP_BASIC, GpuCapabilities.SUBGROUP_VOTE, GpuCapabilities.SUBGROUP_ARITHMETIC,
                GpuCapabilities.SUBGROUP_BALLOT, GpuCapabilities.SUBGROUP_SHUFFLE, GpuCapabilities.ATOMIC_FLOAT32_ADD);
        if (id.contains(":" + 0x106B + ":")) return apple;
        if (id.contains(":" + 0x10005 + ":")) {
            var all = new java.util.TreeSet<>(apple); all.add(GpuCapabilities.FP64); all.add(GpuCapabilities.ATOMIC_INT64); return all;
        }
        return java.util.Set.of();
    }
    /** A module that uses a capability the device lacks is refused even when its descriptor does not declare it. */
    @org.junit.jupiter.api.Test void undeclaredCapabilitiesAreRefused() {
        for (var t : ALL) {
            if (reports(t)) continue;
            var undeclared = descriptor(t, false);
            org.junit.jupiter.api.Assertions.assertEquals(ComputeException.Code.UNSUPPORTED,
                org.junit.jupiter.api.Assertions.assertThrows(ComputeException.class, () -> device.pipeline(undeclared)).code(), t.kernel());
            System.out.println("undeclared " + t.requirement() + " refused");
        }
    }
    /** With -Ddynamisgpu.pipelineCache, closing a device saves its pipeline cache; another device loads it; junk is ignored. */
    @org.junit.jupiter.api.Test void pipelineCachePersistsAcrossDevices() throws java.io.IOException {
        var dir = java.nio.file.Files.createTempDirectory("dynamisgpu-cache");
        String previous = System.getProperty(VulkanComputeDriver.PIPELINE_CACHE_PROPERTY);
        System.setProperty(VulkanComputeDriver.PIPELINE_CACHE_PROPERTY, dir.toString());
        try {
            try (var d = provider().adapters().getFirst().open(); var p = d.pipeline(pipeline(64).descriptor())) { org.junit.jupiter.api.Assertions.assertNotNull(p); }
            java.nio.file.Path file;
            try (var files = java.nio.file.Files.list(dir)) { file = files.filter(f -> f.toString().endsWith(".bin")).findFirst().orElseThrow(); }
            org.junit.jupiter.api.Assertions.assertTrue(java.nio.file.Files.size(file) > 0, "cache saved");
            try (var d = provider().adapters().getFirst().open(); var p = d.pipeline(pipeline(64).descriptor())) { org.junit.jupiter.api.Assertions.assertNotNull(p); }
            java.nio.file.Files.write(file, new byte[] {1, 2, 3, 4, 5, 6, 7, 8});   // junk: ignored, not fatal
            try (var d = provider().adapters().getFirst().open(); var p = d.pipeline(pipeline(64).descriptor())) { org.junit.jupiter.api.Assertions.assertNotNull(p); }
            org.junit.jupiter.api.Assertions.assertTrue(java.nio.file.Files.size(file) > 8, "rewritten after junk");
        } finally {
            if (previous == null) System.clearProperty(VulkanComputeDriver.PIPELINE_CACHE_PROPERTY); else System.setProperty(VulkanComputeDriver.PIPELINE_CACHE_PROPERTY, previous);
            try (var files = java.nio.file.Files.walk(dir)) { files.sorted(java.util.Comparator.reverseOrder()).forEach(f -> f.toFile().delete()); }
        }
    }
    /**
     * Interop as an engine would use it: rebuild LWJGL handles from the exported ones, record DynamisGPU work into an
     * engine-owned command buffer, follow it with an engine copy through an exported buffer handle, submit, and close
     * the external recording after the engine's fence.
     */
    @org.junit.jupiter.api.Test void recordsIntoAnEngineCommandBuffer() {
        int n = 256;
        var interop = device.interop(); var nd = interop.nativeDevice();
        try (var a = buffer(n * 4, Placement.PRIVATE); var b = buffer(n * 4, Placement.PRIVATE); var c = buffer(n * 4, Placement.PRIVATE);
             var d = buffer(n * 4, Placement.PRIVATE); var p = pipeline(n); var stack = org.lwjgl.system.MemoryStack.stackPush()) {
            var app = org.lwjgl.vulkan.VkApplicationInfo.calloc(stack).sType$Default().apiVersion(VulkanProvider.instanceApiVersion());
            var instance = new org.lwjgl.vulkan.VkInstance(nd.instance(), org.lwjgl.vulkan.VkInstanceCreateInfo.calloc(stack).sType$Default().pApplicationInfo(app));
            var physical = new org.lwjgl.vulkan.VkPhysicalDevice(nd.physicalDevice(), instance);
            var vk = new org.lwjgl.vulkan.VkDevice(nd.device(), physical, org.lwjgl.vulkan.VkDeviceCreateInfo.calloc(stack).sType$Default());
            var queue = new org.lwjgl.vulkan.VkQueue(nd.queue(), vk);
            var out = stack.mallocLong(1);
            VulkanProvider.check(org.lwjgl.vulkan.VK10.vkCreateCommandPool(vk, org.lwjgl.vulkan.VkCommandPoolCreateInfo.calloc(stack).sType$Default().queueFamilyIndex(nd.queueFamily()), null, out), "engine pool");
            long pool = out.get(0);
            var ptr = stack.mallocPointer(1);
            VulkanProvider.check(org.lwjgl.vulkan.VK10.vkAllocateCommandBuffers(vk, org.lwjgl.vulkan.VkCommandBufferAllocateInfo.calloc(stack).sType$Default().commandPool(pool)
                    .level(org.lwjgl.vulkan.VK10.VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1), ptr), "engine command buffer");
            var cmd = new org.lwjgl.vulkan.VkCommandBuffer(ptr.get(0), vk);
            VulkanProvider.check(org.lwjgl.vulkan.VK10.vkBeginCommandBuffer(cmd, org.lwjgl.vulkan.VkCommandBufferBeginInfo.calloc(stack).sType$Default()), "begin");
            AutoCloseable external;
            try (var e = device.encoder()) {
                var x = java.nio.ByteBuffer.allocate(n * 4).order(java.nio.ByteOrder.nativeOrder()); var y = java.nio.ByteBuffer.allocate(n * 4).order(java.nio.ByteOrder.nativeOrder());
                for (int i = 0; i < n; i++) { x.putFloat(i); y.putFloat(100); }
                e.upload(a, x.array()).upload(b, y.array()).dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), new Dimensions(n / 64, 1, 1));
                external = interop.recordInto(e, cmd.address());
            }
            // the engine copies DynamisGPU's result into d through the exported handles
            var nc = interop.nativeBuffer(c); var ndst = interop.nativeBuffer(d);
            var region = org.lwjgl.vulkan.VkBufferCopy.calloc(1, stack); region.get(0).srcOffset(nc.offset()).dstOffset(ndst.offset()).size(n * 4L);
            org.lwjgl.vulkan.VK10.vkCmdCopyBuffer(cmd, nc.handle(), ndst.handle(), region);
            VulkanProvider.check(org.lwjgl.vulkan.VK10.vkEndCommandBuffer(cmd), "end");
            VulkanProvider.check(org.lwjgl.vulkan.VK10.vkCreateFence(vk, org.lwjgl.vulkan.VkFenceCreateInfo.calloc(stack).sType$Default(), null, out), "engine fence");
            long fence = out.get(0);
            VulkanProvider.check(org.lwjgl.vulkan.VK10.vkQueueSubmit(queue, org.lwjgl.vulkan.VkSubmitInfo.calloc(stack).sType$Default().pCommandBuffers(stack.pointers(cmd.address())), fence), "engine submit");
            VulkanProvider.check(org.lwjgl.vulkan.VK10.vkWaitForFences(vk, new long[] {fence}, true, 10_000_000_000L), "engine wait");
            external.close();
            org.lwjgl.vulkan.VK10.vkDestroyFence(vk, fence, null); org.lwjgl.vulkan.VK10.vkDestroyCommandPool(vk, pool, null);
            try (var e = device.encoder()) {
                var r = e.download(d); await(device.queue().submit(e));
                var got = java.nio.ByteBuffer.wrap(r.bytes()).order(java.nio.ByteOrder.nativeOrder());
                for (int i = 0; i < n; i++) org.junit.jupiter.api.Assertions.assertEquals(i + 100f, got.getFloat(4 * i), "element " + i);
            }
        } catch (Exception ex) { throw new AssertionError(ex); }
    }
    @Override protected GpuProvider provider() { return new VulkanProvider(); }
    @Override protected KernelBundle kernel() {
        try (var stream = ComputeConformance.class.getResourceAsStream("/compute/vector_add.spv")) {
            return new KernelBundle("vector-add", "1", Map.of(Backend.VULKAN, new KernelBundle.Executable(KernelBundle.Format.SPIRV, stream.readAllBytes())));
        } catch (java.io.IOException e) { throw new AssertionError(e); }
    }
}
