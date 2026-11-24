package org.dynamisengine.gpu.vulkan.compute;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.spi.ComputeDevice;
import org.lwjgl.system.*;
import org.lwjgl.vulkan.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import static org.lwjgl.vulkan.VK10.*;

/** Headless compute discovery. No surface, swapchain, or presentation extensions are requested. */
public final class VulkanProvider implements GpuProvider {
    @Override public Backend backend() { return Backend.VULKAN; }
    static synchronized void loader() {
        if (System.getProperty("dynamisgpu.vk.loader") != null && !System.getProperty("dynamisgpu.vk.loader").isBlank())
            Configuration.VULKAN_LIBRARY_NAME.set(System.getProperty("dynamisgpu.vk.loader"));
        else if (System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("mac") && Files.exists(Path.of("/usr/local/lib/libvulkan.1.dylib")))
            Configuration.VULKAN_LIBRARY_NAME.set("/usr/local/lib/libvulkan.1.dylib");
    }
    static void check(int result, String operation) {
        if (result != VK_SUCCESS) throw new ComputeException(result == VK_ERROR_DEVICE_LOST ? ComputeException.Code.DEVICE_LOST
                : result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY ? ComputeException.Code.RESOURCE_EXHAUSTED
                : ComputeException.Code.NATIVE_ERROR, "Vulkan " + operation + " returned VkResult=" + result);
    }
    static boolean extension(VkPhysicalDevice physical, String name) {
        try (var stack = MemoryStack.stackPush()) {
            var count = stack.ints(0);
            if (physical == null) check(vkEnumerateInstanceExtensionProperties((ByteBuffer) null, count, null), "instance extensions");
            else check(vkEnumerateDeviceExtensionProperties(physical, (ByteBuffer) null, count, null), "device extensions");
            var properties = VkExtensionProperties.calloc(count.get(0), stack);
            if (physical == null) check(vkEnumerateInstanceExtensionProperties((ByteBuffer) null, count, properties), "instance extension list");
            else check(vkEnumerateDeviceExtensionProperties(physical, (ByteBuffer) null, count, properties), "device extension list");
            for (var p : properties) if (p.extensionNameString().equals(name)) return true;
            return false;
        }
    }
    private static int instanceApi;
    /**
     * The API version DynamisGPU asks of the instance: the loader's, up to 1.3, and at least 1.1. Device features of
     * Vulkan 1.2 (16-bit floats, 8-bit storage, 64-bit atomics) are queried and enabled only at 1.2 and above.
     */
    static synchronized int instanceApiVersion() {
        if (instanceApi == 0) {
            loader();
            try (var stack = MemoryStack.stackPush()) {
                var version = stack.ints(VK11.VK_API_VERSION_1_1);
                int loaderVersion = VK11.vkEnumerateInstanceVersion(version) == VK_SUCCESS ? version.get(0) & ~0xFFF : VK11.VK_API_VERSION_1_1;
                instanceApi = Integer.compareUnsigned(loaderVersion, VK13.VK_API_VERSION_1_3) > 0 ? VK13.VK_API_VERSION_1_3
                        : Integer.compareUnsigned(loaderVersion, VK11.VK_API_VERSION_1_1) < 0 ? VK11.VK_API_VERSION_1_1 : loaderVersion;
            }
        }
        return instanceApi;
    }
    /** MoltenVK's fast-math setting and its value MVK_CONFIG_FAST_MATH_NEVER. */
    static final String MOLTENVK_FAST_MATH = "MVK_CONFIG_FAST_MATH_ENABLED";
    static final int MOLTENVK_FAST_MATH_NEVER = 0;

    /**
     * A headless instance. Floating point follows IEEE 754 on every driver: MoltenVK otherwise compiles shaders with
     * Metal fast math, which reassociates arithmetic and assumes no NaN, so its fast math is switched off through
     * VK_EXT_layer_settings (other drivers ignore the MoltenVK setting).
     */
    static VkInstance instance() {
        loader();
        try (var stack = MemoryStack.stackPush()) {
            var app = VkApplicationInfo.calloc(stack).sType$Default().pApplicationName(stack.UTF8("DynamisGPU headless compute")).apiVersion(instanceApiVersion());
            var info = VkInstanceCreateInfo.calloc(stack).sType$Default().pApplicationInfo(app);
            List<String> extensions = new ArrayList<>();
            if (extension(null, "VK_KHR_portability_enumeration")) {
                info.flags(1); extensions.add("VK_KHR_portability_enumeration");
            }
            // debug labels for captures (RenderDoc, Xcode through MoltenVK) where the loader offers them
            if (extension(null, EXTDebugUtils.VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) extensions.add(EXTDebugUtils.VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            if (extension(null, EXTLayerSettings.VK_EXT_LAYER_SETTINGS_EXTENSION_NAME)) {
                extensions.add(EXTLayerSettings.VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
                var value = stack.malloc(4).order(ByteOrder.nativeOrder()).putInt(0, MOLTENVK_FAST_MATH_NEVER);
                var setting = VkLayerSettingEXT.calloc(1, stack);
                setting.get(0).pLayerName(stack.UTF8("MoltenVK")).pSettingName(stack.UTF8(MOLTENVK_FAST_MATH))
                        .type(EXTLayerSettings.VK_LAYER_SETTING_TYPE_INT32_EXT).valueCount(1).pValues(value);
                info.pNext(VkLayerSettingsCreateInfoEXT.calloc(stack).sType$Default().pSettings(setting));
            }
            if (!extensions.isEmpty()) {
                var names = stack.mallocPointer(extensions.size()); for (var name : extensions) names.put(stack.UTF8(name)); names.flip();
                info.ppEnabledExtensionNames(names);
            }
            var out = stack.mallocPointer(1); check(vkCreateInstance(info, null, out), "create instance"); return new VkInstance(out.get(0), info);
        } catch (LinkageError | RuntimeException ex) {
            if (ex instanceof ComputeException c) throw c;
            throw new ComputeException(ComputeException.Code.UNAVAILABLE, "Cannot initialize Vulkan loader; configure dynamisgpu.vk.loader and VK_DRIVER_FILES: " + ex, ex);
        }
    }
    static List<VkPhysicalDevice> physicals(VkInstance instance) {
        try (var stack = MemoryStack.stackPush()) {
            var count = stack.ints(0); check(vkEnumeratePhysicalDevices(instance, count, null), "enumerate devices");
            var out = stack.mallocPointer(count.get(0)); check(vkEnumeratePhysicalDevices(instance, count, out), "device list");
            List<VkPhysicalDevice> list = new ArrayList<>(); for (int i = 0; i < count.get(0); i++) list.add(new VkPhysicalDevice(out.get(i), instance)); return list;
        }
    }
    static int family(VkPhysicalDevice physical) {
        try (var stack = MemoryStack.stackPush()) {
            var count = stack.ints(0); vkGetPhysicalDeviceQueueFamilyProperties(physical, count, null);
            var properties = VkQueueFamilyProperties.calloc(count.get(0), stack); vkGetPhysicalDeviceQueueFamilyProperties(physical, count, properties);
            for (int i = 0; i < properties.capacity(); i++) if (properties.get(i).queueCount() > 0 && (properties.get(i).queueFlags() & VK_QUEUE_COMPUTE_BIT) != 0) return i;
            return -1;
        }
    }
    /** Valid timestamp bits of the queue family (0 when it cannot write timestamps). */
    static int timestampBits(VkPhysicalDevice physical, int family) {
        try (var stack = MemoryStack.stackPush()) {
            var count = stack.ints(0); vkGetPhysicalDeviceQueueFamilyProperties(physical, count, null);
            var properties = VkQueueFamilyProperties.calloc(count.get(0), stack); vkGetPhysicalDeviceQueueFamilyProperties(physical, count, properties);
            return properties.get(family).timestampValidBits();
        }
    }
    @Override public List<GpuAdapter> adapters() {
        var instance = instance();
        try (var stack = MemoryStack.stackPush()) {
            List<GpuAdapter> result = new ArrayList<>(); var physicals = physicals(instance);
            for (int i = 0; i < physicals.size(); i++) {
                var physical = physicals.get(i); if (family(physical) < 0) continue;
                var p = VkPhysicalDeviceProperties.calloc(stack); vkGetPhysicalDeviceProperties(physical, p); var l = p.limits();
                long maxStorage = Integer.toUnsignedLong(l.maxStorageBufferRange());
                var features = VulkanFeatures.query(physical);
                var caps = new GpuCapabilities(maxStorage, maxStorage, Math.toIntExact(l.minStorageBufferOffsetAlignment()),
                        Math.min(l.maxPerStageDescriptorStorageBuffers(), l.maxDescriptorSetStorageBuffers()), l.maxPushConstantsSize(),
                        new Dimensions(l.maxComputeWorkGroupSize(0), l.maxComputeWorkGroupSize(1), l.maxComputeWorkGroupSize(2)), l.maxComputeWorkGroupInvocations(),
                        new Dimensions(l.maxComputeWorkGroupCount(0), l.maxComputeWorkGroupCount(1), l.maxComputeWorkGroupCount(2)),
                        features.dataTypes(), features.shaderFeatures(), timestampBits(physical, family(physical)) > 0 && l.timestampPeriod() > 0,
                        features.minSubgroupSize(), features.maxSubgroupSize());
                result.add(new Adapter(i, p.deviceNameString(), p.vendorID(), p.deviceID(), p.apiVersion(), p.driverVersion(), caps));
            }
            return List.copyOf(result);
        } finally { vkDestroyInstance(instance, null); }
    }
    private record Adapter(int index, String name, int vendor, int deviceId, int apiVersion, int driverVersion, GpuCapabilities capabilities) implements GpuAdapter {
        @Override public Backend backend() { return Backend.VULKAN; }
        @Override public String id() { return "vulkan:" + index + ":" + vendor + ":" + deviceId + ":api=" + apiVersion + ":driver=" + driverVersion; }
        @Override public GpuDevice open() { return new ComputeDevice(this, new VulkanComputeDriver(index)); }
    }
}
