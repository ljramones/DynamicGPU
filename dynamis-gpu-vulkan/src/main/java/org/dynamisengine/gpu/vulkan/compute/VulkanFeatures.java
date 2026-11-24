package org.dynamisengine.gpu.vulkan.compute;

import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.api.compute.GpuCapabilities;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;
import static org.lwjgl.vulkan.VK10.*;
import static org.lwjgl.vulkan.VK11.*;

/**
 * The optional data types and shader features of a Vulkan device: queried for the adapter's capabilities, enabled at
 * device creation, and checked against the SPIR-V capabilities each module declares. Only what is listed here is
 * enabled; DynamisGPU reports nothing it does not enable.
 */
record VulkanFeatures(boolean int64, boolean int16, boolean float64, boolean storage16, boolean float16, boolean int8,
        boolean storage8, boolean int64Atomics, boolean float32AtomicAdd, int subgroupOperations, int minSubgroupSize,
        int maxSubgroupSize, boolean vulkan12) {
    static final String ATOMIC_FLOAT = "VK_EXT_shader_atomic_float";
    private static final String SIZE_CONTROL = "VK_EXT_subgroup_size_control";

    /** The device's supported features, given the instance's API version. */
    static VulkanFeatures query(VkPhysicalDevice physical) {
        try (var stack = MemoryStack.stackPush()) {
            var properties = VkPhysicalDeviceProperties.calloc(stack); vkGetPhysicalDeviceProperties(physical, properties);
            int api = Math.min(properties.apiVersion() & ~0xFFF, VulkanProvider.instanceApiVersion());
            boolean v12 = Integer.compareUnsigned(api, VK12.VK_API_VERSION_1_2) >= 0;
            boolean v13 = Integer.compareUnsigned(api, VK13.VK_API_VERSION_1_3) >= 0;
            boolean atomicFloat = VulkanProvider.extension(physical, ATOMIC_FLOAT);

            var f2 = VkPhysicalDeviceFeatures2.calloc(stack).sType$Default();
            var f11 = VkPhysicalDeviceVulkan11Features.calloc(stack).sType$Default();
            var f12 = VkPhysicalDeviceVulkan12Features.calloc(stack).sType$Default();
            var s16 = VkPhysicalDevice16BitStorageFeatures.calloc(stack).sType$Default();
            var af = VkPhysicalDeviceShaderAtomicFloatFeaturesEXT.calloc(stack).sType$Default();
            long next = 0;
            if (atomicFloat) { af.pNext(next); next = af.address(); }
            if (v12) { f12.pNext(next); f11.pNext(f12.address()); next = f11.address(); }
            else { s16.pNext(next); next = s16.address(); }
            f2.pNext(next);
            vkGetPhysicalDeviceFeatures2(physical, f2);
            var core = f2.features();

            var p2 = VkPhysicalDeviceProperties2.calloc(stack).sType$Default();
            var subgroup = VkPhysicalDeviceSubgroupProperties.calloc(stack).sType$Default();
            var sizes = VkPhysicalDeviceSubgroupSizeControlProperties.calloc(stack).sType$Default();
            boolean sizeControl = v13 || VulkanProvider.extension(physical, SIZE_CONTROL);
            if (sizeControl) { subgroup.pNext(sizes.address()); }
            p2.pNext(subgroup.address());
            vkGetPhysicalDeviceProperties2(physical, p2);
            int operations = (subgroup.supportedStages() & VK_SHADER_STAGE_COMPUTE_BIT) != 0 ? subgroup.supportedOperations() : 0;
            int min = operations == 0 ? 0 : sizeControl && sizes.minSubgroupSize() > 0 ? sizes.minSubgroupSize() : subgroup.subgroupSize();
            int max = operations == 0 ? 0 : sizeControl && sizes.maxSubgroupSize() > 0 ? sizes.maxSubgroupSize() : subgroup.subgroupSize();
            // Slang declares the uniform-and-storage forms of 16- and 8-bit access, so both are required (and enabled)
            boolean storage16 = v12 ? f11.storageBuffer16BitAccess() && f11.uniformAndStorageBuffer16BitAccess()
                    : s16.storageBuffer16BitAccess() && s16.uniformAndStorageBuffer16BitAccess();
            boolean storage8 = v12 && f12.storageBuffer8BitAccess() && f12.uniformAndStorageBuffer8BitAccess();
            return new VulkanFeatures(core.shaderInt64(), core.shaderInt16(), core.shaderFloat64(), storage16,
                    v12 && f12.shaderFloat16(), v12 && f12.shaderInt8(), storage8,
                    v12 && f12.shaderBufferInt64Atomics(), atomicFloat && af.shaderBufferFloat32AtomicAdd(),
                    operations, min, max, v12);
        }
    }

    Set<String> dataTypes() {
        Set<String> s = new TreeSet<>(List.of(GpuCapabilities.FP32, GpuCapabilities.INT32, GpuCapabilities.BOOL));
        if (float16 && storage16) s.add(GpuCapabilities.FP16);
        if (int16 && storage16) s.add(GpuCapabilities.INT16);
        if (int8 && storage8) s.add(GpuCapabilities.INT8);
        if (int64) s.add(GpuCapabilities.INT64);
        if (float64) s.add(GpuCapabilities.FP64);
        return s;
    }

    Set<String> shaderFeatures() {
        Set<String> s = new TreeSet<>();
        if ((subgroupOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0) {
            s.add(GpuCapabilities.SUBGROUP_BASIC);
            if ((subgroupOperations & VK_SUBGROUP_FEATURE_VOTE_BIT) != 0) s.add(GpuCapabilities.SUBGROUP_VOTE);
            if ((subgroupOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0) s.add(GpuCapabilities.SUBGROUP_ARITHMETIC);
            if ((subgroupOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0) s.add(GpuCapabilities.SUBGROUP_BALLOT);
            if ((subgroupOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0) s.add(GpuCapabilities.SUBGROUP_SHUFFLE);
        }
        if (float32AtomicAdd) s.add(GpuCapabilities.ATOMIC_FLOAT32_ADD);
        if (int64 && int64Atomics) s.add(GpuCapabilities.ATOMIC_INT64);
        return s;
    }

    /** The extensions to enable for these features (besides portability). */
    List<String> extensions() { return float32AtomicAdd ? List.of(ATOMIC_FLOAT) : List.of(); }

    /** A feature chain enabling exactly the supported features DynamisGPU reports, for {@code VkDeviceCreateInfo.pNext}. */
    long enableChain(MemoryStack stack) {
        var f2 = VkPhysicalDeviceFeatures2.calloc(stack).sType$Default();
        f2.features().shaderInt64(int64).shaderInt16(int16).shaderFloat64(float64);
        long next = 0;
        if (float32AtomicAdd) {
            var af = VkPhysicalDeviceShaderAtomicFloatFeaturesEXT.calloc(stack).sType$Default().shaderBufferFloat32AtomicAdd(true);
            next = af.address();
        }
        if (vulkan12) {
            var f12 = VkPhysicalDeviceVulkan12Features.calloc(stack).sType$Default().pNext(next).shaderFloat16(float16)
                    .shaderInt8(int8).storageBuffer8BitAccess(storage8).uniformAndStorageBuffer8BitAccess(storage8)
                    .shaderBufferInt64Atomics(int64Atomics);
            var f11 = VkPhysicalDeviceVulkan11Features.calloc(stack).sType$Default().pNext(f12.address())
                    .storageBuffer16BitAccess(storage16).uniformAndStorageBuffer16BitAccess(storage16);
            next = f11.address();
        } else {
            var s16 = VkPhysicalDevice16BitStorageFeatures.calloc(stack).sType$Default().pNext(next)
                    .storageBuffer16BitAccess(storage16).uniformAndStorageBuffer16BitAccess(storage16);
            next = s16.address();
        }
        return f2.pNext(next).address();
    }

    /** SPIR-V capabilities that need a device feature, by capability number, and whether this device enabled it. */
    private Map<Integer, Boolean> gated() {
        boolean basic = (subgroupOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0;
        Map<Integer, Boolean> m = new HashMap<>();
        m.put(9, float16);                  // Float16
        m.put(10, float64);                 // Float64
        m.put(11, int64);                   // Int64
        m.put(12, int64 && int64Atomics);   // Int64Atomics
        m.put(22, int16);                   // Int16
        m.put(39, int8);                    // Int8
        m.put(4433, storage16);             // StorageBuffer16BitAccess
        m.put(4434, storage16);             // UniformAndStorageBuffer16BitAccess
        m.put(4435, false);                 // StoragePushConstant16 (not enabled)
        m.put(4436, false);                 // StorageInputOutput16
        m.put(4448, storage8);              // StorageBuffer8BitAccess
        m.put(4449, storage8);              // UniformAndStorageBuffer8BitAccess
        m.put(4450, false);                 // StoragePushConstant8 (not enabled)
        m.put(61, basic);                   // GroupNonUniform
        m.put(62, basic && (subgroupOperations & VK_SUBGROUP_FEATURE_VOTE_BIT) != 0);
        m.put(63, basic && (subgroupOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0);
        m.put(64, basic && (subgroupOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0);
        m.put(65, basic && (subgroupOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0);
        m.put(66, false);                   // GroupNonUniformShuffleRelative (not reported)
        m.put(67, false);                   // GroupNonUniformClustered (not reported)
        m.put(68, false);                   // GroupNonUniformQuad (not reported)
        m.put(6033, float32AtomicAdd);      // AtomicFloat32AddEXT
        return m;
    }

    /** Rejects a module that declares a capability this device did not enable. */
    void checkModule(byte[] spirv, String entryPoint) {
        var gated = gated(); List<Integer> missing = new ArrayList<>();
        for (int capability : declaredCapabilities(spirv)) if (Boolean.FALSE.equals(gated.get(capability))) missing.add(capability);
        if (!missing.isEmpty())
            throw new ComputeException(ComputeException.Code.UNSUPPORTED, "SPIR-V module for " + entryPoint
                    + " declares capabilities " + missing + " that this Vulkan device does not enable");
    }

    /** The {@code OpCapability} operands of a SPIR-V module. */
    static Set<Integer> declaredCapabilities(byte[] spirv) {
        var words = ByteBuffer.wrap(spirv).order(ByteOrder.LITTLE_ENDIAN).asIntBuffer();
        Set<Integer> caps = new TreeSet<>();
        for (int i = 5; i < words.limit(); ) {
            int word = words.get(i), count = word >>> 16, opcode = word & 0xFFFF;
            if (count == 0) break;
            if (opcode == 17 && i + 1 < words.limit()) caps.add(words.get(i + 1)); // OpCapability
            i += count;
        }
        return caps;
    }
}
