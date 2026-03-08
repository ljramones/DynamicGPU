package org.dynamisengine.gpu.bench.ingest.meshforge;

import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VK;
import org.lwjgl.vulkan.VK10;
import org.lwjgl.vulkan.VkApplicationInfo;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkDeviceCreateInfo;
import org.lwjgl.vulkan.VkDeviceQueueCreateInfo;
import org.lwjgl.vulkan.VkExtensionProperties;
import org.lwjgl.vulkan.VkInstance;
import org.lwjgl.vulkan.VkInstanceCreateInfo;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkQueueFamilyProperties;
import org.lwjgl.vulkan.VK11;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import static org.lwjgl.vulkan.KHRPortabilityEnumeration.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
import static org.lwjgl.vulkan.VK10.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
import static org.lwjgl.vulkan.VK10.VK_NULL_HANDLE;
import static org.lwjgl.vulkan.VK10.VK_QUEUE_GRAPHICS_BIT;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_APPLICATION_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
import static org.lwjgl.vulkan.VK10.VK_SUCCESS;
import static org.lwjgl.vulkan.VK10.vkCreateCommandPool;
import static org.lwjgl.vulkan.VK10.vkCreateDevice;
import static org.lwjgl.vulkan.VK10.vkCreateInstance;
import static org.lwjgl.vulkan.VK10.vkDestroyCommandPool;
import static org.lwjgl.vulkan.VK10.vkDestroyDevice;
import static org.lwjgl.vulkan.VK10.vkDestroyInstance;
import static org.lwjgl.vulkan.VK10.vkEnumerateDeviceExtensionProperties;
import static org.lwjgl.vulkan.VK10.vkEnumeratePhysicalDevices;
import static org.lwjgl.vulkan.VK10.vkGetDeviceQueue;
import static org.lwjgl.vulkan.VK10.vkGetPhysicalDeviceQueueFamilyProperties;
import static org.lwjgl.vulkan.VK10.vkGetPhysicalDeviceProperties;

/**
 * Minimal Vulkan context for ingestion-harness upload execution.
 */
public final class VulkanHarnessContext implements AutoCloseable {
  public static final int VK_HEADER_VERSION_COMPLETE_DERIVED =
      VK10.VK_MAKE_API_VERSION(0, 1, 0, VK10.VK_HEADER_VERSION);
  public static final int REQUESTED_APP_API_VERSION = VK10.VK_MAKE_API_VERSION(0, 1, 1, 0);
  private final VkInstance instance;
  private final VkPhysicalDevice physicalDevice;
  private final VkDevice device;
  private final VkQueue graphicsQueue;
  private final int graphicsQueueFamilyIndex;
  private final long commandPool;

  private VulkanHarnessContext(
      VkInstance instance,
      VkPhysicalDevice physicalDevice,
      VkDevice device,
      VkQueue graphicsQueue,
      int graphicsQueueFamilyIndex,
      long commandPool) {
    this.instance = instance;
    this.physicalDevice = physicalDevice;
    this.device = device;
    this.graphicsQueue = graphicsQueue;
    this.graphicsQueueFamilyIndex = graphicsQueueFamilyIndex;
    this.commandPool = commandPool;
  }

  public static VulkanHarnessContext create() throws GpuException {
    try (MemoryStack stack = MemoryStack.stackPush()) {
      InstanceSession instanceSession = createInstanceSession(stack, "DynamisGPU Ingestion Harness");
      VkInstance instance = instanceSession.instance();

      DeviceSelection selection =
          pickPhysicalDevice(
              instance,
              stack,
              instanceSession.requestedInstanceExtensions(),
              instanceSession.portabilityEnumerationEnabled());
      VkPhysicalDevice physicalDevice = selection.physicalDevice();
      int queueFamilyIndex = selection.graphicsQueueFamilyIndex();

      float[] priority = {1.0f};
      VkDeviceQueueCreateInfo.Buffer queueInfo = VkDeviceQueueCreateInfo.calloc(1, stack)
          .sType(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO)
          .queueFamilyIndex(queueFamilyIndex)
          .pQueuePriorities(stack.floats(priority));

      List<String> requestedDeviceExtensions = requiredDeviceExtensions(physicalDevice);
      VkDeviceCreateInfo deviceInfo = VkDeviceCreateInfo.calloc(stack)
          .sType(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
          .pQueueCreateInfos(queueInfo)
          .ppEnabledExtensionNames(pointerBufferOfUtf8(stack, requestedDeviceExtensions));

      var pDevice = stack.mallocPointer(1);
      int deviceResult = vkCreateDevice(physicalDevice, deviceInfo, null, pDevice);
      if (deviceResult != VK_SUCCESS) {
        vkDestroyInstance(instance, null);
        throw new GpuException(
            GpuErrorCode.BACKEND_INIT_FAILED,
            "stage=device_create result="
                + deviceResult
                + " selectedDevice="
                + selection.deviceName()
                + " queueFamily="
                + queueFamilyIndex
                + " deviceSummaries="
                + selection.discoveredDeviceSummaries()
                + " requestedDeviceExtensions="
                + requestedDeviceExtensions,
            false);
      }
      VkDevice device = new VkDevice(pDevice.get(0), physicalDevice, deviceInfo);

      var pQueue = stack.mallocPointer(1);
      vkGetDeviceQueue(device, queueFamilyIndex, 0, pQueue);
      VkQueue graphicsQueue = new VkQueue(pQueue.get(0), device);

      var pCommandPool = stack.longs(VK_NULL_HANDLE);
      var poolInfo = org.lwjgl.vulkan.VkCommandPoolCreateInfo.calloc(stack)
          .sType(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO)
          .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
          .queueFamilyIndex(queueFamilyIndex);
      int poolResult = vkCreateCommandPool(device, poolInfo, null, pCommandPool);
      if (poolResult != VK_SUCCESS || pCommandPool.get(0) == VK_NULL_HANDLE) {
        vkDestroyDevice(device, null);
        vkDestroyInstance(instance, null);
        throw new GpuException(
            GpuErrorCode.BACKEND_INIT_FAILED,
            "stage=command_pool_create result="
                + poolResult
                + " selectedDevice="
                + selection.deviceName()
                + " queueFamily="
                + queueFamilyIndex,
            false);
      }

      return new VulkanHarnessContext(
          instance, physicalDevice, device, graphicsQueue, queueFamilyIndex, pCommandPool.get(0));
    }
  }

  public static VulkanProbeReport probe() throws GpuException {
    try (MemoryStack stack = MemoryStack.stackPush()) {
      InstanceSession instanceSession;
      try {
        instanceSession = createInstanceSession(stack, "DynamisGPU Vulkan Probe");
      } catch (GpuException e) {
        InstanceVersionQuery versionQuery = queryInstanceVersion(stack);
        int instanceResult = extractInstanceCreateResultCode(e);
        List<String> requestedInstanceExtensions = requiredInstanceExtensions();
        boolean portabilityEnumerationEnabled =
            requestedInstanceExtensions.contains("VK_KHR_portability_enumeration");
        return new VulkanProbeReport(
            "INSTANCE_CREATE_FAILED",
            instanceResult,
            requestedInstanceExtensions,
            portabilityEnumerationEnabled,
            List.of(),
            "vkCreateInstance returned " + instanceResult,
            VK_HEADER_VERSION_COMPLETE_DERIVED,
            REQUESTED_APP_API_VERSION,
            versionQuery.resultCode(),
            versionQuery.apiVersion(),
            -1);
      }
      VkInstance instance = instanceSession.instance();
      try {
        InstanceVersionQuery versionQuery = queryInstanceVersion(stack);
        int capabilityApiVersion = VK.getInstanceVersionSupported();
        IntBuffer pCount = stack.ints(0);
        int enumResult = vkEnumeratePhysicalDevices(instance, pCount, null);
        if (enumResult != VK_SUCCESS) {
          return new VulkanProbeReport(
              "PHYSICAL_DEVICE_ENUMERATION_FAILED",
              enumResult,
              instanceSession.requestedInstanceExtensions(),
              instanceSession.portabilityEnumerationEnabled(),
              List.of(),
              "vkEnumeratePhysicalDevices(count) returned " + enumResult,
              VK_HEADER_VERSION_COMPLETE_DERIVED,
              REQUESTED_APP_API_VERSION,
              versionQuery.resultCode(),
              versionQuery.apiVersion(),
              capabilityApiVersion);
        }

        PointerBuffer devices = stack.mallocPointer(pCount.get(0));
        int enumListResult = vkEnumeratePhysicalDevices(instance, pCount, devices);
        if (enumListResult != VK_SUCCESS) {
          return new VulkanProbeReport(
              "PHYSICAL_DEVICE_ENUMERATION_LIST_FAILED",
              enumListResult,
              instanceSession.requestedInstanceExtensions(),
              instanceSession.portabilityEnumerationEnabled(),
              List.of(),
              "vkEnumeratePhysicalDevices(list) returned " + enumListResult,
              VK_HEADER_VERSION_COMPLETE_DERIVED,
              REQUESTED_APP_API_VERSION,
              versionQuery.resultCode(),
              versionQuery.apiVersion(),
              capabilityApiVersion);
        }

        List<String> summaries = new ArrayList<>(pCount.get(0));
        for (int i = 0; i < pCount.get(0); i++) {
          VkPhysicalDevice candidate = new VkPhysicalDevice(devices.get(i), instance);
          String name = deviceName(candidate, stack);
          IntBuffer queueCount = stack.ints(0);
          vkGetPhysicalDeviceQueueFamilyProperties(candidate, queueCount, null);
          VkQueueFamilyProperties.Buffer properties =
              VkQueueFamilyProperties.calloc(queueCount.get(0), stack);
          vkGetPhysicalDeviceQueueFamilyProperties(candidate, queueCount, properties);
          int graphicsQueueFamily = -1;
          for (int q = 0; q < properties.capacity(); q++) {
            if ((properties.get(q).queueFlags() & VK_QUEUE_GRAPHICS_BIT) != 0) {
              graphicsQueueFamily = q;
              break;
            }
          }
          summaries.add(
              "name="
                  + name
                  + ",queueFamilies="
                  + properties.capacity()
                  + ",graphicsQueueFamily="
                  + graphicsQueueFamily
                  + ",hasPortabilitySubset="
                  + hasDeviceExtension(candidate, "VK_KHR_portability_subset"));
        }

        return new VulkanProbeReport(
              "SUCCESS",
              VK_SUCCESS,
              instanceSession.requestedInstanceExtensions(),
              instanceSession.portabilityEnumerationEnabled(),
              summaries,
              null,
              VK_HEADER_VERSION_COMPLETE_DERIVED,
            REQUESTED_APP_API_VERSION,
            versionQuery.resultCode(),
            versionQuery.apiVersion(),
            capabilityApiVersion);
      } finally {
        vkDestroyInstance(instance, null);
      }
    }
  }

  public VkPhysicalDevice physicalDevice() {
    return physicalDevice;
  }

  public VkDevice device() {
    return device;
  }

  public VkQueue graphicsQueue() {
    return graphicsQueue;
  }

  public int graphicsQueueFamilyIndex() {
    return graphicsQueueFamilyIndex;
  }

  public long commandPool() {
    return commandPool;
  }

  @Override
  public void close() {
    if (device != null && commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device, commandPool, null);
    }
    if (device != null) {
      vkDestroyDevice(device, null);
    }
    if (instance != null) {
      vkDestroyInstance(instance, null);
    }
  }

  private static DeviceSelection pickPhysicalDevice(
      VkInstance instance,
      MemoryStack stack,
      List<String> requestedInstanceExtensions,
      boolean portabilityEnumerationEnabled)
      throws GpuException {
    IntBuffer pCount = stack.ints(0);
    int countResult = vkEnumeratePhysicalDevices(instance, pCount, null);
    if (countResult != VK_SUCCESS || pCount.get(0) <= 0) {
      throw new GpuException(
          GpuErrorCode.BACKEND_INIT_FAILED,
          "stage=physical_device_enumeration result="
              + countResult
              + " count="
              + pCount.get(0)
              + " requestedInstanceExtensions="
              + requestedInstanceExtensions
              + " portabilityEnumerationEnabled="
              + portabilityEnumerationEnabled
              + " note=no_usable_vulkan_device_or_driver",
          false);
    }
    PointerBuffer devices = stack.mallocPointer(pCount.get(0));
    int enumResult = vkEnumeratePhysicalDevices(instance, pCount, devices);
    if (enumResult != VK_SUCCESS) {
      throw new GpuException(
          GpuErrorCode.BACKEND_INIT_FAILED,
          "stage=physical_device_enumeration_list result=" + enumResult,
          false);
    }

    List<String> deviceNames = new ArrayList<>(pCount.get(0));
    List<String> deviceSummaries = new ArrayList<>(pCount.get(0));
    for (int i = 0; i < pCount.get(0); i++) {
      VkPhysicalDevice candidate = new VkPhysicalDevice(devices.get(i), instance);
      String name = deviceName(candidate, stack);
      deviceNames.add(name);

      IntBuffer queueCount = stack.ints(0);
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, queueCount, null);
      VkQueueFamilyProperties.Buffer properties =
          VkQueueFamilyProperties.calloc(queueCount.get(0), stack);
      vkGetPhysicalDeviceQueueFamilyProperties(candidate, queueCount, properties);
      int graphicsQueueFamily = -1;
      for (int q = 0; q < properties.capacity(); q++) {
        if ((properties.get(q).queueFlags() & VK_QUEUE_GRAPHICS_BIT) != 0) {
          graphicsQueueFamily = q;
          break;
        }
      }
      boolean hasPortabilitySubset = hasDeviceExtension(candidate, "VK_KHR_portability_subset");
      deviceSummaries.add(
          "name="
              + name
              + ",queueFamilies="
              + properties.capacity()
              + ",graphicsQueueFamily="
              + graphicsQueueFamily
              + ",hasPortabilitySubset="
              + hasPortabilitySubset);
      if (graphicsQueueFamily >= 0) {
        return new DeviceSelection(
            candidate, name, graphicsQueueFamily, List.copyOf(deviceNames), List.copyOf(deviceSummaries));
      }
    }
    throw new GpuException(
        GpuErrorCode.BACKEND_INIT_FAILED,
        "stage=queue_family_selection result=no_graphics_queue deviceNames="
            + deviceNames
            + " deviceSummaries="
            + deviceSummaries,
        false);
  }

  private static List<String> requiredInstanceExtensions() {
    List<String> requested = new ArrayList<>();
    if (hasExtension("VK_KHR_portability_enumeration")) {
      requested.add("VK_KHR_portability_enumeration");
    }
    if (hasExtension("VK_EXT_debug_utils")) {
      // Optional during harness bring-up.
      requested.add("VK_EXT_debug_utils");
    }
    return requested;
  }

  private static List<String> requiredDeviceExtensions(VkPhysicalDevice physicalDevice)
      throws GpuException {
    try (MemoryStack stack = MemoryStack.stackPush()) {
      IntBuffer pCount = stack.ints(0);
      int countResult =
          vkEnumerateDeviceExtensionProperties(physicalDevice, (ByteBuffer) null, pCount, null);
      if (countResult != VK_SUCCESS) {
        throw new GpuException(
            GpuErrorCode.BACKEND_INIT_FAILED,
            "stage=device_extension_enumeration result=" + countResult,
            false);
      }
      VkExtensionProperties.Buffer extProps = VkExtensionProperties.calloc(pCount.get(0), stack);
      int enumResult =
          vkEnumerateDeviceExtensionProperties(physicalDevice, (ByteBuffer) null, pCount, extProps);
      if (enumResult != VK_SUCCESS) {
        throw new GpuException(
            GpuErrorCode.BACKEND_INIT_FAILED,
            "stage=device_extension_enumeration_list result=" + enumResult,
            false);
      }
      boolean hasPortabilitySubset = false;
      for (int i = 0; i < extProps.capacity(); i++) {
        if ("VK_KHR_portability_subset".equals(extProps.get(i).extensionNameString())) {
          hasPortabilitySubset = true;
          break;
        }
      }
      List<String> requested = new ArrayList<>();
      if (hasPortabilitySubset) {
        requested.add("VK_KHR_portability_subset");
      }
      return requested;
    }
  }

  private static boolean hasDeviceExtension(VkPhysicalDevice physicalDevice, String extensionName) {
    try (MemoryStack stack = MemoryStack.stackPush()) {
      IntBuffer pCount = stack.ints(0);
      int countResult =
          vkEnumerateDeviceExtensionProperties(physicalDevice, (ByteBuffer) null, pCount, null);
      if (countResult != VK_SUCCESS || pCount.get(0) <= 0) {
        return false;
      }
      VkExtensionProperties.Buffer extProps = VkExtensionProperties.calloc(pCount.get(0), stack);
      int enumResult =
          vkEnumerateDeviceExtensionProperties(physicalDevice, (ByteBuffer) null, pCount, extProps);
      if (enumResult != VK_SUCCESS) {
        return false;
      }
      for (int i = 0; i < extProps.capacity(); i++) {
        if (extensionName.equals(extProps.get(i).extensionNameString())) {
          return true;
        }
      }
      return false;
    }
  }

  private static PointerBuffer pointerBufferOfUtf8(MemoryStack stack, List<String> values) {
    PointerBuffer names = stack.mallocPointer(values.size());
    for (String value : values) {
      names.put(stack.UTF8(value));
    }
    names.flip();
    return names;
  }

  private static String deviceName(VkPhysicalDevice device, MemoryStack stack) {
    VkPhysicalDeviceProperties properties = VkPhysicalDeviceProperties.calloc(stack);
    vkGetPhysicalDeviceProperties(device, properties);
    return properties.deviceNameString();
  }

  private static boolean hasExtension(String extensionName) {
    try (MemoryStack stack = MemoryStack.stackPush()) {
      IntBuffer pCount = stack.ints(0);
      int countResult = VK10.vkEnumerateInstanceExtensionProperties((String) null, pCount, null);
      if (countResult != VK_SUCCESS || pCount.get(0) <= 0) {
        return false;
      }
      VkExtensionProperties.Buffer extensions =
          VkExtensionProperties.calloc(pCount.get(0), stack);
      int enumResult = VK10.vkEnumerateInstanceExtensionProperties((String) null, pCount, extensions);
      if (enumResult != VK_SUCCESS) {
        return false;
      }
      for (int i = 0; i < extensions.capacity(); i++) {
        if (extensionName.equals(extensions.get(i).extensionNameString())) {
          return true;
        }
      }
      return false;
    }
  }

  private record DeviceSelection(
      VkPhysicalDevice physicalDevice,
      String deviceName,
      int graphicsQueueFamilyIndex,
      List<String> discoveredDeviceNames,
      List<String> discoveredDeviceSummaries) {}

  private static InstanceSession createInstanceSession(MemoryStack stack, String applicationName)
      throws GpuException {
    List<InstanceCreateAttempt> attempts = buildInstanceCreateAttempts();
    List<String> attemptResults = new ArrayList<>(attempts.size());
    for (InstanceCreateAttempt attempt : attempts) {
      VkApplicationInfo appInfo = VkApplicationInfo.calloc(stack)
          .sType(VK_STRUCTURE_TYPE_APPLICATION_INFO)
          .pApplicationName(stack.UTF8(applicationName))
          .pEngineName(stack.UTF8("DynamisEngine"))
          .apiVersion(attempt.apiVersion());

      VkInstanceCreateInfo instanceInfo = VkInstanceCreateInfo.calloc(stack)
          .sType(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
          .pApplicationInfo(appInfo);
      if (!attempt.extensions().isEmpty()) {
        instanceInfo.ppEnabledExtensionNames(pointerBufferOfUtf8(stack, attempt.extensions()));
      }
      if (!attempt.layers().isEmpty()) {
        instanceInfo.ppEnabledLayerNames(pointerBufferOfUtf8(stack, attempt.layers()));
      }
      if (attempt.portabilityEnumerationEnabled()) {
        instanceInfo.flags(VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR);
      }
      logInstanceAttempt(attempt);

      var pInstance = stack.mallocPointer(1);
      int instanceResult = vkCreateInstance(instanceInfo, null, pInstance);
      attemptResults.add(attempt.label() + "=" + instanceResult);
      if (instanceResult == VK_SUCCESS) {
        VkInstance instance = new VkInstance(pInstance.get(0), instanceInfo);
        return new InstanceSession(
            instance,
            List.copyOf(attempt.extensions()),
            attempt.portabilityEnumerationEnabled(),
            List.copyOf(attempt.layers()),
            attempt.apiVersion(),
            attempt.label(),
            List.copyOf(attemptResults));
      }
    }

    throw new GpuException(
        GpuErrorCode.BACKEND_INIT_FAILED,
        "stage=instance_create result="
            + extractFinalResult(attemptResults)
            + " requestedInstanceExtensions="
            + attempts.get(0).extensions()
            + " requestedInstanceLayers="
            + attempts.get(0).layers()
            + " attemptResults="
            + attemptResults,
        false);
  }

  private static int extractInstanceCreateResultCode(GpuException exception) {
    String message = exception.getMessage();
    if (message == null) {
      return -1;
    }
    int marker = message.indexOf("result=");
    if (marker < 0) {
      return -1;
    }
    int start = marker + "result=".length();
    int end = start;
    while (end < message.length()) {
      char c = message.charAt(end);
      if (!(c == '-' || Character.isDigit(c))) {
        break;
      }
      end++;
    }
    if (end == start) {
      return -1;
    }
    try {
      return Integer.parseInt(message.substring(start, end));
    } catch (NumberFormatException ignored) {
      return -1;
    }
  }

  private static List<InstanceCreateAttempt> buildInstanceCreateAttempts() {
    List<String> baseExtensions = requiredInstanceExtensions();
    List<String> baseLayers = requiredInstanceLayers();
    List<String> withoutDebugUtils = stripExtension(baseExtensions, "VK_EXT_debug_utils");
    List<String> withoutValidationLayers = stripValidationLayers(baseLayers);
    int apiVersion = REQUESTED_APP_API_VERSION;

    List<InstanceCreateAttempt> attempts = new ArrayList<>(3);
    attempts.add(new InstanceCreateAttempt("A", apiVersion, baseExtensions, baseLayers));
    attempts.add(new InstanceCreateAttempt("B", apiVersion, withoutDebugUtils, baseLayers));
    attempts.add(new InstanceCreateAttempt("C", apiVersion, withoutDebugUtils, withoutValidationLayers));
    return attempts;
  }

  private static List<String> requiredInstanceLayers() {
    List<String> requested = new ArrayList<>();
    if (isDebugRuntimeEnabled() && hasInstanceLayer("VK_LAYER_KHRONOS_validation")) {
      requested.add("VK_LAYER_KHRONOS_validation");
    }
    return requested;
  }

  private static boolean isDebugRuntimeEnabled() {
    String lwjglDebug = System.getProperty("org.lwjgl.util.Debug");
    if (lwjglDebug != null && Boolean.parseBoolean(lwjglDebug)) {
      return true;
    }
    String explicit = System.getProperty("dynamisgpu.vk.debug");
    if (explicit != null && Boolean.parseBoolean(explicit)) {
      return true;
    }
    String env = System.getenv("DYNAMISGPU_VK_DEBUG");
    return env != null && Boolean.parseBoolean(env);
  }

  private static List<String> stripExtension(List<String> source, String extension) {
    List<String> stripped = new ArrayList<>(source.size());
    for (String value : source) {
      if (!extension.equals(value)) {
        stripped.add(value);
      }
    }
    return stripped;
  }

  private static List<String> stripValidationLayers(List<String> source) {
    List<String> stripped = new ArrayList<>(source.size());
    for (String value : source) {
      String lower = value.toLowerCase(Locale.ROOT);
      if (!lower.contains("validation") && !lower.contains("debug")) {
        stripped.add(value);
      }
    }
    return stripped;
  }

  private static void logInstanceAttempt(InstanceCreateAttempt attempt) {
    int flags = attempt.portabilityEnumerationEnabled() ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0;
    System.out.println(
        "[VulkanHarnessContext] instanceAttempt="
            + attempt.label()
            + " apiVersion="
            + apiVersionToString(attempt.apiVersion())
            + " extensions="
            + attempt.extensions()
            + " layers="
            + attempt.layers()
            + " flags="
            + flags);
  }

  private static String apiVersionToString(int apiVersion) {
    return "variant="
        + VK10.VK_API_VERSION_VARIANT(apiVersion)
        + ",major="
        + VK10.VK_API_VERSION_MAJOR(apiVersion)
        + ",minor="
        + VK10.VK_API_VERSION_MINOR(apiVersion)
        + ",patch="
        + VK10.VK_API_VERSION_PATCH(apiVersion)
        + " ("
        + apiVersion
        + ")";
  }

  private static int extractFinalResult(List<String> attemptResults) {
    if (attemptResults.isEmpty()) {
      return -1;
    }
    String last = attemptResults.get(attemptResults.size() - 1);
    int idx = last.indexOf('=');
    if (idx < 0 || idx + 1 >= last.length()) {
      return -1;
    }
    try {
      return Integer.parseInt(last.substring(idx + 1));
    } catch (NumberFormatException ignored) {
      return -1;
    }
  }

  private static boolean hasInstanceLayer(String layerName) {
    try (MemoryStack stack = MemoryStack.stackPush()) {
      IntBuffer pCount = stack.ints(0);
      int countResult = VK10.vkEnumerateInstanceLayerProperties(pCount, null);
      if (countResult != VK_SUCCESS || pCount.get(0) <= 0) {
        return false;
      }
      var layers = org.lwjgl.vulkan.VkLayerProperties.calloc(pCount.get(0), stack);
      int enumResult = VK10.vkEnumerateInstanceLayerProperties(pCount, layers);
      if (enumResult != VK_SUCCESS) {
        return false;
      }
      for (int i = 0; i < layers.capacity(); i++) {
        if (layerName.equals(layers.get(i).layerNameString())) {
          return true;
        }
      }
      return false;
    }
  }

  public record VulkanProbeReport(
      String stage,
      int resultCode,
      List<String> requestedInstanceExtensions,
      boolean portabilityEnumerationEnabled,
      List<String> discoveredDeviceSummaries,
      String detail,
      int vkHeaderVersionComplete,
      int requestedAppApiVersion,
      int enumerateInstanceVersionResult,
      int enumerateInstanceApiVersion,
      int capabilitiesApiVersion) {}

  private static InstanceVersionQuery queryInstanceVersion(MemoryStack stack) {
    IntBuffer pVersion = stack.ints(0);
    try {
      int result = VK11.vkEnumerateInstanceVersion(pVersion);
      return new InstanceVersionQuery(result, pVersion.get(0));
    } catch (Throwable t) {
      return new InstanceVersionQuery(-1, 0);
    }
  }

  private record InstanceVersionQuery(int resultCode, int apiVersion) {}

  private record InstanceSession(
      VkInstance instance,
      List<String> requestedInstanceExtensions,
      boolean portabilityEnumerationEnabled,
      List<String> requestedInstanceLayers,
      int apiVersion,
      String successfulAttemptLabel,
      List<String> attemptResults) {}

  private record InstanceCreateAttempt(
      String label,
      int apiVersion,
      List<String> extensions,
      List<String> layers) {
    boolean portabilityEnumerationEnabled() {
      return extensions.contains("VK_KHR_portability_enumeration");
    }
  }
}
