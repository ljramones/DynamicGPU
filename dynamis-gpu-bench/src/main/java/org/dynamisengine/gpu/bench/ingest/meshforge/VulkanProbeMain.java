package org.dynamisengine.gpu.bench.ingest.meshforge;

import org.dynamisengine.gpu.api.error.GpuException;
import org.lwjgl.vulkan.VK10;

/**
 * Minimal Vulkan probe entrypoint for diagnosing loader/runtime/device discovery.
 */
public final class VulkanProbeMain {
  private VulkanProbeMain() {}

  public static void main(String[] args) {
    boolean debug = args.length > 0 && "--debug".equals(args[0]);
    VulkanLoaderBootstrap.bootstrap(debug);

    try {
      VulkanHarnessContext.VulkanProbeReport report = VulkanHarnessContext.probe();
      System.out.println(
          "probeStage="
              + report.stage()
              + " resultCode="
              + report.resultCode()
              + " vkHeaderVersionComplete="
              + report.vkHeaderVersionComplete()
              + " requestedAppApiVersion="
              + apiVersionString(report.requestedAppApiVersion())
              + " enumerateInstanceVersionResult="
              + report.enumerateInstanceVersionResult()
              + " enumerateInstanceApiVersion="
              + apiVersionString(report.enumerateInstanceApiVersion())
              + " instanceVersionSupported="
              + apiVersionString(report.capabilitiesApiVersion())
              + " portabilityEnumerationEnabled="
              + report.portabilityEnumerationEnabled()
              + " requestedInstanceExtensions="
              + report.requestedInstanceExtensions()
              + " discoveredDevices="
              + report.discoveredDeviceSummaries()
              + " detail="
              + report.detail());
    } catch (GpuException e) {
      System.out.println(
          "probeStage=EXCEPTION resultCode=NA detail=" + e.getMessage());
    }
  }

  private static String apiVersionString(int apiVersion) {
    if (apiVersion <= 0) {
      return "NA";
    }
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
}
