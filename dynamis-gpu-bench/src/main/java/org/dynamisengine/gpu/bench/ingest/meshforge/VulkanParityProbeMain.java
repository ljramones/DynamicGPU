package org.dynamisengine.gpu.bench.ingest.meshforge;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.dynamisengine.gpu.api.error.GpuException;
import org.lwjgl.vulkan.VK10;

/**
 * Prints process/runtime parity diagnostics and runs the Vulkan probe.
 */
public final class VulkanParityProbeMain {
  private static final List<String> TRACKED_ENV_KEYS =
      List.of(
          "VK_DRIVER_FILES",
          "VK_ICD_FILENAMES",
          "VK_ADD_DRIVER_FILES",
          "VK_LAYER_PATH",
          "VK_ADD_LAYER_PATH",
          "DYLD_LIBRARY_PATH",
          "PATH");

  private VulkanParityProbeMain() {}

  public static void main(String[] args) {
    boolean debug = containsArg(args, "--debug");
    printParitySnapshot();
    VulkanLoaderBootstrap.bootstrap(debug);
    runProbe();
  }

  private static void printParitySnapshot() {
    System.out.println(
        "[VulkanParityProbe] cwd=" + Path.of(".").toAbsolutePath().normalize());
    System.out.println(
        "[VulkanParityProbe] loaderCandidate="
            + VulkanLoaderBootstrap.selectedLoaderPathForCurrentOs()
            + " loaderOverrideProperty="
            + System.getProperty("dynamisgpu.vk.loader"));
    System.out.println(
        "[VulkanParityProbe] java.library.path=" + System.getProperty("java.library.path"));
    System.out.println(
        "[VulkanParityProbe] org.lwjgl.librarypath=" + System.getProperty("org.lwjgl.librarypath"));
    System.out.println(
        "[VulkanParityProbe] org.lwjgl.vulkan.libname=" + System.getProperty("org.lwjgl.vulkan.libname"));

    for (String key : TRACKED_ENV_KEYS) {
      String value = System.getenv(key);
      System.out.println("[VulkanParityProbe] env." + key + "=" + value);
      if (isIcdEnv(key) && value != null && !value.isBlank()) {
        for (String manifest : splitEnvPathList(value)) {
          Path path = Path.of(manifest);
          System.out.println(
              "[VulkanParityProbe] manifestCandidate="
                  + path
                  + " exists="
                  + Files.isRegularFile(path));
        }
      }
    }
    System.out.println(
        "[VulkanParityProbe] note=ICD selection is loader env-driven. In-process Java cannot reliably mutate process env; prefer shell-exported VK_DRIVER_FILES/VK_ICD_FILENAMES.");
  }

  private static void runProbe() {
    try {
      VulkanHarnessContext.VulkanProbeReport report = VulkanHarnessContext.probe();
      System.out.println(
          "[VulkanParityProbe] probeStage="
              + report.stage()
              + " resultCode="
              + report.resultCode()
              + " requestedAppApiVersion="
              + apiVersionString(report.requestedAppApiVersion())
              + " enumerateInstanceVersionResult="
              + report.enumerateInstanceVersionResult()
              + " enumerateInstanceApiVersion="
              + apiVersionString(report.enumerateInstanceApiVersion())
              + " portabilityEnumerationEnabled="
              + report.portabilityEnumerationEnabled()
              + " requestedInstanceExtensions="
              + report.requestedInstanceExtensions()
              + " discoveredDevices="
              + report.discoveredDeviceSummaries()
              + " detail="
              + report.detail());
    } catch (GpuException e) {
      System.out.println("[VulkanParityProbe] probeStage=EXCEPTION detail=" + e.getMessage());
    }
  }

  private static boolean isIcdEnv(String key) {
    return "VK_DRIVER_FILES".equals(key)
        || "VK_ICD_FILENAMES".equals(key)
        || "VK_ADD_DRIVER_FILES".equals(key);
  }

  private static List<String> splitEnvPathList(String value) {
    List<String> parts = new ArrayList<>();
    for (String token : value.split(":")) {
      if (!token.isBlank()) {
        parts.add(token.trim());
      }
    }
    return parts;
  }

  private static boolean containsArg(String[] args, String target) {
    for (String arg : args) {
      if (target.equals(arg)) {
        return true;
      }
    }
    return false;
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
