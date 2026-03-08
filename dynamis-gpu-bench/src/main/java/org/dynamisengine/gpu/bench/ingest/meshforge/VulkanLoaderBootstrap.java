package org.dynamisengine.gpu.bench.ingest.meshforge;

import org.lwjgl.system.Configuration;
import org.lwjgl.vulkan.VK;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Bootstraps Vulkan native loader discovery for bench entrypoints.
 */
public final class VulkanLoaderBootstrap {
  private static final String MACOS_PRIMARY_LOADER = "/usr/local/lib/libvulkan.1.dylib";
  private static final String MACOS_SECONDARY_LOADER = "/usr/local/lib/libvulkan.dylib";

  private VulkanLoaderBootstrap() {}

  public static void bootstrap() {
    bootstrap(resolveDebugRequested());
  }

  public static void bootstrap(boolean enableDebug) {
    if (enableDebug) {
      // Must be set before LWJGL native loader initialization to be effective.
      System.setProperty("org.lwjgl.util.Debug", "true");
      System.setProperty("org.lwjgl.util.DebugLoader", "true");
    }

    String osName = System.getProperty("os.name", "unknown");
    String lowerOs = osName.toLowerCase(Locale.ROOT);
    String selected = System.getProperty("dynamisgpu.vk.loader");
    if (lowerOs.contains("mac")) {
      if (selected == null || selected.isBlank()) {
        selected = selectMacLoaderPathFromSdkOrSystem();
      }
      if (selected != null) {
        Configuration.VULKAN_LIBRARY_NAME.set(selected);
      }
    }

    // Ensure LWJGL does not implicitly initialize Vulkan before loader config is set.
    Configuration.VULKAN_EXPLICIT_INIT.set(true);

    try {
      System.out.println(
          "[VulkanLoaderBootstrap] os="
              + osName
              + " debug="
              + enableDebug
              + " selectedLoader="
              + selected
              + " VULKAN_SDK="
              + System.getenv("VULKAN_SDK")
              + " VK_DRIVER_FILES="
              + System.getenv("VK_DRIVER_FILES")
              + " VK_ICD_FILENAMES="
              + System.getenv("VK_ICD_FILENAMES")
              + " VK_ADD_LAYER_PATH="
              + System.getenv("VK_ADD_LAYER_PATH")
              + " DYLD_LIBRARY_PATH="
              + System.getenv("DYLD_LIBRARY_PATH")
              + " org.lwjgl.vulkan.libname="
              + System.getProperty("org.lwjgl.vulkan.libname")
              + " java.library.path="
              + System.getProperty("java.library.path")
              + " org.lwjgl.librarypath="
              + System.getProperty("org.lwjgl.librarypath"));
      VK.create();
    } catch (Throwable t) {
      System.err.println("[VulkanLoaderBootstrap] Vulkan initialization failed");
      System.err.println("[VulkanLoaderBootstrap] os.name=" + osName);
      System.err.println(
          "[VulkanLoaderBootstrap] org.lwjgl.vulkan.libname="
              + System.getProperty("org.lwjgl.vulkan.libname"));
      System.err.println(
          "[VulkanLoaderBootstrap] configuredLoader="
              + (System.getProperty("org.lwjgl.vulkan.libname") == null
                  ? selectMacLoaderPathFromSdkOrSystem()
                  : System.getProperty("org.lwjgl.vulkan.libname")));
      System.err.println(
          "[VulkanLoaderBootstrap] java.library.path=" + System.getProperty("java.library.path"));
      System.err.println(
          "[VulkanLoaderBootstrap] org.lwjgl.librarypath="
              + System.getProperty("org.lwjgl.librarypath"));
      System.err.println("[VulkanLoaderBootstrap] VULKAN_SDK=" + System.getenv("VULKAN_SDK"));
      System.err.println("[VulkanLoaderBootstrap] VK_DRIVER_FILES=" + System.getenv("VK_DRIVER_FILES"));
      System.err.println("[VulkanLoaderBootstrap] VK_ICD_FILENAMES=" + System.getenv("VK_ICD_FILENAMES"));
      System.err.println("[VulkanLoaderBootstrap] VK_ADD_LAYER_PATH=" + System.getenv("VK_ADD_LAYER_PATH"));
      System.err.println("[VulkanLoaderBootstrap] DYLD_LIBRARY_PATH=" + System.getenv("DYLD_LIBRARY_PATH"));
      if (lowerOs.contains("mac")) {
        System.err.println(
            "[VulkanLoaderBootstrap] hint: set DYLD_LIBRARY_PATH to Vulkan loader location, e.g. /usr/local/lib");
      }
      throw t;
    }
  }

  public static String selectedLoaderPathForCurrentOs() {
    String osName = System.getProperty("os.name", "unknown").toLowerCase(Locale.ROOT);
    if (!osName.contains("mac")) {
      return null;
    }
    return selectMacLoaderPathFromSdkOrSystem();
  }

  private static boolean resolveDebugRequested() {
    String property = System.getProperty("dynamisgpu.vk.debug");
    if (property != null) {
      return Boolean.parseBoolean(property);
    }
    String env = System.getenv("DYNAMISGPU_VK_DEBUG");
    return env != null && Boolean.parseBoolean(env);
  }

  private static String selectMacLoaderPathFromSdkOrSystem() {
    List<String> candidates = new ArrayList<>();
    String vulkanSdk = System.getenv("VULKAN_SDK");
    if (vulkanSdk != null && !vulkanSdk.isBlank()) {
      candidates.add(Path.of(vulkanSdk, "lib", "libvulkan.1.dylib").toString());
      candidates.add(Path.of(vulkanSdk, "lib", "libvulkan.dylib").toString());
    }
    candidates.add(MACOS_PRIMARY_LOADER);
    candidates.add(MACOS_SECONDARY_LOADER);

    for (String candidate : candidates) {
      if (Files.isRegularFile(Path.of(candidate))) {
        return candidate;
      }
    }
    return null;
  }
}
