package org.dynamisengine.gpu.bench.ingest.meshforge;

import org.lwjgl.system.Configuration;
import org.lwjgl.vulkan.VK;

import java.nio.file.Files;
import java.nio.file.Path;
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
    String selected = null;
    if (lowerOs.contains("mac")) {
      selected = selectMacLoaderPath();
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
                  ? selectMacLoaderPath()
                  : System.getProperty("org.lwjgl.vulkan.libname")));
      System.err.println(
          "[VulkanLoaderBootstrap] java.library.path=" + System.getProperty("java.library.path"));
      System.err.println(
          "[VulkanLoaderBootstrap] org.lwjgl.librarypath="
              + System.getProperty("org.lwjgl.librarypath"));
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
    return selectMacLoaderPath();
  }

  private static boolean resolveDebugRequested() {
    String property = System.getProperty("dynamisgpu.vk.debug");
    if (property != null) {
      return Boolean.parseBoolean(property);
    }
    String env = System.getenv("DYNAMISGPU_VK_DEBUG");
    return env != null && Boolean.parseBoolean(env);
  }

  private static String selectMacLoaderPath() {
    if (Files.isRegularFile(Path.of(MACOS_PRIMARY_LOADER))) {
      return MACOS_PRIMARY_LOADER;
    }
    if (Files.isRegularFile(Path.of(MACOS_SECONDARY_LOADER))) {
      return MACOS_SECONDARY_LOADER;
    }
    return null;
  }
}
