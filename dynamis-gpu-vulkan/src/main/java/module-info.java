module org.dynamisengine.gpu.vulkan {
    requires transitive org.dynamisengine.gpu.api;
    requires java.logging;
    requires org.lwjgl;
    requires org.lwjgl.vulkan;

    exports org.dynamisengine.gpu.vulkan;
    exports org.dynamisengine.gpu.vulkan.buffer;
    exports org.dynamisengine.gpu.vulkan.compute;
    exports org.dynamisengine.gpu.vulkan.descriptor;
    exports org.dynamisengine.gpu.vulkan.memory;
    exports org.dynamisengine.gpu.vulkan.sync;
    exports org.dynamisengine.gpu.vulkan.upload;
}
