package org.dynamisengine.gpu.api.compute;
import java.util.Set;
/**
 * A buffer to allocate.
 *
 * @param label the buffer's memory class, for {@link GpuDevice#memoryByLabel()} (for example {@code "kv"} or
 *        {@code "textures"}); empty for none (1.6.0)
 */
public record BufferDescriptor(long bytes, Set<BufferUsage> usages, Placement placement, String label) {
    /** An unlabelled buffer. */
    public BufferDescriptor(long bytes, Set<BufferUsage> usages, Placement placement) { this(bytes, usages, placement, ""); }
    /** This descriptor with memory class {@code label}. */
    public BufferDescriptor withLabel(String label) { return new BufferDescriptor(bytes, usages, placement, label); }
    public BufferDescriptor {
        label = label == null ? "" : label;
        usages = Set.copyOf(usages);
        if (bytes <= 0 || usages.isEmpty() || placement == null) throw new IllegalArgumentException("Positive size, usages and placement required");
        if (placement == Placement.PRIVATE && (usages.contains(BufferUsage.MAP_READ) || usages.contains(BufferUsage.MAP_WRITE)))
            throw new IllegalArgumentException("Private buffers cannot be mapped");
    }
}
