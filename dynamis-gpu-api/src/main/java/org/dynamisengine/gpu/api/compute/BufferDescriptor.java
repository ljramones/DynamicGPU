package org.dynamisengine.gpu.api.compute;
import java.util.Set;
public record BufferDescriptor(long bytes, Set<BufferUsage> usages, Placement placement) {
    public BufferDescriptor {
        usages = Set.copyOf(usages);
        if (bytes <= 0 || usages.isEmpty() || placement == null) throw new IllegalArgumentException("Positive size, usages and placement required");
        if (placement == Placement.PRIVATE && (usages.contains(BufferUsage.MAP_READ) || usages.contains(BufferUsage.MAP_WRITE)))
            throw new IllegalArgumentException("Private buffers cannot be mapped");
    }
}
