package org.dynamisengine.gpu.api.compute;
import java.util.List;
import java.util.HashSet;
/** Storage bindings occupy explicit slots. Scalar parameters use a separate ABI slot. */
public record BindingLayout(List<BufferBinding> buffers, int parameterBytes) {
    public record BufferBinding(int slot, Access access, long minimumBytes) {
        public BufferBinding {
            if (slot < 0 || access == null || minimumBytes <= 0) throw new IllegalArgumentException("Invalid buffer binding");
        }
    }
    public BindingLayout {
        buffers = List.copyOf(buffers);
        var slots = new HashSet<Integer>();
        for (var b : buffers) if (!slots.add(b.slot())) throw new IllegalArgumentException("Duplicate binding " + b.slot());
        if (parameterBytes < 0 || parameterBytes % 4 != 0) throw new IllegalArgumentException("Parameter size must be a nonnegative multiple of four");
    }
    /** Metal setBytes slot; Vulkan parameters are push constants at offset zero. */
    public int parameterSlot() { return Math.addExact(buffers.stream().mapToInt(BufferBinding::slot).max().orElse(-1), 1); }
}
