package org.dynamisengine.gpu.api.compute;

/**
 * The live buffers of one memory class (the label in their {@link BufferDescriptor}) and the bytes they asked for.
 *
 * @since 1.6.0
 */
public record MemoryUse(long buffers, long bytes) { }
