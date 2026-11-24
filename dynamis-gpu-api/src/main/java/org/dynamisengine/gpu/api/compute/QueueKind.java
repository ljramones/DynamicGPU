package org.dynamisengine.gpu.api.compute;

/**
 * What a queue runs. {@link #COMPUTE} runs everything; {@link #TRANSFER} runs uploads, downloads, copies and fills,
 * on the device's transfer engine where it has one, so streaming overlaps compute. Order work across queues with
 * {@link GpuFence}s.
 *
 * @since 1.6.0
 */
public enum QueueKind { COMPUTE, TRANSFER }
