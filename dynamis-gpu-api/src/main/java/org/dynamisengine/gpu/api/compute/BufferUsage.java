package org.dynamisengine.gpu.api.compute;
/** How a buffer may be used. {@code INDIRECT} (1.6.0): it holds group counts for indirect dispatches. */
public enum BufferUsage { STORAGE, UNIFORM, TRANSFER_SOURCE, TRANSFER_DESTINATION, MAP_READ, MAP_WRITE, INDIRECT }
