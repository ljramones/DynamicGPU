package org.dynamisengine.gpu.api.resource;

/**
 * Optional runtime payload compression mode at the MeshForge -> DynamisGPU ingestion boundary.
 */
public enum RuntimePayloadCompressionMode {
  /** Uncompressed payload bytes. */
  NONE,
  /** Deflate-compressed payload bytes. */
  DEFLATE
}

