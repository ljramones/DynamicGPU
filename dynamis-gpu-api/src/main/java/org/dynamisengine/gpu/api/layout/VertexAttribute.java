package org.dynamisengine.gpu.api.layout;

import java.util.Objects;

/**
 * One vertex attribute declaration within a vertex layout.
 *
 * @param location shader attribute location
 * @param offsetBytes attribute byte offset in one vertex record
 * @param format storage format
 */
public record VertexAttribute(int location, int offsetBytes, VertexFormat format) {
  public VertexAttribute {
    if (location < 0) {
      throw new IllegalArgumentException("location must be >= 0");
    }
    if (offsetBytes < 0) {
      throw new IllegalArgumentException("offsetBytes must be >= 0");
    }
    Objects.requireNonNull(format, "format");
  }
}
