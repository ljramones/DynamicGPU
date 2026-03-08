package org.dynamisengine.gpu.api.layout;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Vertex record layout declaration.
 *
 * @param strideBytes byte stride for one vertex
 * @param attributes declared attributes for one vertex record
 */
public record VertexLayout(int strideBytes, List<VertexAttribute> attributes) {
  public VertexLayout {
    if (strideBytes <= 0) {
      throw new IllegalArgumentException("strideBytes must be > 0");
    }
    Objects.requireNonNull(attributes, "attributes");
    if (attributes.isEmpty()) {
      throw new IllegalArgumentException("attributes must not be empty");
    }

    Set<Integer> locations = new HashSet<>(attributes.size());
    for (VertexAttribute attribute : attributes) {
      Objects.requireNonNull(attribute, "attributes must not contain null");
      if (!locations.add(attribute.location())) {
        throw new IllegalArgumentException("duplicate attribute location: " + attribute.location());
      }
      int attributeEnd = attribute.offsetBytes() + attribute.format().byteSize();
      if (attributeEnd > strideBytes) {
        throw new IllegalArgumentException(
            "attribute exceeds stride: location=" + attribute.location() + ", end=" + attributeEnd + ", stride=" + strideBytes);
      }
    }
    attributes = List.copyOf(attributes);
  }
}
