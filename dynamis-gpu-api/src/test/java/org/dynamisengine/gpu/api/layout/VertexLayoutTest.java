package org.dynamisengine.gpu.api.layout;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.List;
import org.junit.jupiter.api.Test;

class VertexLayoutTest {
  private static final VertexFormat FORMAT = VertexFormat.FLOAT1;

  @Test
  void acceptsAttributesInsideTheStride() {
    int size = FORMAT.byteSize();
    var layout = new VertexLayout(2 * size, List.of(new VertexAttribute(0, 0, FORMAT), new VertexAttribute(1, size, FORMAT)));
    assertEquals(2, layout.attributes().size());
  }

  @Test
  void rejectsAnAttributePastTheStride() {
    int size = FORMAT.byteSize();
    assertThrows(IllegalArgumentException.class,
        () -> new VertexLayout(size, List.of(new VertexAttribute(0, 1, FORMAT))));
  }

  /** offset + size used to overflow int and pass the stride check. */
  @Test
  void rejectsAnOffsetWhoseEndOverflowsInt() {
    assertThrows(IllegalArgumentException.class,
        () -> new VertexLayout(16, List.of(new VertexAttribute(0, Integer.MAX_VALUE - 2, FORMAT))));
  }
}
