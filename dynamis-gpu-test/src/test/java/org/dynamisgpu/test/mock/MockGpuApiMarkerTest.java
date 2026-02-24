package org.dynamisgpu.test.mock;

import static org.junit.jupiter.api.Assertions.assertNotNull;

import org.junit.jupiter.api.Test;

class MockGpuApiMarkerTest {

  @Test
  void createsMockMarker() {
    assertNotNull(new MockGpuApiMarker());
  }
}
