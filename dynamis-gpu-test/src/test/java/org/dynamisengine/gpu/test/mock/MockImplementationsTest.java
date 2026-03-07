package org.dynamisengine.gpu.test.mock;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class MockImplementationsTest {

  @Test
  void recordsCoreCalls() {
    MockVkCommandBuffer cmd = new MockVkCommandBuffer(77L);

    MockComputeCullingDispatch culling = new MockComputeCullingDispatch();
    culling.dispatch(cmd, 32, new float[] {1, 2, 3, 4, 5, 6});
    assertEquals(1, culling.calls().size());
    assertEquals(32, culling.calls().getFirst().meshCount());

    MockDrawMetaWriter drawMeta = new MockDrawMetaWriter(128);
    drawMeta.write(1, 2, 3, 4, 5, 6, 7, 8);
    drawMeta.flush();
    assertEquals(128, drawMeta.capacity());
    assertEquals(1, drawMeta.writes().size());
    assertEquals(1, drawMeta.flushCount());

    MockIndirectCommandBuffer indirect =
        new MockIndirectCommandBuffer(10L, 11L, new int[] {0, 64}, new int[] {64, 64});
    indirect.writeCommand(0, 12, 1, 0, 0, 0);
    assertEquals(10L, indirect.bufferHandle());
    assertEquals(11L, indirect.countBufferHandle());
    assertEquals(1, indirect.writes().size());

    MockWeightBuffer weights = new MockWeightBuffer(40L);
    weights.upload(new float[] {0.1f, 0.2f});
    assertArrayEquals(new float[] {0.1f, 0.2f}, weights.uploads().getFirst());

    MockStagingScheduler scheduler = new MockStagingScheduler();
    scheduler.markDirty(100L, 16L, 32L);
    assertEquals(1, scheduler.dirtyRanges().size());
    scheduler.flush(cmd);
    assertTrue(scheduler.dirtyRanges().isEmpty());
    assertEquals(1, scheduler.flushCount());
  }
}
