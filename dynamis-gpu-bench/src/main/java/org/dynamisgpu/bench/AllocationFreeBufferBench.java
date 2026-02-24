package org.dynamisgpu.bench;

import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;

/**
 * Placeholder benchmark to keep JMH wiring valid while implementations are added.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
public class AllocationFreeBufferBench {

  @Benchmark
  public int baseline() {
    return 1;
  }
}
