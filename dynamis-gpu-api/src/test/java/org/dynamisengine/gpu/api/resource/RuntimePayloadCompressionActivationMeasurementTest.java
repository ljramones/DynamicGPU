package org.dynamisengine.gpu.api.resource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/**
 * Focused measurement pass for optional compressed payload activation.
 *
 * <p>This is intentionally lightweight and compares ingestion-boundary behavior only:
 * uncompressed ingestion versus compressed ingestion with DEFLATE.
 */
class RuntimePayloadCompressionActivationMeasurementTest {
  private static final int WARMUP_RUNS = 8;
  private static final int MEASURED_RUNS = 25;

  @Test
  void measureVisibilityAndStreamingCompressionActivation() {
    VisibilityScenario visibility = buildVisibilityScenario(20_000);
    StreamingScenario streaming = buildStreamingScenario(2_048);

    MeasurementResult visibilityResult = measureVisibility(visibility);
    MeasurementResult streamingResult = measureStreaming(streaming);

    System.out.println(visibilityResult.format());
    System.out.println(streamingResult.format());

    // Basic sanity checks only; this is a measurement test, not a strict perf gate.
    assertTrue(visibilityResult.compressionRatio() < 1.0);
    assertTrue(streamingResult.compressionRatio() < 1.0);
    assertTrue(visibilityResult.compressedMedianMicros() > 0);
    assertTrue(streamingResult.compressedMedianMicros() > 0);
  }

  private static MeasurementResult measureVisibility(VisibilityScenario scenario) {
    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
      MeshletBoundsPayloadIngestion.ingest(
          scenario.meshletCount,
          scenario.offsetFloats,
          scenario.strideFloats,
          scenario.floatCount,
          scenario.expectedFloatCount,
          scenario.canonicalBytes.duplicate().order(ByteOrder.LITTLE_ENDIAN));
      MeshletBoundsPayloadIngestion.ingestCompressed(
          scenario.meshletCount,
          scenario.offsetFloats,
          scenario.strideFloats,
          scenario.floatCount,
          scenario.expectedFloatCount,
          scenario.compressedPayload);
    }

    long[] uncompressedNanos = new long[MEASURED_RUNS];
    long[] compressedNanos = new long[MEASURED_RUNS];
    for (int i = 0; i < MEASURED_RUNS; i++) {
      long t0 = System.nanoTime();
      GpuMeshletBoundsPayload uncompressed =
          MeshletBoundsPayloadIngestion.ingest(
              scenario.meshletCount,
              scenario.offsetFloats,
              scenario.strideFloats,
              scenario.floatCount,
              scenario.expectedFloatCount,
              scenario.canonicalBytes.duplicate().order(ByteOrder.LITTLE_ENDIAN));
      long t1 = System.nanoTime();
      GpuMeshletBoundsPayload compressed =
          MeshletBoundsPayloadIngestion.ingestCompressed(
              scenario.meshletCount,
              scenario.offsetFloats,
              scenario.strideFloats,
              scenario.floatCount,
              scenario.expectedFloatCount,
              scenario.compressedPayload);
      long t2 = System.nanoTime();

      assertEquals(uncompressed.boundsByteSize(), compressed.boundsByteSize());
      assertEquals(uncompressed.boundsFloatCount(), compressed.boundsFloatCount());
      uncompressedNanos[i] = t1 - t0;
      compressedNanos[i] = t2 - t1;
    }

    return MeasurementResult.of(
        "visibility",
        scenario.canonicalBytes.remaining(),
        scenario.compressedPayload.payloadBytes().length,
        uncompressedNanos,
        compressedNanos);
  }

  private static MeasurementResult measureStreaming(StreamingScenario scenario) {
    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
      MeshletStreamingPayloadIngestion.ingest(
          scenario.unitCount,
          scenario.offsetInts,
          scenario.strideInts,
          scenario.intCount,
          scenario.expectedIntCount,
          scenario.canonicalBytes.duplicate().order(ByteOrder.LITTLE_ENDIAN));
      MeshletStreamingPayloadIngestion.ingestCompressed(
          scenario.unitCount,
          scenario.offsetInts,
          scenario.strideInts,
          scenario.intCount,
          scenario.expectedIntCount,
          scenario.compressedPayload);
    }

    long[] uncompressedNanos = new long[MEASURED_RUNS];
    long[] compressedNanos = new long[MEASURED_RUNS];
    for (int i = 0; i < MEASURED_RUNS; i++) {
      long t0 = System.nanoTime();
      GpuMeshletStreamingPayload uncompressed =
          MeshletStreamingPayloadIngestion.ingest(
              scenario.unitCount,
              scenario.offsetInts,
              scenario.strideInts,
              scenario.intCount,
              scenario.expectedIntCount,
              scenario.canonicalBytes.duplicate().order(ByteOrder.LITTLE_ENDIAN));
      long t1 = System.nanoTime();
      GpuMeshletStreamingPayload compressed =
          MeshletStreamingPayloadIngestion.ingestCompressed(
              scenario.unitCount,
              scenario.offsetInts,
              scenario.strideInts,
              scenario.intCount,
              scenario.expectedIntCount,
              scenario.compressedPayload);
      long t2 = System.nanoTime();

      assertEquals(uncompressed.streamUnitsByteSize(), compressed.streamUnitsByteSize());
      assertEquals(uncompressed.streamUnitsIntCount(), compressed.streamUnitsIntCount());
      uncompressedNanos[i] = t1 - t0;
      compressedNanos[i] = t2 - t1;
    }

    return MeasurementResult.of(
        "streaming",
        scenario.canonicalBytes.remaining(),
        scenario.compressedPayload.payloadBytes().length,
        uncompressedNanos,
        compressedNanos);
  }

  private static VisibilityScenario buildVisibilityScenario(int meshletCount) {
    int strideFloats = GpuMeshletBoundsPayload.BOUNDS_COMPONENTS;
    int floatCount = meshletCount * strideFloats;
    ByteBuffer canonicalBytes = ByteBuffer.allocate(floatCount * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (int i = 0; i < meshletCount; i++) {
      float x = (i % 512) * 0.25f;
      float y = ((i / 512) % 256) * 0.125f;
      float z = (i % 97) * 0.0625f;
      canonicalBytes.putFloat(x).putFloat(y).putFloat(z).putFloat(x + 0.5f).putFloat(y + 0.5f).putFloat(z + 0.5f);
    }
    canonicalBytes.flip();

    byte[] raw = toByteArray(canonicalBytes);
    byte[] compressed = RuntimePayloadCompression.compress(raw, RuntimePayloadCompressionMode.DEFLATE);
    CompressedRuntimePayload compressedPayload =
        new CompressedRuntimePayload(RuntimePayloadCompressionMode.DEFLATE, raw.length, compressed);

    return new VisibilityScenario(
        meshletCount,
        0,
        strideFloats,
        floatCount,
        floatCount,
        canonicalBytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN),
        compressedPayload);
  }

  private static StreamingScenario buildStreamingScenario(int unitCount) {
    int strideInts = GpuMeshletStreamingPayload.STREAM_UNIT_COMPONENTS;
    int intCount = unitCount * strideInts;
    ByteBuffer canonicalBytes = ByteBuffer.allocate(intCount * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    int meshletStart = 0;
    int payloadOffset = 0;
    for (int i = 0; i < unitCount; i++) {
      int meshletCount = 32 + ((i % 4) * 16);
      int payloadSize = 2048 + ((i % 8) * 512);
      canonicalBytes
          .putInt(i)
          .putInt(meshletStart)
          .putInt(meshletCount)
          .putInt(payloadOffset)
          .putInt(payloadSize);
      meshletStart += meshletCount;
      payloadOffset += payloadSize;
    }
    canonicalBytes.flip();

    byte[] raw = toByteArray(canonicalBytes);
    byte[] compressed = RuntimePayloadCompression.compress(raw, RuntimePayloadCompressionMode.DEFLATE);
    CompressedRuntimePayload compressedPayload =
        new CompressedRuntimePayload(RuntimePayloadCompressionMode.DEFLATE, raw.length, compressed);

    return new StreamingScenario(
        unitCount,
        0,
        strideInts,
        intCount,
        intCount,
        canonicalBytes.asReadOnlyBuffer().order(ByteOrder.LITTLE_ENDIAN),
        compressedPayload);
  }

  private static byte[] toByteArray(ByteBuffer bytes) {
    ByteBuffer view = bytes.asReadOnlyBuffer();
    byte[] out = new byte[view.remaining()];
    view.get(out);
    return out;
  }

  private record VisibilityScenario(
      int meshletCount,
      int offsetFloats,
      int strideFloats,
      int floatCount,
      int expectedFloatCount,
      ByteBuffer canonicalBytes,
      CompressedRuntimePayload compressedPayload) {}

  private record StreamingScenario(
      int unitCount,
      int offsetInts,
      int strideInts,
      int intCount,
      int expectedIntCount,
      ByteBuffer canonicalBytes,
      CompressedRuntimePayload compressedPayload) {}

  private record MeasurementResult(
      String label,
      int rawBytes,
      int compressedBytes,
      double compressionRatio,
      double uncompressedMedianMicros,
      double compressedMedianMicros,
      double uncompressedP95Micros,
      double compressedP95Micros,
      double deltaMedianMicros) {
    static MeasurementResult of(
        String label, int rawBytes, int compressedBytes, long[] uncompressedNanos, long[] compressedNanos) {
      long[] uncompressedSorted = uncompressedNanos.clone();
      long[] compressedSorted = compressedNanos.clone();
      Arrays.sort(uncompressedSorted);
      Arrays.sort(compressedSorted);
      double uncMedianMicros = toMicros(percentile(uncompressedSorted, 0.50));
      double cmpMedianMicros = toMicros(percentile(compressedSorted, 0.50));
      return new MeasurementResult(
          label,
          rawBytes,
          compressedBytes,
          ((double) compressedBytes) / Math.max(1, rawBytes),
          uncMedianMicros,
          cmpMedianMicros,
          toMicros(percentile(uncompressedSorted, 0.95)),
          toMicros(percentile(compressedSorted, 0.95)),
          cmpMedianMicros - uncMedianMicros);
    }

    String format() {
      return String.format(
          java.util.Locale.ROOT,
          "compression-activation[%s]: raw=%d compressed=%d ratio=%.3f ingest-uncompressed-median=%.3fus "
              + "ingest-compressed-median=%.3fus ingest-uncompressed-p95=%.3fus ingest-compressed-p95=%.3fus "
              + "delta-median=%.3fus",
          label,
          rawBytes,
          compressedBytes,
          compressionRatio,
          uncompressedMedianMicros,
          compressedMedianMicros,
          uncompressedP95Micros,
          compressedP95Micros,
          deltaMedianMicros);
    }

    private static long percentile(long[] sorted, double quantile) {
      if (sorted.length == 0) {
        return 0L;
      }
      int index = (int) Math.ceil((sorted.length - 1) * quantile);
      return sorted[index];
    }

    private static double toMicros(long nanos) {
      return nanos / 1_000.0;
    }
  }
}

