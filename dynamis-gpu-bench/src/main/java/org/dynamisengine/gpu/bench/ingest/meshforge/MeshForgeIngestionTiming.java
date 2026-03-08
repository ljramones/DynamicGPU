package org.dynamisengine.gpu.bench.ingest.meshforge;

/**
 * End-to-end MeshForge -> upload timing segments.
 *
 * @param cacheHitLoadNanos cache-hit runtime loader time
 * @param bridgeConversionNanos payload-to-plan conversion time
 * @param uploadExecutionNanos upload execution and validation time, or {@code -1} when unavailable
 * @param totalIngestionNanos total ingestion time
 */
public record MeshForgeIngestionTiming(
    long cacheHitLoadNanos,
    long bridgeConversionNanos,
    long uploadExecutionNanos,
    long totalIngestionNanos) {
  public boolean hasUploadTiming() {
    return uploadExecutionNanos >= 0L;
  }
}
