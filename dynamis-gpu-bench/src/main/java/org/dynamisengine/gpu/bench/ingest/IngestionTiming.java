package org.dynamisengine.gpu.bench.ingest;

/**
 * Timings for one ingestion execution.
 *
 * @param planIntakeNanos upload-plan supply and pre-validation time
 * @param uploadExecutionNanos executor upload + post-validation time
 * @param totalExecutorNanos total harness execution time
 */
public record IngestionTiming(
    long planIntakeNanos, long uploadExecutionNanos, long totalExecutorNanos) {}
