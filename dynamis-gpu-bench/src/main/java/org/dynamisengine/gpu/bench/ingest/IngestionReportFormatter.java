package org.dynamisengine.gpu.bench.ingest;

import java.util.Locale;
import java.util.Objects;

/**
 * Text formatter for fixture-independent ingestion reports.
 */
public final class IngestionReportFormatter {
  private IngestionReportFormatter() {}

  public static String toLine(IngestionRunReport report) {
    Objects.requireNonNull(report, "report");
    IngestionTiming t = report.timing();
    ValidationSummary v = report.validation();
    return String.format(
        Locale.ROOT,
        "fixture=%s vertexCount=%d indexCount=%d stride=%d indexType=%s submeshes=%d intakeMs=%.3f uploadMs=%.3f totalMs=%.3f",
        report.fixtureName(),
        v.vertexCount(),
        v.indexCount(),
        v.strideBytes(),
        v.indexType(),
        v.submeshCount(),
        nanosToMillis(t.planIntakeNanos()),
        nanosToMillis(t.uploadExecutionNanos()),
        nanosToMillis(t.totalExecutorNanos()));
  }

  private static double nanosToMillis(long nanos) {
    return nanos / 1_000_000.0;
  }
}
