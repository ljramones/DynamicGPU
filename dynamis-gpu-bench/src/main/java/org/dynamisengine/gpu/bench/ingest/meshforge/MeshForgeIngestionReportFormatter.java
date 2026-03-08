package org.dynamisengine.gpu.bench.ingest.meshforge;

import java.util.Locale;
import java.util.Objects;

/**
 * One-line formatter for MeshForge ingestion reports.
 */
public final class MeshForgeIngestionReportFormatter {
  private MeshForgeIngestionReportFormatter() {}

  public static String toLine(MeshForgeIngestionRunReport report) {
    Objects.requireNonNull(report, "report");
    var timing = report.timing();
    var validation = report.validation();
    String uploadMs = timing.hasUploadTiming() ? formatMillis(timing.uploadExecutionNanos()) : "NA";
    String detail = report.detail() == null ? "-" : report.detail();
    return String.format(
        Locale.ROOT,
        "fixture=%s status=%s source=%s vertexCount=%s indexCount=%s stride=%s indexType=%s submeshes=%s cacheHitLoadMs=%.3f bridgeMs=%.3f uploadMs=%s totalMs=%.3f detail=%s",
        report.fixtureName(),
        report.status(),
        report.source(),
        validation == null ? "NA" : Integer.toString(validation.vertexCount()),
        validation == null ? "NA" : Integer.toString(validation.indexCount()),
        validation == null ? "NA" : Integer.toString(validation.strideBytes()),
        validation == null ? "NA" : String.valueOf(validation.indexType()),
        validation == null ? "NA" : Integer.toString(validation.submeshCount()),
        nanosToMillis(timing.cacheHitLoadNanos()),
        nanosToMillis(timing.bridgeConversionNanos()),
        uploadMs,
        nanosToMillis(timing.totalIngestionNanos()),
        sanitize(detail));
  }

  private static double nanosToMillis(long nanos) {
    return nanos / 1_000_000.0;
  }

  private static String formatMillis(long nanos) {
    return String.format(Locale.ROOT, "%.3f", nanosToMillis(nanos));
  }

  private static String sanitize(String value) {
    return value.replace('\n', ' ').replace('\r', ' ');
  }
}
