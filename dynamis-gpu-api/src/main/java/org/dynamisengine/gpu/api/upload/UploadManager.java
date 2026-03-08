package org.dynamisengine.gpu.api.upload;

/**
 * Minimal pull-based upload scheduling boundary.
 */
public interface UploadManager extends AutoCloseable {
  /**
   * Submits an upload plan. The manager may queue or dispatch immediately.
   *
   * @param plan upload plan
   * @return ticket for completion tracking
   */
  UploadTicket submit(GpuGeometryUploadPlan plan);

  /**
   * Pumps scheduler state by dispatching queued work when in-flight capacity is available.
   */
  void pump();

  /**
   * Blocks until queued and in-flight work is drained.
   */
  void drain();

  /**
   * @return current queued request count
   */
  int backlogSize();

  /**
   * @return current in-flight request count
   */
  int inflightCount();

  /**
   * @return immutable telemetry snapshot
   */
  UploadTelemetry telemetry();

  /**
   * @return compact diagnostic snapshot suitable for logs and benchmark traces
   */
  default String debugSnapshot() {
    UploadTelemetry telemetry = telemetry();
    return "inflight="
        + telemetry.inflightCount()
        + ", backlog="
        + telemetry.backlogDepth()
        + ", inflightBytes="
        + telemetry.inflightBytes()
        + ", completedUploads="
        + telemetry.completedUploads()
        + ", throughputGbps="
        + String.format(java.util.Locale.ROOT, "%.3f", telemetry.throughputGbps())
        + ", avgTtfuMs="
        + String.format(java.util.Locale.ROOT, "%.3f", telemetry.averageTtfuMillis())
        + ", p95TtfuMs="
        + String.format(java.util.Locale.ROOT, "%.3f", telemetry.p95TtfuMillis())
        + ", avgCompletionLatencyMs="
        + String.format(java.util.Locale.ROOT, "%.3f", telemetry.averageCompletionLatencyMillis());
  }

  /**
   * Releases manager resources and rejects future submissions.
   */
  @Override
  void close();
}
