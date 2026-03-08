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
   * Releases manager resources and rejects future submissions.
   */
  @Override
  void close();
}

