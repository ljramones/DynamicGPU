package org.dynamisengine.gpu.api.upload;

/**
 * Snapshot metrics for upload scheduling behavior and completion health.
 *
 * @param inflightCount current number of in-flight submissions
 * @param backlogDepth current number of queued submissions
 * @param inflightBytes current bytes represented by in-flight submissions
 * @param maxInflightSubmissions high-water mark of in-flight submissions
 * @param maxInflightBytes high-water mark of in-flight bytes
 * @param maxBacklogDepth high-water mark of backlog depth
 * @param completedUploads completed submission count
 * @param completedBytes completed upload bytes
 * @param throughputGbps effective completed throughput in gigabytes/second
 * @param averageTtfuMillis average request-to-ready time in milliseconds
 * @param p95TtfuMillis p95 request-to-ready time in milliseconds
 * @param averageCompletionLatencyMillis average dispatch-to-ready time in milliseconds
 */
public record UploadTelemetry(
    int inflightCount,
    int backlogDepth,
    long inflightBytes,
    int maxInflightSubmissions,
    long maxInflightBytes,
    int maxBacklogDepth,
    long completedUploads,
    long completedBytes,
    double throughputGbps,
    double averageTtfuMillis,
    double p95TtfuMillis,
    double averageCompletionLatencyMillis) {}

