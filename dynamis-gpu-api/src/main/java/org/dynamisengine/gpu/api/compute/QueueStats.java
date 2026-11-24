package org.dynamisengine.gpu.api.compute;

/**
 * A queue's counters, kept on the host: reading them never waits for the device.
 *
 * @param pending submissions not yet known to be complete
 * @param submitted submissions since the device opened
 * @param completed submissions that completed
 * @param failed submissions that failed (kernel errors included)
 * @param transferBytesInFlight upload and download bytes of the pending submissions
 * @since 1.6.0
 */
public record QueueStats(int pending, long submitted, long completed, long failed, long transferBytesInFlight) { }
