package org.dynamisengine.gpu.api.upload;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicLong;
import org.dynamisengine.gpu.api.error.GpuErrorCode;
import org.dynamisengine.gpu.api.error.GpuException;
import org.dynamisengine.gpu.api.resource.GpuMeshResource;

/**
 * Default bounded-backlog upload manager with pull-based dispatch.
 */
public final class DefaultUploadManager implements UploadManager {
  private static final int DEFAULT_TARGET_INFLIGHT = 2;
  private static final int DEFAULT_MAX_BACKLOG = 1024;
  private static final int DEFAULT_SAMPLE_WINDOW = 2048;

  private final GpuUploadExecutor uploadExecutor;
  private final int targetInflight;
  private final int maxBacklog;
  private final int sampleWindow;
  private final ExecutorService workerPool;
  private final AtomicLong nextTicketId = new AtomicLong(1L);
  private final ArrayDeque<UploadTask> backlog = new ArrayDeque<>();
  private final Map<Long, UploadTask> inflight = new LinkedHashMap<>();
  private final ArrayDeque<Long> ttfuSamplesNanos = new ArrayDeque<>();
  private final ArrayDeque<Long> completionSamplesNanos = new ArrayDeque<>();

  private boolean closed;
  private long inflightBytes;
  private int maxInflightSubmissionsObserved;
  private long maxInflightBytesObserved;
  private int maxBacklogDepthObserved;
  private long completedUploads;
  private long completedBytes;
  private long firstDispatchNanos;
  private long lastCompletionNanos;
  private long ttfuSampleSumNanos;
  private long completionSampleSumNanos;

  public DefaultUploadManager(GpuUploadExecutor uploadExecutor) {
    this(uploadExecutor, DEFAULT_TARGET_INFLIGHT, DEFAULT_MAX_BACKLOG);
  }

  public DefaultUploadManager(GpuUploadExecutor uploadExecutor, int targetInflight, int maxBacklog) {
    this(uploadExecutor, targetInflight, maxBacklog, DEFAULT_SAMPLE_WINDOW);
  }

  public DefaultUploadManager(
      GpuUploadExecutor uploadExecutor, int targetInflight, int maxBacklog, int sampleWindow) {
    this.uploadExecutor = Objects.requireNonNull(uploadExecutor, "uploadExecutor");
    if (targetInflight < 1) {
      throw new IllegalArgumentException("targetInflight must be >= 1");
    }
    if (maxBacklog < 1) {
      throw new IllegalArgumentException("maxBacklog must be >= 1");
    }
    if (sampleWindow < 1) {
      throw new IllegalArgumentException("sampleWindow must be >= 1");
    }
    this.targetInflight = targetInflight;
    this.maxBacklog = maxBacklog;
    this.sampleWindow = sampleWindow;
    this.workerPool = Executors.newFixedThreadPool(targetInflight, new UploadWorkerThreadFactory());
  }

  @Override
  public synchronized UploadTicket submit(GpuGeometryUploadPlan plan) {
    Objects.requireNonNull(plan, "plan");
    ensureOpen();
    if (backlog.size() >= maxBacklog) {
      throw new IllegalStateException("upload backlog is full (maxBacklog=" + maxBacklog + ")");
    }

    long id = nextTicketId.getAndIncrement();
    UploadTask task = new UploadTask(id, plan, totalPlanBytes(plan), System.nanoTime());
    backlog.addLast(task);
    if (backlog.size() > maxBacklogDepthObserved) {
      maxBacklogDepthObserved = backlog.size();
    }
    dispatchAvailableLocked();
    return new DefaultUploadTicket(task);
  }

  @Override
  public synchronized void pump() {
    ensureOpen();
    dispatchAvailableLocked();
  }

  @Override
  public void drain() {
    while (true) {
      CompletableFuture<GpuMeshResource> waitFuture;
      synchronized (this) {
        if (backlog.isEmpty() && inflight.isEmpty()) {
          return;
        }
        dispatchAvailableLocked();
        waitFuture = firstInflightFutureLocked();
      }

      if (waitFuture == null) {
        Thread.onSpinWait();
        continue;
      }
      try {
        waitFuture.join();
      } catch (CompletionException ignored) {
        // Failure is tracked in the ticket. Drain only ensures settlement.
      }
    }
  }

  @Override
  public synchronized int backlogSize() {
    return backlog.size();
  }

  @Override
  public synchronized int inflightCount() {
    return inflight.size();
  }

  @Override
  public synchronized UploadTelemetry telemetry() {
    long durationNanos =
        firstDispatchNanos > 0L && lastCompletionNanos >= firstDispatchNanos
            ? (lastCompletionNanos - firstDispatchNanos)
            : 0L;
    double throughputGbps =
        durationNanos > 0L ? (completedBytes / (double) durationNanos) : 0.0;
    return new UploadTelemetry(
        inflight.size(),
        backlog.size(),
        inflightBytes,
        maxInflightSubmissionsObserved,
        maxInflightBytesObserved,
        maxBacklogDepthObserved,
        completedUploads,
        completedBytes,
        throughputGbps,
        nanosToMillis(averageNanos(ttfuSamplesNanos, ttfuSampleSumNanos)),
        nanosToMillis(p95Nanos(ttfuSamplesNanos)),
        nanosToMillis(averageNanos(completionSamplesNanos, completionSampleSumNanos)));
  }

  @Override
  public void close() {
    List<UploadTask> pending;
    synchronized (this) {
      if (closed) {
        return;
      }
      closed = true;
      pending = new ArrayList<>(backlog);
      pending.addAll(inflight.values());
      backlog.clear();
      inflight.clear();
      inflightBytes = 0L;
    }

    GpuException closedException = closedManagerException();
    for (UploadTask task : pending) {
      task.future.completeExceptionally(closedException);
    }
    workerPool.shutdownNow();
  }

  public int targetInflight() {
    return targetInflight;
  }

  public int maxBacklog() {
    return maxBacklog;
  }

  private void dispatchAvailableLocked() {
    while (!closed && inflight.size() < targetInflight && !backlog.isEmpty()) {
      UploadTask task = backlog.removeFirst();
      long now = System.nanoTime();
      task.dispatchNanos = now;
      if (firstDispatchNanos == 0L) {
        firstDispatchNanos = now;
      }

      inflight.put(task.id, task);
      inflightBytes += task.sizeBytes;
      if (inflight.size() > maxInflightSubmissionsObserved) {
        maxInflightSubmissionsObserved = inflight.size();
      }
      if (inflightBytes > maxInflightBytesObserved) {
        maxInflightBytesObserved = inflightBytes;
      }

      CompletableFuture
          .supplyAsync(
              () -> {
                try {
                  return uploadExecutor.upload(task.plan);
                } catch (GpuException e) {
                  throw new CompletionException(e);
                }
              },
              workerPool)
          .whenComplete((resource, error) -> finishTask(task, resource, error));
    }
  }

  private void finishTask(UploadTask task, GpuMeshResource resource, Throwable error) {
    synchronized (this) {
      if (!inflight.containsKey(task.id)) {
        if (resource != null) {
          resource.close();
        }
        return;
      }

      inflight.remove(task.id);
      inflightBytes -= task.sizeBytes;
      long completionNanos = System.nanoTime();
      task.completeNanos = completionNanos;
      lastCompletionNanos = completionNanos;

      completedUploads++;
      completedBytes += task.sizeBytes;
      addTtfuSampleLocked(task.completeNanos - task.enqueueNanos);
      addCompletionSampleLocked(task.completeNanos - task.dispatchNanos);

      Throwable unwrapped = unwrapCompletion(error);
      if (unwrapped == null) {
        task.future.complete(resource);
      } else {
        task.future.completeExceptionally(unwrapped);
      }

      dispatchAvailableLocked();
    }
  }

  private void addTtfuSampleLocked(long nanos) {
    ttfuSamplesNanos.addLast(nanos);
    ttfuSampleSumNanos += nanos;
    trimSampleWindow(ttfuSamplesNanos, true);
  }

  private void addCompletionSampleLocked(long nanos) {
    completionSamplesNanos.addLast(nanos);
    completionSampleSumNanos += nanos;
    trimSampleWindow(completionSamplesNanos, false);
  }

  private void trimSampleWindow(ArrayDeque<Long> samples, boolean ttfu) {
    while (samples.size() > sampleWindow) {
      long removed = samples.removeFirst();
      if (ttfu) {
        ttfuSampleSumNanos -= removed;
      } else {
        completionSampleSumNanos -= removed;
      }
    }
  }

  private CompletableFuture<GpuMeshResource> firstInflightFutureLocked() {
    for (UploadTask task : inflight.values()) {
      return task.future;
    }
    return null;
  }

  private static long totalPlanBytes(GpuGeometryUploadPlan plan) {
    long vertexBytes = plan.vertexData().remaining();
    long indexBytes = plan.indexData() == null ? 0L : plan.indexData().remaining();
    return vertexBytes + indexBytes;
  }

  private static Throwable unwrapCompletion(Throwable error) {
    if (error == null) {
      return null;
    }
    if (error instanceof CompletionException || error instanceof ExecutionException) {
      Throwable cause = error.getCause();
      return cause == null ? error : cause;
    }
    return error;
  }

  private static long averageNanos(ArrayDeque<Long> samples, long sum) {
    if (samples.isEmpty()) {
      return 0L;
    }
    return sum / samples.size();
  }

  private static long p95Nanos(ArrayDeque<Long> samples) {
    if (samples.isEmpty()) {
      return 0L;
    }
    long[] values = new long[samples.size()];
    int i = 0;
    for (Long sample : samples) {
      values[i++] = sample;
    }
    java.util.Arrays.sort(values);
    int index = (int) Math.ceil(values.length * 0.95d) - 1;
    return values[Math.max(index, 0)];
  }

  private static double nanosToMillis(long nanos) {
    return nanos / 1_000_000.0;
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("upload manager is closed");
    }
  }

  private static GpuException closedManagerException() {
    return new GpuException(
        GpuErrorCode.BACKEND_INIT_FAILED, "Upload manager closed before completion", true);
  }

  private static GpuException asGpuException(Throwable throwable) {
    if (throwable instanceof GpuException gpuException) {
      return gpuException;
    }
    return new GpuException(
        GpuErrorCode.BACKEND_INIT_FAILED,
        "Upload failed: " + throwable.getClass().getSimpleName() + ": " + throwable.getMessage(),
        throwable,
        false);
  }

  private static final class UploadTask {
    private final long id;
    private final GpuGeometryUploadPlan plan;
    private final long sizeBytes;
    private final long enqueueNanos;
    private final CompletableFuture<GpuMeshResource> future = new CompletableFuture<>();
    private long dispatchNanos;
    private long completeNanos;

    private UploadTask(long id, GpuGeometryUploadPlan plan, long sizeBytes, long enqueueNanos) {
      this.id = id;
      this.plan = plan;
      this.sizeBytes = sizeBytes;
      this.enqueueNanos = enqueueNanos;
    }
  }

  private static final class DefaultUploadTicket implements UploadTicket {
    private final UploadTask task;

    private DefaultUploadTicket(UploadTask task) {
      this.task = task;
    }

    @Override
    public long id() {
      return task.id;
    }

    @Override
    public boolean isComplete() {
      return task.future.isDone();
    }

    @Override
    public GpuMeshResource await() throws InterruptedException, GpuException {
      try {
        return task.future.get();
      } catch (ExecutionException e) {
        throw asGpuException(unwrapCompletion(e));
      }
    }
  }

  private static final class UploadWorkerThreadFactory implements ThreadFactory {
    private final AtomicLong threadId = new AtomicLong(1L);

    @Override
    public Thread newThread(Runnable runnable) {
      Thread thread = new Thread(runnable, "dynamis-upload-worker-" + threadId.getAndIncrement());
      thread.setDaemon(true);
      return thread;
    }
  }
}
