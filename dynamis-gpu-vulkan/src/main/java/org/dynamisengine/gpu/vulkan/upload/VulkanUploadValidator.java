package org.dynamisengine.gpu.vulkan.upload;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Package-private validation utilities for copy-operation overlap detection during Vulkan uploads.
 */
final class VulkanUploadValidator {

  private final Map<Long, List<Range>> srcRangesByBufferScratch = new HashMap<>();
  private final Map<Long, List<Range>> dstRangesByBufferScratch = new HashMap<>();

  private final boolean debugUpload;

  VulkanUploadValidator(boolean debugUpload) {
    this.debugUpload = debugUpload;
  }

  void validateCopyOps(List<PreparedPlan> preparedPlans, long submissionId) {
    clearRanges(srcRangesByBufferScratch);
    clearRanges(dstRangesByBufferScratch);
    for (PreparedPlan prepared : preparedPlans) {
      validateCopy(prepared.vertexCopy(), srcRangesByBufferScratch, dstRangesByBufferScratch, submissionId);
      if (prepared.indexCopy() != null) {
        validateCopy(prepared.indexCopy(), srcRangesByBufferScratch, dstRangesByBufferScratch, submissionId);
      }
    }
  }

  private static void clearRanges(Map<Long, List<Range>> rangesByBuffer) {
    for (List<Range> ranges : rangesByBuffer.values()) {
      ranges.clear();
    }
    rangesByBuffer.clear();
  }

  private void validateCopy(
      CopyOp copy,
      Map<Long, List<Range>> srcRangesByBuffer,
      Map<Long, List<Range>> dstRangesByBuffer,
      long submissionId) {
    if (copy.sizeBytes() <= 0L) {
      throw new IllegalStateException(
          "Invalid copy sizeBytes=" + copy.sizeBytes() + " submissionId=" + submissionId);
    }
    if (copy.srcBuffer() == copy.dstBuffer()) {
      throw new IllegalStateException(
          "Source and destination buffers are equal in copy submissionId="
              + submissionId + " buffer=" + copy.srcBuffer());
    }
    Range srcRange = new Range(copy.srcOffsetBytes(), copy.srcOffsetBytes() + copy.sizeBytes());
    Range dstRange = new Range(copy.dstOffsetBytes(), copy.dstOffsetBytes() + copy.sizeBytes());
    ensureNoOverlap(
        srcRangesByBuffer.computeIfAbsent(copy.srcBuffer(), unused -> new ArrayList<>()),
        srcRange, "src", submissionId, copy.srcBuffer());
    ensureNoOverlap(
        dstRangesByBuffer.computeIfAbsent(copy.dstBuffer(), unused -> new ArrayList<>()),
        dstRange, "dst", submissionId, copy.dstBuffer());
    srcRangesByBuffer.get(copy.srcBuffer()).add(srcRange);
    dstRangesByBuffer.get(copy.dstBuffer()).add(dstRange);
    if (debugUpload) {
      System.out.println(
          "[VulkanGpuUploadExecutor] copy submissionId="
              + submissionId
              + " srcBuffer=" + copy.srcBuffer()
              + " dstBuffer=" + copy.dstBuffer()
              + " srcOffset=" + copy.srcOffsetBytes()
              + " dstOffset=" + copy.dstOffsetBytes()
              + " size=" + copy.sizeBytes());
    }
  }

  private static void ensureNoOverlap(
      List<Range> existing, Range candidate, String kind, long submissionId, long bufferHandle) {
    for (Range range : existing) {
      if (candidate.overlaps(range)) {
        throw new IllegalStateException(
            "Detected " + kind
                + " overlap for submissionId=" + submissionId
                + " buffer=" + bufferHandle
                + " existing=" + range
                + " candidate=" + candidate);
      }
    }
  }

  record Range(long startInclusive, long endExclusive) {
    boolean overlaps(Range other) {
      return this.startInclusive < other.endExclusive && other.startInclusive < this.endExclusive;
    }
  }

  // Package-private records shared with VulkanGpuUploadExecutor.
  record PreparedPlan(
      org.dynamisengine.gpu.api.upload.GpuGeometryUploadPlan plan,
      org.dynamisengine.gpu.vulkan.buffer.VulkanGpuBuffer vertexBuffer,
      org.dynamisengine.gpu.vulkan.buffer.VulkanGpuBuffer indexBuffer,
      CopyOp vertexCopy,
      CopyOp indexCopy) {}

  record CopyOp(
      long srcBuffer, long srcOffsetBytes, long dstBuffer, long dstOffsetBytes, long sizeBytes) {}
}
