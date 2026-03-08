package org.dynamisengine.gpu.bench.ingest;

import org.dynamisengine.gpu.api.layout.IndexType;

/**
 * Fixture-independent geometry validation metadata.
 *
 * @param vertexCount derived vertex count
 * @param indexCount derived index count (0 for non-indexed plans)
 * @param strideBytes vertex stride
 * @param indexType index element type (null for non-indexed plans)
 * @param submeshCount number of submesh ranges
 */
public record ValidationSummary(
    int vertexCount, int indexCount, int strideBytes, IndexType indexType, int submeshCount) {}
