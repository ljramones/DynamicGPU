package org.dynamisengine.gpu.bench.ingest.meshforge;

import org.dynamisengine.gpu.bench.ingest.ValidationSummary;
import org.dynamisengine.meshforge.gpu.cache.RuntimeGeometryLoader;

/**
 * Report for one MeshForge fixture ingestion run.
 *
 * @param fixtureName fixture name
 * @param status run status
 * @param source cache source observed for measured load step
 * @param timing segmented timings
 * @param validation upload validation summary
 * @param detail optional status detail (for example blocked or failure reason)
 */
public record MeshForgeIngestionRunReport(
    String fixtureName,
    MeshForgeIngestionStatus status,
    RuntimeGeometryLoader.Source source,
    MeshForgeIngestionTiming timing,
    ValidationSummary validation,
    String detail) {}
