package org.dynamisengine.gpu.bench.ingest;

/**
 * Fixture-independent ingestion run result.
 *
 * @param fixtureName fixture identifier
 * @param timing measured timing segments
 * @param validation extracted and validated geometry summary
 */
public record IngestionRunReport(
    String fixtureName, IngestionTiming timing, ValidationSummary validation) {}
