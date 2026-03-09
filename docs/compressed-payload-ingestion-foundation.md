# Compressed Payload Ingestion Foundation

This pass adds optional compressed ingestion support for the two MeshForge payload classes
activated for compression:

- meshlet visibility bounds payloads
- meshlet streaming metadata payloads

## What Changed

Added shared compressed-form boundary contracts in `dynamis-gpu-api`:

- `RuntimePayloadCompressionMode` (`NONE`, `DEFLATE`)
- `RuntimePayloadCompression`
- `CompressedRuntimePayload`

Extended ingestion seams:

- `MeshletBoundsPayloadIngestion.ingestCompressed(...)`
- `MeshletStreamingPayloadIngestion.ingestCompressed(...)`

## Boundary Behavior

Decompression is kept local to ingestion boundaries:

1. accept `CompressedRuntimePayload`
2. restore canonical uncompressed bytes (`NONE` passthrough, `DEFLATE` inflate)
3. validate uncompressed size
4. route into existing canonical ingestion logic

Downstream resource and upload paths continue to operate on canonical uncompressed payload models.

## Validation

Added/updated tests cover:

- bounds ingestion with compressed `NONE`
- bounds ingestion with compressed `DEFLATE`
- streaming ingestion with compressed `NONE`
- streaming ingestion with compressed `DEFLATE`
- malformed size mismatch rejection
- compression helper round-trip correctness

## Deferred

- default-on compression policy
- rollout to other payload classes
- alternate codecs
- renderer-level integration/policy

This keeps compression optional, selective, and boundary-scoped.

