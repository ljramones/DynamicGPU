# DynamisGPU Release Prep Plan

## Scope
Prepare this repository for ecosystem-aligned release under the DynamisEngine coordinates and release flow.

## Decisions (Locked for this pass)
- Target Java package root: `org.dynamisengine.gpu`.
- Target Maven groupId: `org.dynamisengine`.
- Parent strategy: align with `org.dynamisengine:dynamis-parent`.
- Publish policy:
  - Publish: `dynamis-gpu-api`, `dynamis-gpu-vulkan`.
  - Publish (test helper): `dynamis-gpu-test`.
  - Internal only (no normal Central release): `dynamis-gpu-bench`.

## Current State Snapshot
- Current package root in code: `org.dynamisgpu...`.
- Current groupId in POMs: `org.dynamisgpu`.
- Modules present: `dynamis-gpu-api`, `dynamis-gpu-vulkan`, `dynamis-gpu-test`, `dynamis-gpu-bench`.
- Root uses a local parent (`dynamisgpu-parent`) and not `dynamis-parent`.
- Current dependency surface:
  - No Vectrix dependency or imports detected in any module.
  - No DynamisCore dependency currently declared.

## Phase 1A Dependency Audit (2026-03-07)

### Adopt Core Candidates
- `none (safe in this pass)`: no direct, type-compatible Core contract currently maps to the GPU API surface.
- `future candidate: exception hierarchy alignment`:
  - Current `GpuException` is a checked exception with GPU-specific error code/recoverability.
  - Core `DynamisException` is runtime-oriented; replacing now would be behavioral, not dependency hygiene.
- `future candidate: resource-lifecycle contracts`:
  - Core has `Disposable`/`ResourceHandle`, but GPU contracts are backend-facing buffers/command abstractions without stable handle semantics yet.
  - Adopting now would require API design changes across modules.

### Keep Vectrix Candidates
- `none`: no compile/runtime/test usage found.

### Remove Vectrix Candidates
- `all modules (no-op removal)`:
  - `dynamis-gpu-api`, `dynamis-gpu-vulkan`, `dynamis-gpu-test`, `dynamis-gpu-bench`.
  - Reason: Vectrix is not declared nor imported, so no dependency retention is justified.

## Phase Plan

### Phase 1: Parent + Coordinates Alignment
- Update root and module POMs to `org.dynamisengine` group coordinates.
- Adopt `dynamis-parent` inheritance model.
- Keep artifactIds stable unless collision/branding requires change.
- Mark `dynamis-gpu-bench` as non-release (for example via release profile exclusions or skip deploy).

### Phase 2: Namespace Migration
- Rename packages and imports from `org.dynamisgpu` to `org.dynamisengine.gpu` across all modules.
- Verify file paths match package declarations.
- Update any reflective strings or docs mentioning old packages.

### Phase 3: Docs + Module Metadata
- Update README examples and API snippets to new namespace.
- Update `AGENTS.md` and any contributor docs for new coordinates.
- Update `module-info.java` if introduced/required by downstream standards.

### Phase 4: Release Flow Standardization
- Add/align `build.sh` and `deploy.sh` with ecosystem conventions.
- Remove legacy OSSRH/Nexus-era config if still present.
- Ensure sources/javadocs/signing/Central metadata are inherited and active in release profile.

## Phase 1C Dependency Realignment Result (2026-03-07)
- `Vectrix`: no action required.
  - No Vectrix dependency/import usage exists in this repository, so there is nothing to remove.
- `DynamisCore`: deferred (no safe slice in this pass).
  - No direct contract match was found that can replace existing GPU API types without changing public behavior.
  - `GpuException` (checked + GPU error semantics) is not a drop-in for Core runtime exceptions.
  - Core resource contracts (`Disposable`, `ResourceHandle`) would require API redesign across modules.
- `Bench internal policy`: implemented.
  - `dynamis-gpu-bench` now sets `maven.deploy.skip=true` to stay non-release in standard deploy flows.

## Validation Checklist
- `mvn -q test`
- `mvn -q -DskipTests package`
- `mvn -q -Prelease -DskipTests package`
- `rg -n "org\.dynamisgpu"` returns no production references.
- Bench module is not published in standard release flow.
- Generated artifacts contain expected coordinates and metadata.

## Codex Execution Task List
1. Coordinate migration PR (POM-only).
2. Package rename PR (source + tests).
3. Docs/metadata PR (`README`, `AGENTS.md`, scripts).
4. Release dry-run PR (`-Prelease` validation logs attached).
