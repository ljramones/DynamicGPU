# Dynamis Ecosystem Architecture Inventory (Mapping Pass)

Date: 2026-03-09

## Scope and Method

This is a **mapping/review pass**, not a refactor decision document.

Evidence used in this pass:

- Repository/module layout under `/Users/larrymitchell/Dynamis`
- Root `README.md` and root `pom.xml` module declarations where present
- Existing lower-stack boundary work for `vectrix`, `MeshForge`, `DynamisGPU`

Confidence markers used below:

- `confirmed`: directly supported by code/docs/module layout
- `inferred`: plausible from naming/layout, but needs deep repo review

## Reference Anchors (Already Strong)

These three are used as architectural anchors for ecosystem alignment:

1. `vectrix`
2. `MeshForge`
3. `DynamisGPU`

Current anchor model:

- one-way dependency direction at lower stack
- prepared-data contracts owned by producer side (`MeshForge`)
- ingestion/upload/execution seams owned by GPU side (`DynamisGPU`)
- policy/planning deferred above execution layers

## 1) 28-Component Inventory

| Component | Dominant Role | Likely Purpose (confidence) | Provisional Ownership | Likely Upstream Deps | Likely Downstream Dependents | Public Boundary Guess | Internal-Only Guess | Overlap / Drift Signals | Provisional Status |
|---|---|---|---|---|---|---|---|---|---|
| `Animis` | Feature subsystem | Skeletal animation runtime + loaders + neural extension (`confirmed`) | Character animation data + evaluation | `vectrix`, `MeshForge` (`confirmed from README`) | World, Scene, LightEngine-facing animation consumers (`inferred`) | `animis`, `animis-runtime` model/eval APIs | Neural internals, demos, perf | Potential overlap with ECS/world state ownership if animation state mutates gameplay directly (`inferred`) | mostly clear |
| `DynamisAI` | Runtime service subsystem | Deterministic AI stack (perception/memory/planning/social/voice/crowd) (`confirmed`) | Agent cognition and decision systems | Core identity/event/time context (`inferred`) | Scripting, WorldEngine, UI debug tools (`inferred`) | `dynamis-ai-*` service contracts | Tooling internals, demo wiring | Possible overlap with Scripting/Session ownership of canonical truth (`inferred`) | ambiguous |
| `DynamisEvent` | Foundational substrate | Event bus implementation for ecosystem (`confirmed`) | Publish/subscribe transport + ordering semantics | `DynamisCore` contracts (`confirmed`) | Nearly all runtime subsystems (`confirmed/inferred`) | Event bus API and subscription semantics | Dispatch internals/perf details | Potential duplication with custom intra-module event systems (`inferred`) | mostly clear |
| `DynamisSceneGraph` | Runtime service subsystem | Hierarchy/transforms/bounds/culling/extraction (`confirmed`) | Spatial structure + extraction, not rendering | `DynamisCore` identity, `vectrix` math (`confirmed`) | LightEngine, world runtime (`inferred`) | Scene API + extraction output contracts | runtime culling data structures | Possible overlap with ECS and WorldEngine authoritative transform ownership (`inferred`) | ambiguous |
| `DynamisWindow` | Integration/adaptor layer | Windowing abstraction + GLFW impl (`confirmed`) | OS window/input surface abstraction | Platform/native bindings (`confirmed`) | LightEngine, input runtime, demos (`inferred`) | `window-api` | backend-specific impl details | Could overlap with Input ownership for device lifecycle (`inferred`) | mostly clear |
| `vectrix` | Foundational substrate | Math kernel for graphics/simulation (`confirmed`) | Math types/ops and numeric kernels | none/very low (`confirmed`) | Almost all technical subsystems | math API packages | low-level optimization paths | Should stay pure substrate; drift risk if engine policy enters math module (`inferred`) | ratified |
| `DynamisAssetPipeline` | Data/asset preparation layer | Pipeline API/core/CLI/test for assets (`confirmed from modules`) | Offline/import/build asset processing | Mesh/content/source toolchains (`inferred`) | Content/runtime loaders, MeshForge, game build flows (`inferred`) | `pipeline-api` | CLI/test harness internals | Overlap risk with MeshForge loaders and DynamisContent ingestion (`inferred`) | ambiguous |
| `DynamisExpression` | Foundational substrate | MVEL transpile/compile/eval stack (`confirmed`) | Expression parse/transpile runtime | JavaParser fork, compiler toolchain (`confirmed`) | Scripting/AI rules/economy (`inferred`) | expression API surface | transpiler internals and benchmarks | Potential overlap with Scripting DSL ownership (`inferred`) | mostly clear |
| `DynamisScripting` | Policy/planning subsystem | Canonical world-law/time simulation and script-driven mutation (`confirmed`) | Canonical world-state mutation policy | Expression, Event, Content, Core (`inferred`) | AI, WorldEngine, Session, UI (`inferred`) | scripting API/SPI + runtime boundary | module-specific engines (oracle/chronicler/etc.) | High overlap risk with WorldEngine and ECS if ownership not explicit (`inferred`) | ambiguous |
| `DynamisWorldEngine` | Execution/orchestration subsystem | Top-level runtime orchestrator/tick loop (`confirmed`) | Integration runtime composition and bootstrapping | Most subsystems (`confirmed from README`) | Game application host (`confirmed`) | world-api/runtime boundaries | sample host internals | High coupling hotspot by design; boundary drift likely if it absorbs feature logic (`inferred`) | needs deep review |
| `DynamisAudio` | Feature subsystem | Spatial audio + DSP + simulation stack (`confirmed`) | Audio state/mixing/simulation | Core/time/event possibly (`inferred`) | World runtime, LightEngine integration points (`inferred`) | `dynamis-audio-api` | designer/music/procedural internals | Potential overlap with Session/Content for asset ownership (`inferred`) | mostly clear |
| `DynamisGPU` | Execution/runtime services | GPU contracts + Vulkan backend + capability slices (`confirmed`) | ingestion/upload/resource/capability execution | MeshForge payload contracts, Vulkan backend (`confirmed`) | LightEngine and feature services | `dynamis-gpu-api` contracts | Vulkan backend internals | Duplication boilerplate across payload/resource patterns (known) | ratified |
| `DynamisSession` | Runtime service subsystem | Session API/core/runtime (`confirmed from modules`) | session lifecycle, player/context state (`inferred`) | Core/event/content (`inferred`) | WorldEngine, AI, UI, networking-adjacent systems (`inferred`) | `session-api` | runtime/session internals | Potential overlap with WorldEngine and Scripting runtime ownership (`inferred`) | ambiguous |
| `DynamisCollision` | Runtime service subsystem | Collision detection library (`confirmed`) | broad-phase/narrow-phase collision queries | `vectrix`, `MeshForge` (`confirmed`) | Physics, AI navigation, world interactions (`inferred`) | collision APIs | demo/test internals | Potential overlap with Physics backend collision layers (`inferred`) | mostly clear |
| `DynamisInput` | Runtime service subsystem | Input API/core/runtime/test (`confirmed from modules`) | input device events/actions (`inferred`) | Window/event/core (`inferred`) | WorldEngine, UI, gameplay systems (`inferred`) | `input-api` | runtime mapping internals | Overlap risk with Window and UI shortcut ownership (`inferred`) | ambiguous |
| `DynamisSky` | Feature subsystem | Atmosphere/sky/celestial/environment producer (`confirmed`) | sky/environment state generation | GPU, maybe LightEngine interfaces (`inferred`) | Terrain, VFX, LightEngine (`confirmed/inferred`) | sky API outputs | Vulkan implementation details | Overlap risk with LightEngine environmental lighting ownership (`inferred`) | mostly clear |
| `fastnoiselitenouveau` | Foundational substrate | Procedural noise generation toolkit (`confirmed`) | noise kernels/utilities | none/low | Terrain, VFX, content generation (`inferred`) | `noisegen-lib` API | preview/samples tools | Duplicate noise utilities may exist in Terrain/VFX (`inferred`) | mostly clear |
| `DynamisContent` | Data/asset preparation layer | Content API/core/runtime (`confirmed from modules`) | content package/catalog/runtime loading (`inferred`) | Asset pipeline/filesystem/localization (`inferred`) | Session, Scripting, UI, WorldEngine (`inferred`) | `content-api` | runtime/cache internals | Overlap with AssetPipeline and Session data ownership (`inferred`) | ambiguous |
| `DynamisLightEngine` | Policy/planning subsystem | Rendering engine scaffold + backend SPI (`confirmed`) | render planning/passes/feature integration | SceneGraph, GPU, Window, UI, feature producers (`inferred`) | Game runtime and visual features | `engine-api` + `engine-spi` | backend implementations and demos | Main boundary risk point for policy leakage into lower execution layers (`inferred`) | needs deep review |
| `DynamisTerrain` | Feature subsystem | Terrain/foliage/water system with Vulkan path (`confirmed`) | terrain feature data + runtime behavior | Sky, MeshForge, Physics, GPU, noise (`confirmed/inferred`) | LightEngine/world gameplay | `dynamisterrain-api` | vulkan/core test internals | Overlap with Sky/VFX/Physics/Content likely; needs strict boundaries (`inferred`) | likely overlapping |
| `DynamisCore` | Foundational substrate | Shared core contracts/identity/lifecycle (`confirmed`) | cross-cutting minimal contracts | none (`confirmed`) | all subsystems | core API contracts | none/very limited internals | Risk is scope creep if non-core policy enters core (`inferred`) | mostly clear |
| `DynamisLocalization` | Runtime service subsystem | i18n tables/format/runtime (`confirmed`) | localization assets and locale formatting/runtime | Content/event/core (`inferred`) | UI, Scripting, AI, game text systems (`confirmed/inferred`) | localization-api | runtime loading internals | Overlap with UI/content text ownership boundaries (`inferred`) | mostly clear |
| `DynamisUI` | Feature subsystem | Renderer-agnostic retained UI + debug overlays (`confirmed`) | UI widget/layout/runtime behavior | Localization/input/LightEngine bridge (`inferred`) | gameplay and tool surfaces | `ui-api` | widget/debug/runtime internals | Overlap with LightEngine overlay/render pass ownership (`inferred`) | ambiguous |
| `MeshForge` | Data/asset preparation layer | geometry processing/packing/MGI/runtime handoff (`confirmed`) | prepared geometry contracts + payload shaping | vectrix + loaders (`confirmed`) | DynamisGPU, terrain/content pipelines | meshforge + mgi APIs | demo/loaders internals | Lower stack is coherent; preserve boundary as reference | ratified |
| `dynamis-parent` | Tooling/support/build layer | shared Maven parent/build conventions (`confirmed`) | dependency/plugin/release policy | none | all Java repos | parent POM | build script internals | Risk only if business/runtime policy leaks into build repo (`inferred`) | mostly clear |
| `DynamisECS` | Foundational substrate | ECS API/core/runtime (`confirmed from modules`) | entity-component storage/query runtime (`inferred`) | Core identity/lifecycle (`inferred`) | WorldEngine, SceneGraph, gameplay systems (`inferred`) | `ecs-api` | runtime storage internals | Major overlap risk with SceneGraph + WorldEngine + Scripting ownership (`inferred`) | needs deep review |
| `DynamisPhysics` | Runtime service subsystem | Dual-backend physics platform (`confirmed`) | physics simulation and parity abstraction | Collision, vectrix, core (`inferred/confirmed`) | WorldEngine/gameplay/animation (`inferred`) | `dynamisphysics-api` | backend-specific implementations | Overlap with Collision and ECS/world authority boundaries (`inferred`) | ambiguous |
| `DynamisVFX` | Feature subsystem | GPU-resident VFX framework (`confirmed`) | particle/VFX simulation/render-ready outputs | GPU, LightEngine integration, maybe Sky/Terrain (`confirmed/inferred`) | LightEngine runtime and game effects | `dynamisvfx-api` | vulkan/core test internals | Overlap with LightEngine pass ownership and Sky/Terrain effects domains (`inferred`) | mostly clear |

## 2) Role Classification Summary

### Foundational substrate

- `vectrix`
- `DynamisCore`
- `DynamisEvent` (core infra service, effectively substrate)
- `DynamisECS` (provisional, pending deep review)
- `fastnoiselitenouveau`
- `DynamisExpression` (language substrate)

### Data/asset preparation layer

- `MeshForge`
- `DynamisAssetPipeline`
- `DynamisContent` (shared with runtime-service concerns)

### Execution/runtime services

- `DynamisGPU`
- `DynamisCollision`
- `DynamisPhysics`
- `DynamisInput`
- `DynamisLocalization`
- `DynamisSession`
- `DynamisSceneGraph`

### Policy/planning/integration

- `DynamisLightEngine`
- `DynamisScripting`
- `DynamisWorldEngine`

### Feature subsystems

- `Animis`
- `DynamisAI`
- `DynamisAudio`
- `DynamisSky`
- `DynamisTerrain`
- `DynamisUI`
- `DynamisVFX`

### Tooling/support/build

- `dynamis-parent`

## 3) High-Level Dependency Direction Model (Provisional)

Preferred direction (top-to-bottom dependencies allowed; inverse forbidden except via stable APIs/events):

1. **Foundational substrate**
   - `vectrix`, `DynamisCore`, `DynamisEvent`, `fastnoiselitenouveau`, `DynamisExpression`, `DynamisECS` (tentative)
2. **Prepared-data and asset shaping**
   - `MeshForge`, `DynamisAssetPipeline`, parts of `DynamisContent`
3. **Execution/runtime services**
   - `DynamisGPU`, `DynamisCollision`, `DynamisPhysics`, `DynamisInput`, `DynamisSession`, `DynamisLocalization`, `DynamisSceneGraph`
4. **Policy/planning/integration**
   - `DynamisLightEngine`, `DynamisScripting`, `DynamisWorldEngine`
5. **Feature-facing systems**
   - `Animis`, `DynamisAI`, `DynamisAudio`, `DynamisSky`, `DynamisTerrain`, `DynamisUI`, `DynamisVFX`
6. **Tooling/build**
   - `dynamis-parent` (orthogonal)

Important already-validated lower stack edge:

- `MeshForge` -> payload contracts -> `DynamisGPU` (one-way, coherent)

## 4) Likely Overlaps / Redundancies (Priority Risks)

### A. World authority overlap cluster (high risk)

- `DynamisWorldEngine`
- `DynamisScripting`
- `DynamisECS`
- `DynamisSceneGraph`
- `DynamisSession`

Risk pattern:

- multiple places potentially owning canonical state, identity, transforms, or mutation rights.

### B. Rendering policy overlap cluster (high risk)

- `DynamisLightEngine`
- `DynamisGPU`
- `MeshForge`
- `DynamisVFX`
- `DynamisSky`
- `DynamisTerrain`
- `DynamisWindow`

Risk pattern:

- policy/planning in LightEngine leaking downward into execution layers, or feature systems bypassing LightEngine and binding directly to GPU implementation concerns.

### C. Asset/content ownership cluster (medium-high risk)

- `DynamisAssetPipeline`
- `DynamisContent`
- `MeshForge`
- `DynamisSession`

Risk pattern:

- unclear split between offline processing, runtime catalog ownership, and per-session content lifecycle.

### D. Behavior/logic language cluster (medium risk)

- `DynamisExpression`
- `DynamisScripting`
- `DynamisAI`

Risk pattern:

- DSL/transpile responsibilities and runtime decision logic boundaries may blur.

### E. Physics/collision responsibility split (medium risk)

- `DynamisPhysics`
- `DynamisCollision`

Risk pattern:

- duplicated broad-phase/narrow-phase/query API surfaces if ownership is not explicit.

## 5) Strict Boundary Review Framework (for next phase)

Apply this checklist per component in deep review:

1. Ownership
- What does this component exclusively own?
- What must it never own?

2. Dependency contracts
- Allowed dependencies (by layer)
- Forbidden dependencies (upward/sideways policy leaks)

3. Public API boundary
- Canonical public packages/modules
- Which APIs are stable vs experimental

4. Internal containment
- What must remain implementation-private
- What should be split if public surface is too broad

5. Data authority
- Is this component authoritative for data X?
- If not, where is authority, and how is data mirrored/derived?

6. Runtime vs offline role separation
- Does component mix offline prep and runtime execution?
- If mixed, is that intentional and bounded?

7. Integration boundary
- Which higher-layer system consumes it?
- How to prevent direct bypass of intended integration owner?

8. Repo viability
- Should this remain separate?
- If yes, why (release cadence, ownership, dependency isolation)?

Rating output per component:

- `ratified`
- `ratified with constraints`
- `needs boundary tightening`
- `candidate consolidation`

## 6) Recommended Deep-Review Order (Bottom-Up)

Already strong anchors:

1. `vectrix` (anchor)
2. `MeshForge` (anchor)
3. `DynamisGPU` (anchor)

Recommended next wave (3–5 components):

4. `DynamisCore`
5. `DynamisEvent`
6. `DynamisECS`
7. `DynamisSceneGraph`
8. `DynamisLightEngine`

Rationale:

- This sequence resolves state/identity/event/scene boundaries before policy-heavy integration and feature-level composition.

Recommended following wave:

9. `DynamisScripting`
10. `DynamisWorldEngine`
11. `DynamisContent`
12. `DynamisAssetPipeline`
13. `DynamisSession`

Feature/system wave after core boundaries:

14. `DynamisPhysics`
15. `DynamisCollision`
16. `DynamisInput`
17. `DynamisWindow`
18. `Animis`
19. `DynamisAI`
20. `DynamisAudio`
21. `DynamisLocalization`
22. `DynamisUI`
23. `DynamisSky`
24. `DynamisTerrain`
25. `DynamisVFX`
26. `fastnoiselitenouveau`
27. `DynamisExpression`
28. `dynamis-parent` (orthogonal; review for build policy hygiene)

## 7) Mapping-Pass Conclusion

This pass establishes a concrete ecosystem map and identifies the main boundary-risk clusters.

What is clear now:

- lower stack anchor architecture is coherent (`vectrix`, `MeshForge`, `DynamisGPU`)
- the biggest unresolved risks are cross-cutting ownership boundaries in world/state/planning layers, not missing technical capability primitives

Recommended next phase:

- **repo-by-repo deep boundary ratification**, starting with `DynamisCore`/`DynamisEvent`/`DynamisECS`/`DynamisSceneGraph`/`DynamisLightEngine`
- defer consolidation decisions until these boundary ratifications are completed
