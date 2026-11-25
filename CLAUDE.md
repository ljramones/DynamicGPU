# CLAUDE.md

@AGENTS.md

## 1.14.1 handoff

NNJava can use fused attention for padded batches by passing GPU-resident valid key lengths to both
forward and backward. The group size counts query heads per length. The example covers grouped-query
attention, empty rows and the causal offset. [Adapter example](docs/primitives.md#valid-key-lengths-1141) · [Release notes](docs/releases/1.14.1.md)

The cache still stores FP16. The long-context benchmark and CPU codec comparison are included for
reproduction. Revisit four-bit values with captured model activations and a quality target. [Measurements and decision](docs/paged-kv-study.md)

The remaining NNJava kernel requests are tracked in the [roadmap](docs/roadmap.md).
