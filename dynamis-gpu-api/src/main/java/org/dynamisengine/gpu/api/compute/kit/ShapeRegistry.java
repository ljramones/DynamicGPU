package org.dynamisengine.gpu.api.compute.kit;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.BiPredicate;
import java.util.function.ToDoubleBiFunction;
import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.api.compute.GpuCapabilities;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.PipelineDescriptor;

/**
 * Picks a kernel variant by shape and device, so a kernel family keeps one entry point while its variants specialize:
 * a tiled kernel, a matrix-hardware kernel, a skinny-matrix kernel. Each entry says whether it supports a shape on a
 * device and estimates its cost (relative units, from measurements documented with the family); {@link #select}
 * takes the cheapest supported entry, the first registered on a tie.
 * <p>
 * Entries may give the pipeline they compile, so {@link #checkLimits} can verify that every variant a device
 * supports fits its limits: threads per group within the compiled pipeline's limit, group-shared memory within the
 * device's.
 *
 * @param <S> the shape type
 * @param <V> the variant type (an algorithm enum, for example)
 * @since 1.7.0
 */
public final class ShapeRegistry<S, V> {
    /** One variant. {@code pipeline} may be null for variants without a single pipeline. */
    public record Entry<S, V>(String name, V variant, BiPredicate<GpuDevice, S> supports, ToDoubleBiFunction<GpuDevice, S> cost,
            BiFunction<GpuDevice, S, PipelineDescriptor> pipeline) { }

    private final List<Entry<S, V>> entries;

    private ShapeRegistry(List<Entry<S, V>> entries) { this.entries = List.copyOf(entries); }

    /** A registry builder. */
    public static <S, V> Builder<S, V> builder() { return new Builder<>(); }

    /** Builds a registry, entries in registration order. */
    public static final class Builder<S, V> {
        private final List<Entry<S, V>> entries = new ArrayList<>();

        private Builder() { }

        /** Adds a variant. */
        public Builder<S, V> add(String name, V variant, BiPredicate<GpuDevice, S> supports, ToDoubleBiFunction<GpuDevice, S> cost,
                BiFunction<GpuDevice, S, PipelineDescriptor> pipeline) {
            entries.add(new Entry<>(name, variant, supports, cost, pipeline));
            return this;
        }

        public ShapeRegistry<S, V> build() {
            if (entries.isEmpty()) throw new IllegalStateException("a registry needs at least one variant");
            return new ShapeRegistry<>(entries);
        }
    }

    /** The entries in registration order. */
    public List<Entry<S, V>> entries() { return entries; }

    /** The entries that support {@code shape} on {@code device}, cheapest first (stable). */
    public List<Entry<S, V>> ranked(GpuDevice device, S shape) {
        List<Entry<S, V>> ok = new ArrayList<>();
        for (var e : entries) if (e.supports().test(device, shape)) ok.add(e);
        ok.sort((a, b) -> Double.compare(a.cost().applyAsDouble(device, shape), b.cost().applyAsDouble(device, shape)));
        return ok;
    }

    /** The cheapest variant supporting {@code shape} on {@code device}; UNSUPPORTED if none does. */
    public V select(GpuDevice device, S shape) {
        var ranked = ranked(device, shape);
        if (ranked.isEmpty()) throw new ComputeException(ComputeException.Code.UNSUPPORTED, "no variant supports " + shape + " on " + device.adapter().name());
        return ranked.getFirst().variant();
    }

    /**
     * Compiles every variant that supports {@code shape} on {@code device} and has a pipeline, and returns the
     * violations of its limits (empty when all fit).
     */
    public List<String> checkLimits(GpuDevice device, S shape) {
        List<String> problems = new ArrayList<>();
        GpuCapabilities caps = device.capabilities();
        for (var e : entries) {
            if (e.pipeline() == null || !e.supports().test(device, shape)) continue;
            var descriptor = e.pipeline().apply(device, shape);
            try (var p = device.pipeline(descriptor)) {
                var limits = p.limits();
                long threads = descriptor.localSize().volume();
                if (threads > limits.maxThreads()) problems.add(e.name() + ": " + threads + " threads per group exceed the pipeline's " + limits.maxThreads());
                if (limits.groupSharedBytes() > caps.maxGroupSharedBytes()) {
                    problems.add(e.name() + ": " + limits.groupSharedBytes() + " group-shared bytes exceed the device's " + caps.maxGroupSharedBytes());
                }
            }
        }
        return problems;
    }

    /** The entry named {@code name}. */
    public Optional<Entry<S, V>> entry(String name) {
        return entries.stream().filter(e -> e.name().equals(name)).findFirst();
    }
}
