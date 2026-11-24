package org.dynamisengine.gpu.api.compute.kit;

import java.util.List;

/**
 * An operation's accuracy contract, as data: what it reads, how it accumulates, how it rounds its output, whether its
 * combination order is fixed, and how far its result may be from the exact one. A caller comparing GPU results with
 * a CPU reference (NNJava's gradient checks, for example) uses the stated bound instead of a guessed tolerance.
 * <p>
 * Every bound is for normal inputs and results: Apple GPUs flush subnormal values to zero, and other devices may.
 *
 * @param operation what is computed, such as {@code "GemmF16 TILED, FP32 C"}
 * @param inputs the input element types, such as {@code List.of("FP16", "FP16")}
 * @param accumulation the type sums are carried in, or {@code "none"}
 * @param output the output type and its rounding, such as {@code "FP32, rounded once"}
 * @param order {@link Order#FIXED} or {@link Order#DEVICE}
 * @param bound how far the result may be from the exact (infinitely precise) value, or from {@code reference}
 * @param reference the CPU method that reproduces a {@link Order#FIXED} operation bit for bit, or empty
 * @since 1.7.0
 */
public record AccuracyContract(String operation, List<String> inputs, String accumulation, String output, Order order,
        Bound bound, String reference) {
    public AccuracyContract {
        inputs = List.copyOf(inputs);
        if (bound.kind() == Bound.Kind.EXACT && reference.isEmpty()) {
            throw new IllegalArgumentException("an exact contract names the reference it matches");
        }
    }

    /** Whether the combination order is fixed or the device's. */
    public enum Order {
        /**
         * One combination order on every device, so every device that rounds each IEEE operation as written gives
         * the same bits (DynamisGPU's shader build ensures that: no contraction, no reassociation).
         */
        FIXED,
        /** The order, or a function such as {@code exp}, is the device's: results are bounded, not reproducible. */
        DEVICE
    }

    /**
     * A bound.
     *
     * @param kind {@link Kind#EXACT}, {@link Kind#ULPS} or {@link Kind#FORMULA}
     * @param ulps for {@link Kind#ULPS}: the most units in the last place the result may be from the exact value
     * @param formula the bound in words or symbols, with u = 2^-24 the FP32 unit roundoff
     */
    public record Bound(Kind kind, double ulps, String formula) {
        public enum Kind {
            /** Bit for bit equal to the contract's {@code reference}. */
            EXACT,
            /** Within {@code ulps} units in the last place of the exact value. */
            ULPS,
            /** Within the stated formula. */
            FORMULA
        }

        public Bound {
            if (ulps < 0) throw new IllegalArgumentException("negative ulps");
        }

        /** Equal to the reference, bit for bit. */
        public static Bound exact() { return new Bound(Kind.EXACT, 0, "equal to the reference, bit for bit"); }

        /** Within {@code ulps} ULP of the exact value. */
        public static Bound ulps(double ulps, String formula) { return new Bound(Kind.ULPS, ulps, formula); }

        /** Within a stated formula. */
        public static Bound formula(String formula) { return new Bound(Kind.FORMULA, 0, formula); }
    }

    /** Whether the operation gives the same bits on every conforming device. */
    public boolean reproducible() { return order == Order.FIXED; }

    /**
     * The distance between two floats in units in the last place: the number of representable floats between them
     * (0 when equal, including +0 and -0; {@code Long.MAX_VALUE} when either is NaN).
     */
    public static long ulpDistance(float a, float b) {
        if (Float.isNaN(a) || Float.isNaN(b)) return Long.MAX_VALUE;
        return Math.abs(ordered(a) - ordered(b));
    }

    private static long ordered(float f) {
        int bits = Float.floatToRawIntBits(f);
        return bits < 0 ? -(long) (bits & 0x7FFFFFFF) : bits;
    }
}
