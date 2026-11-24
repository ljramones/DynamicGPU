package org.dynamisengine.gpu.test.compute;

import java.util.Optional;
import java.util.function.Function;
import java.util.function.IntToDoubleFunction;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;

/**
 * Runs one operation on two devices (for example native Metal and MoltenVK, or a GPU and lavapipe) and compares the
 * results against the operation's {@link AccuracyContract}:
 * <ul>
 * <li>{@code EXACT}: every element must have the same bits (NaNs match NaNs);</li>
 * <li>{@code ULPS}: each device is within {@code ulps} of the exact value, so they may differ by twice that;</li>
 * <li>{@code FORMULA}: the caller states the allowed absolute difference per element.</li>
 * </ul>
 * The report gives the first element beyond the contract, so a divergence is located, not just detected.
 *
 * @since 1.7.0
 */
public final class CrossBackend {
    private CrossBackend() { }

    /** An element where the devices differ. */
    public record Difference(int index, float first, float second, long ulps) {
        @Override public String toString() {
            return "element " + index + ": " + first + " (0x" + Integer.toHexString(Float.floatToRawIntBits(first)) + ") vs "
                    + second + " (0x" + Integer.toHexString(Float.floatToRawIntBits(second)) + "), " + ulps + " ULP";
        }
    }

    /**
     * The comparison: how many elements differ at all, the worst distance, and the first element beyond the contract
     * (empty when the results agree within it).
     */
    public record Report(String operation, int elements, int differing, long worstUlps, Optional<Difference> firstBeyond) {
        /** Whether the two devices agree within the contract. */
        public boolean withinContract() { return firstBeyond.isEmpty(); }

        @Override public String toString() {
            return operation + ": " + elements + " elements, " + differing + " differ, worst " + worstUlps + " ULP"
                    + firstBeyond.map(d -> "; beyond the contract at " + d).orElse("; within the contract");
        }
    }

    /** Compares two results under an EXACT or ULPS contract. */
    public static Report compare(AccuracyContract contract, float[] first, float[] second) {
        if (contract.bound().kind() == AccuracyContract.Bound.Kind.FORMULA) {
            throw new IllegalArgumentException(contract.operation() + " has a formula bound: pass the allowed difference per element");
        }
        double allowedUlps = 2 * contract.bound().ulps();
        return compare(contract, first, second, i -> -1, allowedUlps);
    }

    /** Compares two results under any contract, with {@code tolerance} the allowed absolute difference of element i. */
    public static Report compare(AccuracyContract contract, float[] first, float[] second, IntToDoubleFunction tolerance) {
        return compare(contract, first, second, tolerance, -1);
    }

    private static Report compare(AccuracyContract contract, float[] first, float[] second, IntToDoubleFunction tolerance, double allowedUlps) {
        if (first.length != second.length) throw new IllegalArgumentException("results differ in length: " + first.length + " vs " + second.length);
        boolean exact = contract.bound().kind() == AccuracyContract.Bound.Kind.EXACT;
        int differing = 0;
        long worst = 0;
        Difference beyond = null;
        for (int i = 0; i < first.length; i++) {
            float a = first[i], b = second[i];
            if (Float.floatToRawIntBits(a) == Float.floatToRawIntBits(b) || (Float.isNaN(a) && Float.isNaN(b))) continue;
            differing++;
            long ulps = AccuracyContract.ulpDistance(a, b);
            worst = Math.max(worst, ulps);
            boolean ok;
            if (exact) ok = false;
            else if (allowedUlps >= 0) ok = ulps <= allowedUlps;
            else ok = Math.abs((double) a - b) <= tolerance.applyAsDouble(i);
            if (!ok && beyond == null) beyond = new Difference(i, a, b, ulps);
        }
        return new Report(contract.operation(), first.length, differing, worst, Optional.ofNullable(beyond));
    }

    /** Runs {@code operation} on both devices and compares the results under an EXACT or ULPS contract. */
    public static Report run(GpuDevice first, GpuDevice second, AccuracyContract contract, Function<GpuDevice, float[]> operation) {
        return compare(contract, operation.apply(first), operation.apply(second));
    }

    /** Runs {@code operation} on both devices and compares the results with a per-element tolerance. */
    public static Report run(GpuDevice first, GpuDevice second, AccuracyContract contract, Function<GpuDevice, float[]> operation,
            IntToDoubleFunction tolerance) {
        return compare(contract, operation.apply(first), operation.apply(second), tolerance);
    }
}
