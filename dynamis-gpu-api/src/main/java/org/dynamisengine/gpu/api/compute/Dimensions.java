package org.dynamisengine.gpu.api.compute;
/** Positive group counts or threads per group, in x/y/z order. */
public record Dimensions(int x, int y, int z) {
    public Dimensions {
        if (x <= 0 || y <= 0 || z <= 0) throw new IllegalArgumentException("Dimensions must be positive");
    }
    public long volume() { return Math.multiplyExact(Math.multiplyExact((long) x, y), z); }
}
