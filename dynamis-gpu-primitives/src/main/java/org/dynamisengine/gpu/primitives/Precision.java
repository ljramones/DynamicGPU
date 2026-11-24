package org.dynamisengine.gpu.primitives;

/** The element type a primitive writes. */
public enum Precision {
    /** IEEE binary32, as computed. */
    FP32(4),
    /** IEEE binary16: the FP32 result rounded once to the nearest half (overflow gives infinity). */
    FP16(2);

    private final int bytes;

    Precision(int bytes) { this.bytes = bytes; }

    /** Bytes per element. */
    public int bytes() { return bytes; }
}
