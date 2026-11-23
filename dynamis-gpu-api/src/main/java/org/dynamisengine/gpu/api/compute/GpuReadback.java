package org.dynamisengine.gpu.api.compute;
/** Host snapshot becomes available when the producing submission has been polled/waited. */
public final class GpuReadback {
    private byte[] data;
    private ComputeException failure;
    public byte[] bytes() {
        if (failure != null) throw failure;
        if (data == null) throw new ComputeException(ComputeException.Code.PENDING, "Await producing submission before reading");
        return data.clone();
    }
    /** Backend SPI: publish a completed snapshot. */
    public void complete(byte[] data) { this.data = data.clone(); }
    /** Backend SPI: publish a failed producing operation. */
    public void fail(ComputeException failure) { this.failure = failure; }
}
