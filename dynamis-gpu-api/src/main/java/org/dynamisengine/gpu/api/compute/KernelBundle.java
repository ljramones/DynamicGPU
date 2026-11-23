package org.dynamisengine.gpu.api.compute;
import java.util.Map;
import java.util.HexFormat;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
/** Versioned logical kernel containing backend-specific executables. No implicit translation. */
public record KernelBundle(String name, String version, Map<Backend, Executable> executables) {
    public enum Format { MOCK, MSL_SOURCE, METALLIB, SPIRV }
    public record Executable(Format format, byte[] bytes) {
        public Executable { java.util.Objects.requireNonNull(format); bytes = bytes.clone(); }
        @Override public byte[] bytes() { return bytes.clone(); }
        public String identity() {
            try { return format + ":" + HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)); }
            catch (NoSuchAlgorithmException e) { throw new AssertionError(e); }
        }
    }
    public KernelBundle {
        java.util.Objects.requireNonNull(name); java.util.Objects.requireNonNull(version);
        executables = Map.copyOf(executables);
    }
    public Executable executable(Backend backend) {
        var result = executables.get(backend);
        if (result == null) throw new ComputeException(ComputeException.Code.UNSUPPORTED, name + " has no executable for " + backend);
        return result;
    }
}
