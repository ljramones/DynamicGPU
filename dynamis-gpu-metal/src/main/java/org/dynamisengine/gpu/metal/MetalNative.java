package org.dynamisengine.gpu.metal;

import org.dynamisengine.gpu.api.compute.*;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.*;
import java.util.*;
import static java.lang.foreign.ValueLayout.*;

/** Versioned C ABI binding. All native strings are copied before another bridge call. */
final class MetalNative {
    private final Arena libraryArena = Arena.ofShared();
    private final SymbolLookup symbols;
    private final Map<String, MethodHandle> functions = new java.util.concurrent.ConcurrentHashMap<>();
    /**
     * The bridge ABI this code calls: 2 adds dg_device_features, 3 dg_gpu_seconds, 4 dg_wait, dg_device_memory and
     * the profiling calls, 5 dg_compute_label, the event (fence) calls and dg_batch_create_on, 6 function constants
     * (dg_pipeline_create_specialized), dg_pipeline_limits and dg_device_group_memory.
     */
    static final int ABI = 6;
    private static MetalNative singleton;
    static synchronized MetalNative instance() {
        if (singleton == null) singleton = new MetalNative();
        return singleton;
    }
    private MetalNative() {
        if (!System.getProperty("os.name").toLowerCase(Locale.ROOT).contains("mac"))
            throw new ComputeException(ComputeException.Code.UNAVAILABLE, "Native Metal requires macOS 13+ on arm64 or x86_64");
        try {
            String configured = System.getProperty("dynamisgpu.metal.library"); Path path;
            if (configured != null) path = Path.of(configured).toAbsolutePath();
            else {
                String arch = System.getProperty("os.arch").equals("aarch64") ? "arm64" : "x86_64";
                String resource = "/native/macos-" + arch + "/libdynamis_metal.dylib";
                try (var stream = MetalNative.class.getResourceAsStream(resource)) {
                    if (stream == null) throw new ComputeException(ComputeException.Code.UNAVAILABLE, "Missing " + resource + "; build with -Pmetal-native or set -Ddynamisgpu.metal.library=/absolute/path");
                    path = Files.createTempFile("dynamis-metal-", ".dylib"); Files.copy(stream, path, StandardCopyOption.REPLACE_EXISTING); path.toFile().deleteOnExit();
                }
            }
            symbols = SymbolLookup.libraryLookup(path, libraryArena);
            if (integer("dg_abi") != ABI) throw new ComputeException(ComputeException.Code.UNSUPPORTED, "Metal bridge ABI mismatch; Java requires ABI " + ABI);
        } catch (ComputeException e) { throw e; }
        catch (Exception | LinkageError e) { throw new ComputeException(ComputeException.Code.UNAVAILABLE, "Cannot load Metal bridge: " + e.getMessage(), e); }
    }
    private Object call(String name, MemoryLayout result, Object... args) {
        try {
            MethodHandle handle = functions.get(name);
            if (handle == null) {
                MemoryLayout[] layouts = Arrays.stream(args).map(a -> a instanceof Integer ? JAVA_INT : a instanceof Long ? JAVA_LONG : ADDRESS).toArray(MemoryLayout[]::new);
                var descriptor = result == null ? FunctionDescriptor.ofVoid(layouts) : FunctionDescriptor.of(result, layouts);
                handle = Linker.nativeLinker().downcallHandle(symbols.find(name).orElseThrow(), descriptor); functions.put(name, handle);
            }
            return handle.invokeWithArguments(args);
        } catch (Throwable e) { throw new ComputeException(ComputeException.Code.NATIVE_ERROR, "Metal ABI call " + name + " failed: " + e, e); }
    }
    int integer(String name, Object... args) { return (int) call(name, JAVA_INT, args); }
    long number(String name, Object... args) { return (long) call(name, JAVA_LONG, args); }
    double real(String name, Object... args) { return (double) call(name, JAVA_DOUBLE, args); }
    MemorySegment address(String name, Object... args) { return (MemorySegment) call(name, ADDRESS, args); }
    MemorySegment pointer(String name, Object... args) {
        var value = address(name, args); if (value.equals(MemorySegment.NULL)) throw error(name); return value;
    }
    void release(String name, MemorySegment pointer) { call(name, null, pointer); }
    void check(String name, Object... args) { if (integer(name, args) != 0) throw error(name); }
    ComputeException error(String operation) {
        MemorySegment text = address("dg_error");
        String detail = text.equals(MemorySegment.NULL) ? "Unknown Metal error" : text.reinterpret(4096).getString(0);
        return new ComputeException(ComputeException.Code.NATIVE_ERROR, "Metal " + operation + ": " + detail);
    }
}
