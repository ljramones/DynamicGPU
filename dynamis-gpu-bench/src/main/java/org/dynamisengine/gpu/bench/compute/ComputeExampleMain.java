package org.dynamisengine.gpu.bench.compute;

import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.metal.MetalProvider;
import org.dynamisengine.gpu.vulkan.compute.VulkanProvider;
import java.io.*;
import java.nio.*;
import java.time.Duration;
import java.util.*;

/** Public compute API example: (a+b)+b with only one final host readback. */
public final class ComputeExampleMain {
    private ComputeExampleMain() { }
    static GpuProvider provider(String name) {
        return switch (name.toLowerCase(Locale.ROOT)) {
            case "metal" -> new MetalProvider(); case "vulkan" -> new VulkanProvider();
            default -> throw new IllegalArgumentException("Select metal or vulkan explicitly");
        };
    }
    static GpuDevice device(String backend, int index) {
        var adapters = provider(backend).adapters();
        if (index < 0 || index >= adapters.size()) throw new ComputeException(ComputeException.Code.UNAVAILABLE, "No selected " + backend + " device at index " + index);
        return adapters.get(index).open();
    }
    static KernelBundle kernel() {
        try {
            var executables = new EnumMap<Backend, KernelBundle.Executable>(Backend.class);
            executables.put(Backend.VULKAN, new KernelBundle.Executable(KernelBundle.Format.SPIRV, resource("/compute/vector_add.spv")));
            String arch = System.getProperty("os.arch").equals("aarch64") ? "arm64" : "x86_64";
            try (var binary = MetalProvider.class.getResourceAsStream("/native/macos-" + arch + "/vector_add.metallib")) {
                executables.put(Backend.METAL, binary == null
                        ? new KernelBundle.Executable(KernelBundle.Format.MSL_SOURCE, resource("/compute/vector_add.metal"))
                        : new KernelBundle.Executable(KernelBundle.Format.METALLIB, binary.readAllBytes()));
            }
            return new KernelBundle("vector-add", "1", executables);
        } catch (IOException ex) { throw new UncheckedIOException(ex); }
    }
    private static byte[] resource(String path) throws IOException {
        try (var stream = ComputeExampleMain.class.getResourceAsStream(path)) {
            if (stream == null) throw new FileNotFoundException(path); return stream.readAllBytes();
        }
    }
    static GpuBuffer buffer(GpuDevice device, int n, Placement placement) {
        return device.allocate(new BufferDescriptor(Math.multiplyExact(n, 4L), EnumSet.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE, BufferUsage.TRANSFER_DESTINATION), placement));
    }
    static GpuComputePipeline pipeline(GpuDevice device, int n) {
        var layout = new BindingLayout(List.of(new BindingLayout.BufferBinding(0, Access.READ, n * 4L),
                new BindingLayout.BufferBinding(1, Access.READ, n * 4L), new BindingLayout.BufferBinding(2, Access.WRITE, n * 4L)), 4);
        var bundle = kernel();
        System.out.println("kernel=" + bundle.name() + " version=" + bundle.version() + " executable=" + bundle.executable(device.adapter().backend()).format());
        return device.pipeline(new PipelineDescriptor(bundle, "vector_add", layout, new Dimensions(64, 1, 1), Map.of(), Set.of("FP32"), Set.of()));
    }
    static byte[] scalar(int n) { return ByteBuffer.allocate(4).order(ByteOrder.nativeOrder()).putInt(n).array(); }
    static void await(GpuSubmission submission) {
        if (!submission.await(Duration.ofSeconds(20))) throw new ComputeException(ComputeException.Code.TIMEOUT, "Compute completion exceeded 20 seconds");
    }
    static void diagnostics(GpuDevice device, int n) {
        System.out.println("backend=" + device.adapter().backend() + " device=" + device.adapter().name() + " id=" + device.adapter().id());
        System.out.println("java=" + System.getProperty("java.version") + " os=" + System.getProperty("os.name") + " " + System.getProperty("os.version")
                + " arch=" + System.getProperty("os.arch") + " elements=" + n + " bytesPerBuffer=" + n * 4L + " capabilities=" + device.capabilities());
    }
    public static void main(String[] args) {
        if (args.length == 0) throw new IllegalArgumentException("Arguments: metal|vulkan [deviceIndex=0] [elements=131]");
        int index = args.length > 1 ? Integer.parseInt(args[1]) : 0;
        int n = args.length > 2 ? Integer.parseInt(args[2]) : 131; if (n <= 0) throw new IllegalArgumentException("Positive element count required");
        try (var device = device(args[0], index); var a = buffer(device, n, Placement.SHARED); var b = buffer(device, n, Placement.PRIVATE);
             var c = buffer(device, n, Placement.PRIVATE); var d = buffer(device, n, Placement.SHARED); var p = pipeline(device, n); var commands = device.encoder()) {
            diagnostics(device, n);
            var input = ByteBuffer.allocate(Math.multiplyExact(n, 4)).order(ByteOrder.nativeOrder()); for (int i = 0; i < n; i++) input.putFloat(i * 0.25f);
            commands.upload(a, input.array()).upload(b, input.array());
            var groups = new Dimensions(Math.toIntExact((n + 63L) / 64), 1, 1);
            commands.dispatch(p, Map.of(0, a, 1, b, 2, c), scalar(n), groups);
            commands.dispatch(p, Map.of(0, c, 1, b, 2, d), scalar(n), groups);
            var read = commands.download(d); await(device.queue().submit(commands));
            var result = ByteBuffer.wrap(read.bytes()).order(ByteOrder.nativeOrder());
            for (int i = 0; i < n; i++) if (result.getFloat() != i * 0.75f) throw new AssertionError("Mismatch at " + i);
            System.out.println("Verified " + n + " FP32 results from two dependent GPU dispatches.");
        }
    }
}
