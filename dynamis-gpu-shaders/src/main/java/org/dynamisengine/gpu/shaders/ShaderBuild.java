package org.dynamisengine.gpu.shaders;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Compiles Slang compute kernels to SPIR-V (Vulkan) and Metal source so that every floating-point operation rounds as
 * written, and records their ABI in a manifest that {@code KernelLibrary} (dynamis-gpu-api) loads at runtime.
 * <ul>
 * <li>slangc runs at {@code -O0}: its optimizer folds {@code (x + c) - c} to {@code x} in SPIR-V, which IEEE
 * arithmetic forbids.</li>
 * <li>SPIR-V: {@link Spirv#noContraction} marks every float add, subtract and fma NoContraction.</li>
 * <li>Metal: the source starts with {@link Metal#PRELUDE}.</li>
 * <li>Declare buffers in slot order with the push-constant parameter block last; the build checks that the Metal
 * buffer indices follow (DynamisGPU's {@code BindingLayout.parameterSlot()}).</li>
 * </ul>
 * The manifest has one {@code key=value} line per entry, sorted: {@code slangc.version}, {@code source.sha256} (over
 * every {@code .slang} file in the hash directories) and, per kernel, {@code entry}, {@code local}, {@code buffers}
 * ({@code name:slot}), {@code params} ({@code name:offset:type}), {@code spirv.sha256} and {@code metal.sha256}.
 * <p>
 * Usage: {@code ShaderBuild --src DIR --out DIR --kernels FILE [--manifest NAME] [--include DIR]... [--hash DIR]...
 * [--slangc PATH]}. The kernels file has one {@code name source entry} line per kernel ({@code #} starts a comment).
 * Include directories are passed to slangc in order, followed by the source directory; the hash directories default
 * to the source directory. slangc is found through {@code --slangc}, {@code $SLANGC}, {@code $VULKAN_SDK/bin}, the
 * newest {@code ~/VulkanSDK/x/macOS/bin}, then {@code PATH}.
 */
public final class ShaderBuild {
    /** One kernel to build: the output name, the source file (relative to the source directory) and the entry point. */
    public record Kernel(String name, String source, String entry) { }

    /** A build: sources, outputs and the slangc to use. */
    public record Config(Path src, Path out, String manifestName, List<Path> includes, List<Path> hashDirs,
            List<Kernel> kernels, Path slangc) { }

    private ShaderBuild() { }

    public static void main(String[] args) throws IOException {
        Path src = null, out = null, kernels = null, slangc = null;
        String manifest = "kernels.properties";
        List<Path> includes = new ArrayList<>(), hashes = new ArrayList<>();
        for (int i = 0; i < args.length; i++) {
            String v = i + 1 < args.length ? args[i + 1] : null;
            switch (args[i]) {
                case "--src" -> { src = Path.of(v); i++; }
                case "--out" -> { out = Path.of(v); i++; }
                case "--kernels" -> { kernels = Path.of(v); i++; }
                case "--manifest" -> { manifest = v; i++; }
                case "--include" -> { includes.add(Path.of(v)); i++; }
                case "--hash" -> { hashes.add(Path.of(v)); i++; }
                case "--slangc" -> { slangc = Path.of(v); i++; }
                default -> throw new IllegalArgumentException("unknown argument " + args[i]);
            }
        }
        if (src == null || out == null || kernels == null) {
            throw new IllegalArgumentException("usage: ShaderBuild --src DIR --out DIR --kernels FILE [--manifest NAME]"
                    + " [--include DIR]... [--hash DIR]... [--slangc PATH]");
        }
        build(new Config(src, out, manifest, includes, hashes.isEmpty() ? List.of(src) : hashes, readKernels(kernels),
                slangc));
    }

    /** Reads a kernels file: {@code name source entry} per line. */
    public static List<Kernel> readKernels(Path file) throws IOException {
        List<Kernel> list = new ArrayList<>();
        for (String line : Files.readAllLines(file)) {
            String t = line.replaceAll("#.*", "").trim();
            if (t.isEmpty()) continue;
            String[] f = t.split("\\s+");
            if (f.length != 3) throw new IllegalArgumentException("expected 'name source entry': " + line);
            list.add(new Kernel(f[0], f[1], f[2]));
        }
        return list;
    }

    /** Builds every kernel and writes the outputs and the manifest. */
    public static void build(Config c) throws IOException {
        Path slangc = c.slangc() != null ? c.slangc() : findSlangc().orElseThrow(
                () -> new IllegalStateException("slangc not found (set SLANGC or VULKAN_SDK, or pass --slangc)"));
        String version = firstLine(run(List.of(slangc.toString(), "-v"), true));
        Optional<Path> spirvVal = which("spirv-val").or(() -> exists(slangc.resolveSibling("spirv-val")));
        Files.createDirectories(c.out());
        Map<String, String> manifest = new TreeMap<>();
        manifest.put("slangc.version", version);
        manifest.put("source.sha256", sourceHash(c.hashDirs()));
        Path tmp = Files.createTempDirectory("dynamis-shaders");
        try {
            for (Kernel k : c.kernels()) {
                List<String> common = new ArrayList<>(List.of(slangc.toString(), c.src().resolve(k.source()).toString()));
                for (Path inc : c.includes()) { common.add("-I"); common.add(inc.toString()); }
                common.addAll(List.of("-I", c.src().toString(), "-entry", k.entry(), "-stage", "compute", "-O0",
                        "-line-directive-mode", "none"));
                Path spvTmp = tmp.resolve(k.name() + ".spv"), refl = tmp.resolve(k.name() + ".json");
                List<String> spirvCmd = new ArrayList<>(common);
                spirvCmd.addAll(List.of("-target", "spirv", "-fvk-use-entrypoint-name", "-reflection-json",
                        refl.toString(), "-o", spvTmp.toString()));
                run(spirvCmd, false);
                byte[] spv = Spirv.noContraction(Files.readAllBytes(spvTmp));
                Path spvOut = c.out().resolve(k.name() + ".spv");
                Files.write(spvOut, spv);
                if (spirvVal.isPresent()) {
                    run(List.of(spirvVal.get().toString(), spvOut.toString()), false);
                } else {
                    System.err.println("warning: spirv-val not found; SPIR-V not validated");
                }
                Path metalTmp = tmp.resolve(k.name() + ".metal");
                List<String> metalCmd = new ArrayList<>(common);
                metalCmd.addAll(List.of("-target", "metal", "-o", metalTmp.toString()));
                run(metalCmd, false);
                String metal = Metal.PRELUDE + Files.readString(metalTmp);
                KernelAbi abi = KernelAbi.fromReflection(Files.readString(refl), k.entry());
                Metal.checkBindings(metal, k.entry(), abi);
                Files.writeString(c.out().resolve(k.name() + ".metal"), metal);
                manifest.put(k.name() + ".entry", k.entry());
                manifest.put(k.name() + ".local", Arrays.stream(abi.localSize()).mapToObj(Integer::toString)
                        .collect(Collectors.joining(",")));
                manifest.put(k.name() + ".buffers", abi.buffers().stream().map(b -> b.name() + ":" + b.slot())
                        .collect(Collectors.joining(",")));
                manifest.put(k.name() + ".params", abi.parameters().stream()
                        .map(p -> p.name() + ":" + p.offset() + ":" + p.scalarType()).collect(Collectors.joining(",")));
                manifest.put(k.name() + ".spirv.sha256", sha256(spv));
                manifest.put(k.name() + ".metal.sha256", sha256(metal.getBytes(StandardCharsets.UTF_8)));
                System.out.printf("%s: %d bytes SPIR-V, %d chars Metal, local %s, buffers %s%n", k.name(), spv.length,
                        metal.length(), Arrays.toString(abi.localSize()), abi.buffers());
            }
        } finally {
            try (Stream<Path> files = Files.walk(tmp)) {
                files.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
            }
        }
        StringBuilder text = new StringBuilder("# Generated by dynamis-gpu-shaders (ShaderBuild); do not edit.\n");
        manifest.forEach((k, v) -> text.append(k).append('=').append(v).append('\n'));
        Files.writeString(c.out().resolve(c.manifestName()), text.toString());
    }

    /**
     * The hash recorded as {@code source.sha256}: SHA-256 over every {@code .slang} file of the directories, sorted by
     * path, each as its file name, a zero byte, its contents and a zero byte. A test can recompute it to check that
     * committed outputs match their sources.
     */
    public static String sourceHash(List<Path> dirs) throws IOException {
        List<Path> files = new ArrayList<>();
        for (Path d : dirs) {
            try (Stream<Path> s = Files.list(d)) {
                files.addAll(s.filter(p -> p.getFileName().toString().endsWith(".slang")).toList());
            }
        }
        files.sort(Comparator.comparing(p -> p.toAbsolutePath().toString()));
        MessageDigest h = digest();
        for (Path f : files) {
            h.update(f.getFileName().toString().getBytes(StandardCharsets.UTF_8));
            h.update((byte) 0);
            h.update(Files.readAllBytes(f));
            h.update((byte) 0);
        }
        return HexFormat.of().formatHex(h.digest());
    }

    static Optional<Path> findSlangc() {
        List<Path> candidates = new ArrayList<>();
        Optional.ofNullable(System.getenv("SLANGC")).map(Path::of).ifPresent(candidates::add);
        Optional.ofNullable(System.getenv("VULKAN_SDK")).map(v -> Path.of(v, "bin", "slangc")).ifPresent(candidates::add);
        Path sdks = Path.of(System.getProperty("user.home"), "VulkanSDK");
        if (Files.isDirectory(sdks)) {
            try (Stream<Path> s = Files.list(sdks)) {
                s.map(p -> p.resolve("macOS/bin/slangc")).sorted(Comparator.comparing(Path::toString).reversed())
                        .forEach(candidates::add);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        which("slangc").ifPresent(candidates::add);
        for (Path p : candidates) {
            if (Files.isRegularFile(p)) {
                try {
                    run(List.of(p.toString(), "-v"), true);
                    return Optional.of(p);
                } catch (IllegalStateException broken) {
                    // for example a slangc whose libraries are missing: try the next one
                }
            }
        }
        return Optional.empty();
    }

    private static Optional<Path> which(String tool) {
        String path = System.getenv("PATH");
        if (path == null) return Optional.empty();
        for (String dir : path.split(java.io.File.pathSeparator)) {
            Path p = Path.of(dir, tool);
            if (Files.isExecutable(p)) return Optional.of(p);
        }
        return Optional.empty();
    }

    private static Optional<Path> exists(Path p) { return Files.isExecutable(p) ? Optional.of(p) : Optional.empty(); }

    private static String firstLine(String s) { return s.strip().lines().findFirst().orElse(""); }

    /** Runs a command; fails with its output unless it exits 0. Returns stdout and stderr together. */
    private static String run(List<String> cmd, boolean quiet) {
        try {
            Process p = new ProcessBuilder(cmd).redirectErrorStream(true).start();
            String output = new String(p.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            if (p.waitFor() != 0) throw new IllegalStateException("command failed: " + String.join(" ", cmd) + "\n" + output);
            return output;
        } catch (IOException e) {
            throw new IllegalStateException("cannot run " + cmd.getFirst(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted running " + cmd.getFirst(), e);
        }
    }

    private static String sha256(byte[] b) { return HexFormat.of().formatHex(digest().digest(b)); }

    private static MessageDigest digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }
}
