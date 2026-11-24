package org.dynamisengine.gpu.shaders;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** The Metal output: the pragmas that keep every rounding, and the check that buffer indices follow the slots. */
public final class Metal {
    /** Without these the Metal compiler fuses multiply-adds even with fast math disabled. */
    public static final String PRELUDE = "#pragma METAL fp math_mode(safe)\n#pragma METAL fp contract(off)\n";

    private Metal() { }

    /**
     * Checks that the kernel's Metal buffer indices equal its slots and that the parameter block takes the next
     * index, as DynamisGPU's {@code BindingLayout.parameterSlot()} expects. Declaring buffers in slot order with the
     * parameter block last gives that.
     */
    static void checkBindings(String metal, String entry, KernelAbi abi) {
        Matcher sig = Pattern.compile("\\[\\[kernel\\]\\]\\s+void\\s+" + Pattern.quote(entry) + "\\((.*?)\\)\\s*\\{",
                Pattern.DOTALL).matcher(metal);
        if (!sig.find()) throw new IllegalStateException(entry + ": kernel signature not found in the Metal output");
        Map<String, Integer> found = new HashMap<>();
        Matcher b = Pattern.compile("\\*\\s*(\\w+?)_\\d+\\s*\\[\\[buffer\\((\\d+)\\)\\]\\]").matcher(sig.group(1));
        while (b.find()) found.put(b.group(1), Integer.parseInt(b.group(2)));
        Map<String, Integer> expect = new HashMap<>();
        int max = -1;
        for (var buffer : abi.buffers()) { expect.put(buffer.name(), buffer.slot()); max = Math.max(max, buffer.slot()); }
        if (abi.parameterName() != null) expect.put(abi.parameterName(), max + 1);
        if (!found.equals(expect)) {
            throw new IllegalStateException(entry + ": Metal buffer indices " + found + " differ from slots " + expect);
        }
    }
}
