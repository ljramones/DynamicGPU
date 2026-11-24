package org.dynamisengine.gpu.shaders;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * A kernel's ABI from slangc's reflection: storage buffers by slot, the push-constant parameter block and the group
 * size.
 *
 * @param buffers the storage buffers in slot order
 * @param parameterName the name of the push-constant parameter block, or null when the kernel has none
 * @param parameters the block's fields in declaration order
 * @param localSize the fixed group size
 */
public record KernelAbi(List<Buffer> buffers, String parameterName, List<Parameter> parameters, int[] localSize) {
    /** A storage buffer binding. */
    public record Buffer(String name, int slot) { }

    /** A field of the parameter block. */
    public record Parameter(String name, int offset, String scalarType) { }

    @SuppressWarnings("unchecked")
    static KernelAbi fromReflection(String json, String entry) {
        Map<String, Object> r = (Map<String, Object>) Json.parse(json);
        List<Buffer> buffers = new ArrayList<>();
        List<Parameter> params = new ArrayList<>();
        String paramName = null;
        for (Object o : (List<Object>) r.get("parameters")) {
            Map<String, Object> p = (Map<String, Object>) o;
            Map<String, Object> binding = (Map<String, Object>) p.get("binding");
            String kind = (String) binding.get("kind");
            if ("descriptorTableSlot".equals(kind)) {
                buffers.add(new Buffer((String) p.get("name"), ((Double) binding.get("index")).intValue()));
            } else if ("pushConstantBuffer".equals(kind)) {
                paramName = (String) p.get("name");
                Map<String, Object> type = (Map<String, Object>) p.get("type");
                Map<String, Object> element = (Map<String, Object>) type.get("elementType");
                for (Object f : (List<Object>) element.get("fields")) {
                    Map<String, Object> field = (Map<String, Object>) f;
                    Map<String, Object> fb = (Map<String, Object>) field.get("binding");
                    Map<String, Object> ft = (Map<String, Object>) field.get("type");
                    params.add(new Parameter((String) field.get("name"), ((Double) fb.get("offset")).intValue(),
                            (String) ft.get("scalarType")));
                }
            }
        }
        buffers.sort(Comparator.comparingInt(Buffer::slot));
        for (Object o : (List<Object>) r.get("entryPoints")) {
            Map<String, Object> ep = (Map<String, Object>) o;
            if (entry.equals(ep.get("name"))) {
                List<Object> size = (List<Object>) ep.get("threadGroupSize");
                int[] local = new int[size.size()];
                for (int k = 0; k < local.length; k++) local[k] = ((Double) size.get(k)).intValue();
                return new KernelAbi(List.copyOf(buffers), paramName, List.copyOf(params), local);
            }
        }
        throw new IllegalArgumentException("entry point " + entry + " not in the reflection");
    }
}
