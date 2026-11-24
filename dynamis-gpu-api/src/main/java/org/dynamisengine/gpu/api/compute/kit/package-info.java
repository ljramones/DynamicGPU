/**
 * The kernel kit: the host-side plumbing every kernel family shares, so that none re-implements it.
 * <ul>
 * <li>{@link org.dynamisengine.gpu.api.compute.kit.ParameterLayout}: parameter blocks written by field name and
 * checked against the kernel's manifest.</li>
 * <li>{@link org.dynamisengine.gpu.api.compute.kit.StridedView}: an element view of a buffer (offset, extents,
 * strides) covering dense, sliced, transposed, batched and interleaved layouts, with one bounds check.</li>
 * <li>{@link org.dynamisengine.gpu.api.compute.kit.Grid}: linear group ranges folded into the device's group-count
 * limits.</li>
 * <li>{@link org.dynamisengine.gpu.api.compute.kit.PipelineSet}: pipelines compiled and closed together, with a
 * placeholder buffer for absent optional arguments.</li>
 * <li>{@link org.dynamisengine.gpu.api.compute.kit.Checks}: argument checks and the 32-bit index limit.</li>
 * </ul>
 * The matching Slang side ships with {@code dynamis-gpu-shaders} ({@code dynamis/kit.slang}).
 *
 * @since 1.7.0
 */
package org.dynamisengine.gpu.api.compute.kit;
