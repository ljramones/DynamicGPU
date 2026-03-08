package org.dynamisengine.gpu.bench.ingest.meshforge;

import org.dynamisengine.meshforge.api.Meshes;
import org.dynamisengine.meshforge.api.Packers;
import org.dynamisengine.meshforge.core.mesh.MeshData;
import org.dynamisengine.meshforge.gpu.MeshForgeGpuBridge;
import org.dynamisengine.meshforge.gpu.RuntimeGeometryPayload;
import org.dynamisengine.meshforge.pack.buffer.PackedMesh;
import org.dynamisengine.meshforge.pack.packer.MeshPacker;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MeshForgeRuntimePlanAdapterTest {
  @Test
  void mapsMeshForgePayloadToApiUploadPlan() {
    MeshData mesh = Meshes.cube(1.0f);
    PackedMesh packed = MeshPacker.pack(mesh, Packers.realtimeFast());
    RuntimeGeometryPayload payload = MeshForgeGpuBridge.payloadFromPackedMesh(packed);

    var plan = MeshForgeRuntimePlanAdapter.toApiPlan(payload);

    assertNotNull(plan.vertexLayout());
    assertEquals(payload.layout().strideBytes(), plan.vertexLayout().strideBytes());
    assertEquals(payload.vertexBytes().remaining(), plan.vertexData().remaining());
    assertEquals(payload.submeshes().size(), plan.submeshes().size());
    assertTrue(plan.submeshes().stream().allMatch(s -> s.baseVertex() == 0));

    if (payload.indexBytes() != null && payload.indexType() != null) {
      assertNotNull(plan.indexData());
      assertEquals(payload.indexBytes().remaining(), plan.indexData().remaining());
      assertNotNull(plan.indexType());
    }
  }
}
