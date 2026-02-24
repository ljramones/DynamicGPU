package org.dynamisgpu.api.gpu;

public interface DrawMetaWriter {
  void write(
      int drawIndex,
      int jointPaletteIndex,
      int morphDeltaIndex,
      int morphWeightIndex,
      int instanceDataIndex,
      int materialIndex,
      int drawFlags,
      int meshIndex);

  void flush();

  void destroy();

  int capacity();
}
