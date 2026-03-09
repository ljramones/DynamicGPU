package org.dynamisengine.gpu.api.gpu;

import java.util.Arrays;

/**
 * Frustum plane payload for meshlet visibility testing.
 *
 * <p>Plane layout: 6 planes x 4 floats each, in {@code [a, b, c, d]} form.
 */
public final class MeshletVisibilityFrustum {
  public static final int PLANE_COUNT = 6;
  public static final int COMPONENTS_PER_PLANE = 4;
  public static final int FLOAT_COUNT = PLANE_COUNT * COMPONENTS_PER_PLANE;

  private final float[] planes24;

  private MeshletVisibilityFrustum(float[] planes24) {
    this.planes24 = planes24;
  }

  public static MeshletVisibilityFrustum of(float[] planes24) {
    if (planes24 == null) {
      throw new NullPointerException("planes24");
    }
    if (planes24.length != FLOAT_COUNT) {
      throw new IllegalArgumentException("planes24 must contain exactly " + FLOAT_COUNT + " values");
    }
    return new MeshletVisibilityFrustum(Arrays.copyOf(planes24, planes24.length));
  }

  public float[] planes24() {
    return Arrays.copyOf(planes24, planes24.length);
  }
}

