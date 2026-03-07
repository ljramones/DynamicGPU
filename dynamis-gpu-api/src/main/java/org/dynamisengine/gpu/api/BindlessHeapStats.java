package org.dynamisengine.gpu.api;

/**
 * Snapshot of bindless GPU heap utilization and lifecycle counters.
 *
 * @param jointUsed currently allocated joint-palette slots
 * @param jointCapacity total joint-palette capacity
 * @param morphDeltaUsed currently allocated morph-delta slots
 * @param morphDeltaCapacity total morph-delta capacity
 * @param morphWeightUsed currently allocated morph-weight slots
 * @param morphWeightCapacity total morph-weight capacity
 * @param instanceUsed currently allocated instance-data slots
 * @param instanceCapacity total instance-data capacity
 * @param allocations total handle allocations performed
 * @param freesQueued handles queued for deferred free
 * @param freesRetired handles actually retired from deferred free queue
 * @param staleHandleRejects writes rejected due to stale/invalid handles
 * @param drawMetaCount draw metadata entries tracked in the current frame
 * @param invalidIndexWrites attempted writes targeting invalid indices
 */
public record BindlessHeapStats(
        int jointUsed,
        int jointCapacity,
        int morphDeltaUsed,
        int morphDeltaCapacity,
        int morphWeightUsed,
        int morphWeightCapacity,
        int instanceUsed,
        int instanceCapacity,
        long allocations,
        long freesQueued,
        long freesRetired,
        long staleHandleRejects,
        int drawMetaCount,
        int invalidIndexWrites
) {
}
