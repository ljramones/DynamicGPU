package org.dynamisengine.gpu.test.compute;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract.Bound;
import org.junit.jupiter.api.Test;

class CrossBackendTest {
    static AccuracyContract contract(Bound bound) {
        return new AccuracyContract("op", List.of("FP32"), "none", "FP32", AccuracyContract.Order.FIXED, bound,
                bound.kind() == Bound.Kind.EXACT ? "Ref.ref" : "");
    }

    @Test void exactContractsLocateTheFirstDifference() {
        float[] a = {1f, Float.NaN, 0f, 3f, 4f}, b = {1f, Float.NaN, -0f, Math.nextUp(3f), 5f};
        var r = CrossBackend.compare(contract(Bound.exact()), a, b);
        assertFalse(r.withinContract());
        assertEquals(3, r.differing(), "-0 vs +0 differs in its bits");
        assertEquals(2, r.firstBeyond().orElseThrow().index());
        assertTrue(CrossBackend.compare(contract(Bound.exact()), a, a.clone()).withinContract());
    }

    @Test void ulpContractsAllowTwiceTheBound() {
        float[] a = {1f, 2f}, b = {Math.nextUp(Math.nextUp(1f)), Math.nextUp(Math.nextUp(Math.nextUp(2f)))};
        var r = CrossBackend.compare(contract(Bound.ulps(1, "1 ULP")), a, b);
        assertEquals(1, r.firstBeyond().orElseThrow().index(), "2 ULP apart is within twice 1 ULP; 3 is not");
        assertEquals(3, r.worstUlps());
    }

    @Test void formulaContractsTakeAToleranceAndInfinitiesNeverMatch() {
        var c = contract(Bound.formula("|err| <= 0.1"));
        assertThrows(IllegalArgumentException.class, () -> CrossBackend.compare(c, new float[1], new float[1]));
        assertTrue(CrossBackend.compare(c, new float[] {1f}, new float[] {1.05f}, i -> 0.1).withinContract());
        assertFalse(CrossBackend.compare(c, new float[] {Float.POSITIVE_INFINITY}, new float[] {1f}, i -> 0.1).withinContract());
    }
}
