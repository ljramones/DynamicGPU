package org.dynamisengine.gpu.primitives;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SplittableRandom;
import java.util.function.DoubleUnaryOperator;
import org.junit.jupiter.api.Test;

/**
 * The backward passes' CPU references (1.12.0), which the GPU kernels are held to, checked against central finite
 * differences of their forward functions in double: L = sum dy y(x), so dL/dx_j is the backward pass's dx_j.
 */
class GradientReferenceTest {
    static final double H = 1e-6;

    static void close(double analytic, double numeric, double scale, String what) {
        assertTrue(Math.abs(analytic - numeric) <= 1e-6 * (scale + 1), what + ": analytic " + analytic + ", finite difference " + numeric);
    }

    @Test void softmaxBackward() {
        var r = new SplittableRandom(1);
        int n = 23, limit = 17;
        double[] s = new double[n];
        float[] dy = new float[n];
        for (int j = 0; j < n; j++) { s[j] = r.nextDouble(-3, 3); dy[j] = (float) r.nextDouble(-2, 2); }
        java.util.function.Function<double[], Double> loss = x -> {
            double mx = Double.NEGATIVE_INFINITY, sum = 0, l = 0;
            for (int j = 0; j < limit; j++) mx = Math.max(mx, x[j]);
            for (int j = 0; j < limit; j++) sum += Math.exp(x[j] - mx);
            for (int j = 0; j < limit; j++) l += dy[j] * Math.exp(x[j] - mx) / sum;
            return l;
        };
        float[] y = new float[n];
        double mx = Double.NEGATIVE_INFINITY, sum = 0;
        for (int j = 0; j < limit; j++) mx = Math.max(mx, s[j]);
        for (int j = 0; j < limit; j++) sum += Math.exp(s[j] - mx);
        for (int j = 0; j < limit; j++) y[j] = (float) (Math.exp(s[j] - mx) / sum);
        double[] dx = Softmax.backwardExact(y, 0, dy, 0, n, limit);
        for (int j = 0; j < n; j++) {
            double[] p = s.clone(), m = s.clone();
            p[j] += H; m[j] -= H;
            double fd = (loss.apply(p) - loss.apply(m)) / (2 * H);
            assertTrue(Math.abs(dx[j] - fd) <= 1e-6, "softmax dx[" + j + "]: " + dx[j] + " against " + fd);   // y is rounded to float
        }
    }

    @Test void normBackward() {
        var r = new SplittableRandom(2);
        int n = 19;
        float eps = 1e-5f;
        for (var kind : Norms.Kind.values()) {
            float[] v = new float[n], w = new float[n], dy = new float[n];
            for (int j = 0; j < n; j++) { v[j] = (float) r.nextDouble(-2, 2); w[j] = (float) r.nextDouble(-1.5, 1.5); dy[j] = (float) r.nextDouble(-1, 1); }
            double[] dw = new double[n], db = new double[n];
            double[] dv = Norms.backwardExact(kind, v, 0, n, w, dy, 0, eps, dw, db);
            for (int j = 0; j < n; j++) {
                final int jj = j;
                double fdv = fd(x -> loss(kind, v, w, dy, eps, jj, x, -1, 0), v[j]);
                close(dv[j], fdv, Math.abs(dv[j]), kind + " dv[" + j + "]");
                double fdw = fd(x -> loss(kind, v, w, dy, eps, -1, 0, jj, x), w[j]);
                close(dw[j], fdw, Math.abs(dw[j]), kind + " dWeight[" + j + "]");
            }
        }
    }

    /** sum dy y for the norm of v with element vj replaced by x (if vj >= 0) and weight wj by y (if wj >= 0). */
    static double loss(Norms.Kind kind, float[] v, float[] w, float[] dy, float eps, int vj, double x, int wj, double wy) {
        int n = v.length;
        double[] vv = new double[n];
        for (int j = 0; j < n; j++) vv[j] = j == vj ? x : v[j];
        double mean = 0, var = 0;
        if (kind == Norms.Kind.LAYER) { for (double a : vv) mean += a; mean /= n; }
        for (double a : vv) var += (a - mean) * (a - mean);
        double inv = 1 / Math.sqrt(var / n + eps), l = 0;
        for (int j = 0; j < n; j++) l += dy[j] * (vv[j] - mean) * inv * (j == wj ? wy : w[j]);
        return l;
    }

    static double fd(DoubleUnaryOperator f, double at) { return (f.applyAsDouble(at + H) - f.applyAsDouble(at - H)) / (2 * H); }

    @Test void crossEntropyGradient() {
        var r = new SplittableRandom(3);
        int n = 31, target = 9;
        float[] l = new float[n];
        for (int j = 0; j < n; j++) l[j] = (float) r.nextDouble(-4, 4);
        double lse = Reduce.logSumExpExact(l, 0, n);
        for (int j = 0; j < n; j++) {
            final int jj = j;
            double fdv = fd(x -> { float[] c = l.clone(); double lseX = 0, mx = Double.NEGATIVE_INFINITY;
                double[] d = new double[n]; for (int i = 0; i < n; i++) { d[i] = i == jj ? x : c[i]; mx = Math.max(mx, d[i]); }
                for (int i = 0; i < n; i++) lseX += Math.exp(d[i] - mx);
                return mx + Math.log(lseX) - d[target]; }, l[j]);
            close(Math.exp(l[j] - lse) - (j == target ? 1 : 0), fdv, 1, "cross-entropy dl[" + j + "]");
        }
        close(CrossEntropy.lossExact(l, 0, n, target), lse - l[target], 1, "the loss");
    }

    /** The attention gradient formulas the conformance test holds the GPU to, against finite differences of O. */
    @Test void attentionGradients() {
        var r = new SplittableRandom(4);
        int nq = 3, nk = 5, d = 4;
        double scale = 0.6;
        double[][] Q = new double[nq][d], K = new double[nk][d], V = new double[nk][d], dO = new double[nq][d];
        for (var m : new double[][][] {Q, K, V, dO}) for (var row : m) for (int c = 0; c < d; c++) row[c] = r.nextDouble(-1, 1);
        java.util.function.Supplier<Double> loss = () -> {
            double l = 0;
            for (int i = 0; i < nq; i++) {
                int limit = i + nk - nq + 1;   // causal, the queries at the end of the keys
                double[] s = new double[limit]; double mx = Double.NEGATIVE_INFINITY, sum = 0;
                for (int j = 0; j < limit; j++) { for (int c = 0; c < d; c++) s[j] += scale * Q[i][c] * K[j][c]; mx = Math.max(mx, s[j]); }
                for (int j = 0; j < limit; j++) sum += Math.exp(s[j] - mx);
                for (int c = 0; c < d; c++) { double o = 0; for (int j = 0; j < limit; j++) o += Math.exp(s[j] - mx) / sum * V[j][c]; l += dO[i][c] * o; }
            }
            return l;
        };
        double[][] gQ = new double[nq][d], gK = new double[nk][d], gV = new double[nk][d];
        for (int i = 0; i < nq; i++) {
            int limit = i + nk - nq + 1;
            double[] s = new double[limit], p = new double[limit]; double mx = Double.NEGATIVE_INFINITY, sum = 0;
            for (int j = 0; j < limit; j++) { for (int c = 0; c < d; c++) s[j] += scale * Q[i][c] * K[j][c]; mx = Math.max(mx, s[j]); }
            for (int j = 0; j < limit; j++) sum += Math.exp(s[j] - mx);
            double L = mx + Math.log(sum), D = 0;
            for (int j = 0; j < limit; j++) p[j] = Math.exp(s[j] - L);
            for (int c = 0; c < d; c++) { double o = 0; for (int j = 0; j < limit; j++) o += p[j] * V[j][c]; D += dO[i][c] * o; }
            for (int j = 0; j < limit; j++) {
                double dp = 0; for (int c = 0; c < d; c++) dp += dO[i][c] * V[j][c];
                double ds = p[j] * (dp - D);
                for (int c = 0; c < d; c++) { gQ[i][c] += scale * ds * K[j][c]; gK[j][c] += scale * ds * Q[i][c]; gV[j][c] += p[j] * dO[i][c]; }
            }
        }
        for (var pair : new Object[][] {{Q, gQ, "dQ"}, {K, gK, "dK"}, {V, gV, "dV"}}) {
            double[][] m = (double[][]) pair[0], g = (double[][]) pair[1];
            for (int a = 0; a < m.length; a++) for (int c = 0; c < d; c++) {
                double keep = m[a][c];
                m[a][c] = keep + H; double up = loss.get();
                m[a][c] = keep - H; double down = loss.get();
                m[a][c] = keep;
                close(g[a][c], (up - down) / (2 * H), Math.abs(g[a][c]), pair[2] + "[" + a + "][" + c + "]");
            }
        }
    }

    @Test void activationDerivatives() {
        double c = Math.sqrt(2 / Math.PI), k = 0.044715;
        DoubleUnaryOperator silu = x -> x / (1 + Math.exp(-x));
        DoubleUnaryOperator gelu = x -> 0.5 * x * (1 + Math.tanh(c * (x + k * x * x * x)));
        for (double x = -8; x <= 8; x += 0.37) {
            double s = 1 / (1 + Math.exp(-x));
            close(s * (1 + x * (1 - s)), fd(silu, x), 1, "silu'(" + x + ")");
            double t = Math.tanh(c * (x + k * x * x * x));
            close(0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * k * x * x), fd(gelu, x), 1, "gelu'(" + x + ")");
        }
    }

    @Test void ropeBackwardIsTheRotationsGradient() {
        // L = sum dy . rope(x) is linear in x, so dL/dx is exact: referenceBackward's dx must equal it to rounding (1.14.0)
        var r = new SplittableRandom(9);
        var table = Rope.Table.scaled(64, 16, 10000, 1);
        for (var layout : Rope.Layout.values()) {
            var shape = new Rope.Shape(2, 2, 24, 16, 48, 48, 24, layout);
            int len = 96;
            float[] dy = new float[len];
            for (int i = 0; i < len; i++) dy[i] = (float) r.nextDouble(-2, 2);
            int[] pos = {5, 63};
            float[] dx = new float[len];
            Rope.referenceBackward(dy, pos, table, dx, shape, Precision.FP32);
            for (int j = 0; j < len; j++) {
                float[] e = new float[len];
                e[j] = 1f;   // y(e_j): the j-th column of the rotation
                float[] y = new float[len];
                Rope.reference(e, pos, table, y, shape, Precision.FP32);
                double g = 0;
                for (int i = 0; i < len; i++) g += (double) dy[i] * y[i];
                close(dx[j], g, Math.abs(dy[j]) + 2, layout + " rope dx[" + j + "]");
            }
        }
    }
}
