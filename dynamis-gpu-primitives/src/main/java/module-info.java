/** Throughput kernels on the DynamisGPU compute API: accuracy-bounded, not bit-exact. */
module org.dynamisengine.gpu.primitives {
    requires transitive org.dynamisengine.gpu.api;
    requires org.dynamisengine.vectrix.ml;
    exports org.dynamisengine.gpu.primitives;
}
