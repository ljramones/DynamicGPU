module org.dynamisengine.gpu.test {
    requires transitive org.dynamisengine.gpu.api;
    requires transitive org.junit.jupiter.api;

    exports org.dynamisengine.gpu.test.mock;
    exports org.dynamisengine.gpu.test.compute;
    opens org.dynamisengine.gpu.test.compute to org.junit.platform.commons;
}
