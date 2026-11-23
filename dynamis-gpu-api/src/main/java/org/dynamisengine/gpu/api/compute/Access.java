package org.dynamisengine.gpu.api.compute;
public enum Access { READ, WRITE, READ_WRITE;
    public boolean writes() { return this != READ; }
}
