/**
 * Backend implementor extension. Drivers receive validated operations and retained opaque
 * resources. A job may release command storage only after terminal completion or device loss;
 * errors from polling must describe a terminal operation. Driver cleanup is idempotent at
 * the shared device layer. User applications should use the parent compute package.
 */
package org.dynamisengine.gpu.api.compute.spi;
