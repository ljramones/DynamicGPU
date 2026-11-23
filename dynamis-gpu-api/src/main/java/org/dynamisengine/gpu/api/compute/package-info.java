/**
 * Headless compute contract, version 1. Devices are explicitly selected and confined to
 * their creating thread. Encoders snapshot host data and retain every referenced native
 * allocation and pipeline. One ordered queue inserts transfer/compute visibility barriers;
 * poll or await submissions to publish readbacks and retire resources. Closing a device
 * drains work with a bounded wait and invalidates all remaining child handles.
 *
 * <p>Buffer slices retain storage independently. Transfer ranges are multiples of four bytes;
 * storage offsets meet the reported device alignment. Binding slots and scalar parameter
 * byte layouts must agree with the supplied shader. FP32, INT32 and BOOL (32-bit mask)
 * form the portable numerical baseline. Backend handles never enter this package.
 */
package org.dynamisengine.gpu.api.compute;
