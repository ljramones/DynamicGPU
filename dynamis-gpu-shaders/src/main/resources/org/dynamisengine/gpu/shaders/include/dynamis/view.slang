// Strided views (Java: kit.StridedView and ParameterLayout.Block.view). A view `x` of rank R is passed as flat
// parameter-block fields xOffset, xExtent0, xStride0, ... xExtent<R-1>, xStride<R-1> (all uint), declared with
// DG_VIEW_FIELDS2/4 and read into a DgView with DG_VIEW2/4. Dimension 0 is the outermost.
#pragma once

#define DG_VIEW_FIELDS1(x) uint x##Offset; uint x##Extent0; uint x##Stride0;
#define DG_VIEW_FIELDS2(x) uint x##Offset; uint x##Extent0; uint x##Stride0; uint x##Extent1; uint x##Stride1;
#define DG_VIEW_FIELDS4(x) uint x##Offset; uint x##Extent0; uint x##Stride0; uint x##Extent1; uint x##Stride1; \
    uint x##Extent2; uint x##Stride2; uint x##Extent3; uint x##Stride3;

// A view of rank up to 4; unused leading dimensions have extent 1.
struct DgView
{
    uint offset;
    uint4 extent;
    uint4 stride;
};

#define DG_VIEW1(p, x) dg_view(p.x##Offset, uint4(1, 1, 1, p.x##Extent0), uint4(0, 0, 0, p.x##Stride0))
#define DG_VIEW2(p, x) dg_view(p.x##Offset, uint4(1, 1, p.x##Extent0, p.x##Extent1), uint4(0, 0, p.x##Stride0, p.x##Stride1))
#define DG_VIEW4(p, x) dg_view(p.x##Offset, uint4(p.x##Extent0, p.x##Extent1, p.x##Extent2, p.x##Extent3), \
    uint4(p.x##Stride0, p.x##Stride1, p.x##Stride2, p.x##Stride3))

DgView dg_view(uint offset, uint4 extent, uint4 stride)
{
    DgView v;
    v.offset = offset;
    v.extent = extent;
    v.stride = stride;
    return v;
}

// The number of elements (the host keeps it within 32 bits).
uint dg_count(DgView v)
{
    return v.extent.x * v.extent.y * v.extent.z * v.extent.w;
}

// The element index of (i0, i1, i2, i3).
uint dg_at(DgView v, uint4 i)
{
    return v.offset + i.x * v.stride.x + i.y * v.stride.y + i.z * v.stride.z + i.w * v.stride.w;
}

// The indices of the linear (row-major) element `linear` of the view.
uint4 dg_coords(DgView v, uint linear)
{
    uint4 i;
    i.w = linear % v.extent.w; linear /= v.extent.w;
    i.z = linear % v.extent.z; linear /= v.extent.z;
    i.y = linear % v.extent.y; linear /= v.extent.y;
    i.x = linear;
    return i;
}

// The (i0, i1, i2) of row `row` of a view whose last dimension is the row: it never divides by the row length,
// so it holds for empty rows too.
uint3 dg_row(DgView v, uint row)
{
    uint3 i;
    i.z = row % v.extent.z; row /= v.extent.z;
    i.y = row % v.extent.y;
    i.x = row / v.extent.y;
    return i;
}

// The element index of the linear (row-major) element `linear` of the view.
uint dg_at_linear(DgView v, uint linear)
{
    return dg_at(v, dg_coords(v, linear));
}

// The same position in another view of the shape, broadcasting along dimensions of extent 1 (stride ignored there).
uint dg_at_broadcast(DgView v, uint4 i)
{
    uint4 j = select(v.extent == uint4(1, 1, 1, 1), uint4(0, 0, 0, 0), i);
    return dg_at(v, j);
}
