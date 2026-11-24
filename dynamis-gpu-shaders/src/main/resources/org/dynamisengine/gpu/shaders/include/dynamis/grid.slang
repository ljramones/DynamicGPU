// Linear groups folded into the group-count limits (Java: kit.Grid). The dispatch has (gridX, rows, 1) groups; the
// kernel takes gridX as a parameter and skips linear groups at or past its count.
#pragma once

uint dg_linear_group(uint3 group, uint gridX)
{
    return group.y * gridX + group.x;
}

// The linear thread index of a 1-D group of size groupThreads.
uint dg_linear_thread(uint3 group, uint3 local, uint gridX, uint groupThreads)
{
    return dg_linear_group(group, gridX) * groupThreads + local.x;
}
