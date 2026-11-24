#!/usr/bin/env python3
"""Record glslang's layouts of random interface blocks as the golden for ShaderLayoutTest.

Needs glslangValidator and spirv-cross on PATH. Usage:
    python3 scripts/gen-shader-layout-golden.py dynamis-gpu-api/src/test/resources/gpu/shader-layout-golden.txt

Each line is  RULES|descriptor|blockSize|path@offset/arrayStride/matrixStride;...
The descriptor is the block's type in the compact syntax ShaderLayoutTest parses:
    {Name;member:type,...}   type = float|int|uint|bool|double|vecN|ivecN|uvecN|bvecN|dvecN|matCxR|dmatCxR
                             | R:matCxR (row-major) | {struct} ; followed by [n] array dimensions, outermost first
Offsets, strides and sizes come from SPIR-V reflection of the compiled shader, never from ShaderLayout.
"""
import json, os, random, subprocess, sys, tempfile

RULES = ['STD140', 'STD430', 'SCALAR']
counter = [0]


def rand_basic(r, row_major_ok):
    k = r.random()
    if k < 0.3:
        return r.choice(['float', 'int', 'uint', 'float', 'double', 'bool'])
    if k < 0.65:
        n = r.randint(2, 4)
        return r.choice(['', '', 'i', 'u', 'd', 'b']) + 'vec' + str(n)
    c, rows = r.randint(2, 4), r.randint(2, 4)
    t = ('d' if r.random() < 0.2 else '') + 'mat%dx%d' % (c, rows)
    # GLSL allows layout(row_major) on block members only, not inside struct definitions
    return ('R:' if row_major_ok and r.random() < 0.3 else '') + t


def rand_type(r, depth, in_block):
    if depth < 2 and r.random() < 0.18:
        t = rand_struct(r, depth + 1)
    else:
        t = rand_basic(r, in_block)
    dims = ''
    if r.random() < 0.3:
        dims += '[%d]' % r.randint(1, 4)
        if r.random() < 0.2:
            dims += '[%d]' % r.randint(1, 3)
    return t, dims


def rand_struct(r, depth, in_block=False):
    counter[0] += 1
    name = 'S%d' % counter[0]
    n = r.randint(1, 6)
    members = []
    for i in range(n):
        t, dims = rand_type(r, depth, in_block)
        members.append(('m%d' % i, t, dims))
    return {'name': name, 'members': members}


def desc(t):
    if isinstance(t, dict):
        return '{' + t['name'] + ';' + ','.join(m + ':' + desc(tt) + d for m, tt, d in t['members']) + '}'
    return t


def struct_decls(t, out, seen):
    """Struct definitions used by t, dependencies first."""
    for _, tt, _ in t['members']:
        if isinstance(tt, dict) and tt['name'] not in seen:
            struct_decls(tt, out, seen)
            seen.add(tt['name'])
            out.append(tt)


def member_glsl(m, tt, d):
    if isinstance(tt, dict):
        return '%s %s%s;' % (tt['name'], m, d)
    q = ''
    if tt.startswith('R:'):
        q, tt = 'layout(row_major) ', tt[2:]
    return '%s%s %s%s;' % (q, tt, m, d)


def shader(block, rules):
    lines = ['#version 450', '#extension GL_EXT_scalar_block_layout : require', 'layout(local_size_x = 1) in;']
    decls = []
    struct_decls(block, decls, set())
    for st in decls:
        lines.append('struct %s {' % st['name'])
        lines += ['    ' + member_glsl(*m) for m in st['members']]
        lines.append('};')
    lines.append('layout(%s, binding = 0) buffer %s {' % (rules.lower(), block['name']))
    lines += ['    ' + member_glsl(*m) for m in block['members']]
    lines.append('} blk;')
    lines.append('void main() { }')  # glslang keeps unused storage blocks
    return '\n'.join(lines) + '\n'


def reflect(src):
    with tempfile.TemporaryDirectory() as d:
        comp, spv = os.path.join(d, 't.comp'), os.path.join(d, 't.spv')
        open(comp, 'w').write(src)
        res = subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.2', comp, '-o', spv],
                             capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit('glslang rejected:\n' + src + res.stdout + res.stderr)
        out = subprocess.run(['spirv-cross', spv, '--reflect'], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def walk(types, tid, prefix, base, out):
    for mem in types[tid]['members']:
        path = prefix + mem['name']
        off = base + mem['offset']
        stride = mem.get('array_stride', 0)
        mstride = mem.get('matrix_stride', 0)
        out.append('%s@%d/%d/%d' % (path, off, stride, mstride))
        if mem['type'] in types:
            walk(types, mem['type'], path + '[0]' * len(mem.get('array', [])) + '.', off, out)


def main():
    dst = sys.argv[1]
    r = random.Random(20261122)
    lines = []
    for i in range(150):
        counter[0] = 0
        block = rand_struct(r, 0, True)
        block['name'] = 'Block'
        for rules in RULES:
            refl = reflect(shader(block, rules))
            ssbo = refl['ssbos'][0]
            entries = []
            walk(refl['types'], ssbo['type'], '', 0, entries)
            lines.append('%s|%s|%d|%s' % (rules, desc(block), ssbo['block_size'], ';'.join(entries)))
    with open(dst, 'w') as f:
        f.write('# glslang layouts of random interface blocks; regenerate with scripts/gen-shader-layout-golden.py\n')
        f.write('\n'.join(lines) + '\n')
    print('wrote %d layouts to %s' % (len(lines), dst))


if __name__ == '__main__':
    main()
