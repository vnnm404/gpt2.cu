"""Compile small MLP instruction regions into exact rectangular tile DAGs.

No polling dependencies are assigned to a worker. The device queue contains
only ready tasks; unrelated operations can progress while a row is incomplete.
"""
import ctypes as ct
import os
from collections import defaultdict

import torch


def rectangle(pointer, stride, r0, r1, c0, c1):
    return (pointer, stride, r0, r1, c0, c1)


def overlaps(a, b):
    assert a[1] == b[1], 'A buffer must have one physical row stride within a region'
    return a[2] < b[3] and b[2] < a[3] and a[4] < b[5] and b[4] < a[5]


def compile_region(program, begin, end):
    from . import Code, Operation
    ops = program.ops[begin:end]
    shapes = program.shapes
    nodes, parents, footprints = [], [], []
    readers, writers = defaultdict(list), defaultdict(list)
    for oi, op in enumerate(ops):
        current = []
        if op.code == Code.GEMM:
            count = op.tiles
        elif op.code in (Code.GELU, Code.GELU_BACKWARD, Code.SUM_SPLITS):
            dest = op.p[1 if op.code in (Code.GELU, Code.SUM_SPLITS) else 2]
            shape = shapes[dest]
            assert len(shape) == 2
            rows, columns = shape
            count = ((rows + 63) // 64) * ((columns + 127) // 128)
        elif op.code == Code.SUM_ROWS:
            assert not op.p[1] and not op.p[2]
            count = op.tiles
        else:
            raise ValueError(f'Unsupported tile instruction: {Code(op.code).name}')
        for tile in range(count):
            read, write, rect_columns = [], [], 0
            if op.code == Code.GEMM:
                nr, nc = (op.m + 63) // 64, (op.n + 127) // 128
                split, t = divmod(tile, nr * nc)
                first = t // (8 * nc) * 8
                group_rows, within = min(8, nr - first), t % (8 * nc)
                r, c = (first + within % group_rows) * 64, (within // group_rows) * 128
                rr, cc = min(r + 64, op.m), min(c + 128, op.n)
                splits = max(1, op.flags >> 8)
                span = ((op.k + splits * 32 - 1) // (splits * 32)) * 32
                k0, k1 = split * span, min((split + 1) * span, op.k)
                assert k0 < k1
                read.append(rectangle(op.p[0], op.m, k0, k1, r, rr) if op.flags & 1
                            else rectangle(op.p[0], op.k, r, rr, k0, k1))
                read.append(rectangle(op.p[1], op.k, c, cc, k0, k1) if op.flags & 2
                            else rectangle(op.p[1], op.n, k0, k1, c, cc))
                write.append(rectangle(op.p[2], op.n, split * op.m + r, split * op.m + rr, c, cc))
                if op.flags & 4:
                    read.extend(write)
                if op.p[3]: read.append(rectangle(op.p[3], op.n, 0, 1, c, cc))
            elif op.code == Code.SUM_ROWS:
                c, cc = tile * 32, min((tile + 1) * 32, op.n)
                read.append(rectangle(op.p[0], op.n, 0, op.m, c, cc))
                write.append(rectangle(op.p[3], op.n, 0, 1, c, cc))
            else:
                rect_columns = columns
                nc = (columns + 127) // 128
                r, c = tile // nc * 64, tile % nc * 128
                rr, cc = min(r + 64, rows), min(c + 128, columns)
                write.append(rectangle(dest, columns, r, rr, c, cc))
                if op.code == Code.SUM_SPLITS:
                    for part in range(op.n):
                        read.append(rectangle(op.p[0], columns, part * rows + r, part * rows + rr, c, cc))
                    if op.flags & 4: read.extend(write)
                    if op.p[2]: read.append(rectangle(op.p[2], columns, 0, 1, c, cc))
                else:
                    read.append(rectangle(op.p[0], columns, r, rr, c, cc))
                    if op.code == Code.GELU_BACKWARD:
                        read.append(rectangle(op.p[1], columns, r, rr, c, cc))
            deps = set()
            for area in read + write:
                deps.update(node for prev, node in writers[area[0]] if overlaps(area, prev))
            for area in write:
                deps.update(node for prev, node in readers[area[0]] if overlaps(area, prev))
            index = len(nodes)
            nodes.append([oi, tile, rect_columns])
            parents.append(sorted(deps))
            footprints.append((read, write))
            current.append(index)
        # Tiles within one instruction have disjoint outputs. Only earlier
        # instructions participate in RAW/WAR/WAW dependency construction.
        for index in current:
            read, write = footprints[index]
            for area in read: readers[area[0]].append((area, index))
            for area in write: writers[area[0]].append((area, index))
    pointers = sorted(set(readers) | set(writers))
    for a, b in zip(pointers, pointers[1:]):
        size = 4
        for d in shapes[a]: size *= d
        if a + size > b:
            raise ValueError('Tile regions require distinct buffers or identical base pointers')
    # Consumers with exactly the same prerequisite tiles share one counter.
    # This preserves tile precision while avoiding identical atomic fan-in for
    # every output column of a GEMM or every feature of a bias reduction.
    groups = defaultdict(list)
    for child, deps in enumerate(parents):
        assert all(parent < child for parent in deps), "Task IDs must be topological"
        groups[tuple(deps)].append(child)
    representatives = [0] * len(nodes)
    children = [set() for _ in nodes]
    for deps, members in groups.items():
        representative = members[0]
        for child in members: representatives[child] = representative
        for parent in deps: children[parent].add(representative)
    edges, roots = [], []
    for i, node in enumerate(nodes):
        node.extend((len(edges), len(children[i]), len(parents[i]), representatives[i], 0, 0))
        edges.extend(sorted(children[i]))
        if not parents[i]: roots.append(i)
    for members in groups.values():
        nodes[members[0]][7:] = [len(edges), len(members)]
        edges.extend(members)
    assert roots and len(nodes) < 2**31

    def ints(values):
        value = torch.tensor(values, dtype=torch.int32, device='cuda')
        program.allocations.append(value)
        return value
    descriptors = torch.frombuffer(bytearray(bytes((Operation * len(ops))(*ops))), dtype=torch.uint8).cuda()
    tasks, edge_data, root_data = ints(nodes), ints(edges), ints(roots)
    queue, remaining, control = ints([-1] * len(nodes)), ints([0] * len(nodes)), ints([0] * 4)
    tracing = os.getenv('TILE_TRACE', '0') == '1'
    trace = torch.zeros((len(nodes), 3), dtype=torch.int64, device='cuda') if tracing else None
    buffers = [descriptors, tasks, edge_data, queue, remaining, control, root_data, trace]
    program.allocations.extend(t for t in buffers if t is not None)
    engine = os.getenv('TILE_ENGINE', 'ready')
    if engine not in ('ready', 'static'):
        raise ValueError('TILE_ENGINE must be ready or static')
    wrapper = Operation(int(Code.TILE_GRAPH), 1, len(nodes), len(roots), len(edges), int(engine == 'static'))
    for i, value in enumerate(buffers): wrapper.p[i] = value.data_ptr() if value is not None else None
    program.tile_graphs.append(dict(begin=begin, end=end, nodes=nodes, parents=parents,
                                   labels=program.labels[begin:end], trace=trace, dependency_groups=len(groups)))
    return wrapper
