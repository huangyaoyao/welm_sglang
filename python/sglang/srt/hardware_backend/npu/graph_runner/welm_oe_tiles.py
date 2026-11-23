"""Access contracts for WeLM over-encoding in fixed decode and draft buckets."""


def _hash_rows(inputs, output, batch, tasks, block, shared=()):
    if (
        batch < 1 or block < 1 or tasks != (batch + block - 1) // block
        or any(t.numel() != batch for t in inputs)
        or output.shape != (4, batch)
    ):
        return None
    # The four stores in the source kernel are separated by batch elements.
    # Treating the output as [batch, 4] would give the wrong inter-program edges.
    return dict(count=batch, block=block,
                reads=[(t, t.element_size()) for t in inputs], writes=[],
                plane_writes=[(output, 4, output.element_size())],
                shared_reads=list(shared))


def _decode_hash(ids, table, requests, positions, output, batch, context, tasks,
                 *, BLOCK_B, **constants):
    if len(table.shape) != 2 or table.shape[1] != context:
        return None
    return _hash_rows((ids, requests, positions), output, batch, tasks, BLOCK_B, (table,))


def _draft_hash(ids, previous1, previous2, output, batch, tasks, *, BLOCK_B, **constants):
    return _hash_rows((ids, previous1, previous2), output, batch, tasks, BLOCK_B)


def _verify_hash(ids, table, requests, offsets, lengths, starts, output, tokens,
                 context, tiles_per_request, tasks, *, BLOCK_T, **constants):
    # With one task per request, metadata is row-local. Device offsets and
    # lengths select both the history and destination at replay time.
    metadata = (requests, offsets, lengths, starts)
    if (tiles_per_request != 1 or tasks <= 0 or output.shape != (4, tokens)
            or ids.numel() != tokens or len(table.shape) != 2 or table.shape[1] != context
            or any(t.numel() != tasks for t in metadata)):
        return None
    return dict(count=tasks, block=1, reads=[(t, t.element_size()) for t in metadata],
                writes=[], shared_reads=[ids, table], whole_writes=[output])


def install_oe_tiles(native):
    from sglang.srt.layers import welmv4_npu_op as kernels

    # The last original JIT is shared with prefill. The runner activates
    # this binding solely while preparing fixed decode/verification buckets.
    for name, layout in (
        ("_welmv4_oe_hash_decode_4way_kernel", _decode_hash),
        ("_welmv4_oe_hash_explicit_history_4way_kernel", _draft_hash),
        ("_welmv4_oe_hash_prefill_4way_kernel", _verify_hash),
    ):
        setattr(kernels, name, native(tile_layout=layout)(getattr(kernels, name)))
