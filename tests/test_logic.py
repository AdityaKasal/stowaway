"""Fast checks of stowaway's bookkeeping logic (no model downloads). Run: python -m pytest tests -q"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))

import moe  # noqa: E402
import repack_experts  # noqa: E402
import streampack  # noqa: E402


def _layout(n_layers=3, n_expert=8, seed=0):
    rng = random.Random(seed)
    layers, pos = {}, 4096
    for L in range(n_layers):
        entries = []
        for kind in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            per = rng.randrange(1, 50) * 32
            entries.append((repack_experts.ORDER[kind], f"blk.{L}.{kind}.weight", 0, pos, per * n_expert))
            pos += per * n_expert + rng.randrange(0, 5000)  # gaps: non-expert tensors in between
        layers[L] = entries
    return layers, pos + 1000


def test_router_matches_the_packed_layout():
    """Every stream byte lands exactly once: expert bytes where plan_layout says, the rest at their own offset."""
    n_expert = 8
    layers, size = _layout(n_expert=n_expert)
    plan, _, total = repack_experts.plan_layout(layers, n_expert)
    stream = bytes(random.Random(1).randrange(256) for _ in range(size))
    packed, gguf = bytearray(total), bytearray(size)
    segs = sorted((foff, nb, lbase, block, off, per) for L, lbase, block, offs in plan for _, _, foff, nb, off, per in offs)
    router = streampack.Router(segs)
    pos = 0
    while pos < size:  # odd-sized chunks, like a network stream
        step = random.Random(pos).randrange(1, 777)
        for is_exp, dst, a, b in router.pieces(pos, min(size, pos + step)):
            (packed if is_exp else gguf)[dst:dst + (b - a)] = stream[a:b]
        pos += step
    for L, lbase, block, offs in plan:  # expert e of each tensor sits at base + e * block + off
        for _, _, foff, nb, off, per in offs:
            for e in range(n_expert):
                assert packed[lbase + e * block + off:][:per] == stream[foff + e * per:][:per]
    expert_ranges = [(s[0], s[0] + s[1]) for s in segs]
    for i in range(size):
        if not any(a <= i < b for a, b in expert_ranges):
            assert gguf[i] == stream[i]
        else:
            assert gguf[i] == 0  # never written: a hole in the real file


def test_plan_layout_is_aligned():
    layers, _ = _layout(n_expert=4)
    plan, lines, total = repack_experts.plan_layout(layers, 4)
    for L, base, block, offs in plan:
        assert base % 4096 == 0 and block % 4096 == 0
        assert all(off % 4096 == 0 for *_, off, per in offs)
    assert len(lines) == 9 and total == sum(block * 4 for _, _, block, _ in plan)


def test_expected_speed_is_monotonic_in_memory_and_drive():
    for name in moe.MEASURED:
        speeds = [moe.expected_speed(name, r, 3.0) for r in (2.0, 3.2, 5.2, 7.3, 15.5)]
        assert speeds == sorted(speeds), name
        drives = [moe.expected_speed(name, 7.3, d) for d in (0.3, 0.55, 1.5, 3.0, 6.0)]
        assert drives == sorted(drives), name


def test_recommendation_by_machine():
    def pick(ram, drive):
        sp = {n: moe.expected_speed(n, ram, drive) for n in moe.QUALITY if n != "qwen3.5-35b"}
        good = [n for n in moe.QUALITY if sp.get(n) and sp[n] >= moe.COMFORT]
        return good[0] if good else max(sp, key=sp.get)
    assert pick(3.2, 3.0) == "gpt-oss-20b"
    assert pick(7.3, 3.0) == "qwen3.6-35b"
    assert pick(15.5, 3.0) == "gpt-oss-120b"


def test_small_machine_plan_streams_and_keeps_a_margin():
    info = {"dense_gb": 2.12, "dense_managed_gb": 2.0, "layers": 40, "expert_gb": 23.8, "active_expert_gb": 0.74}
    plan, err = moe.make_plan(info, 3.2, 3.0)
    assert not err and plan["small"] and plan["dense_stream_gb"] > 0 and plan["ctx"] == 2048
    plan, err = moe.make_plan(info, 1.5, 3.0)
    assert err and "not enough free RAM" in err
    plan, err = moe.make_plan(info, 7.3, 3.0)
    assert not err and not plan["small"] and plan["dense_stream_gb"] == 0


def test_parse_header_roundtrip():
    """A synthetic GGUF header: tensor sizes and the aligned data start come out right."""
    import struct
    from gguf.constants import GGMLQuantizationType
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    body = b"GGUF" + struct.pack("<IQQ", 3, 1, 2)  # version, tensor count, key-value count
    body += s("general.architecture") + struct.pack("<I", 8) + s("llama")
    body += s("general.alignment") + struct.pack("<II", 4, 64)
    body += s("blk.0.ffn_up_exps.weight") + struct.pack("<I", 3) + struct.pack("<QQQ", 64, 32, 4)
    body += struct.pack("<IQ", GGMLQuantizationType.F32, 0)
    kv, tensors, data_start = streampack.parse_header(body + bytes(200))
    assert kv["general.architecture"] == "llama" and kv["general.alignment"] == 64
    assert tensors == [("blk.0.ffn_up_exps.weight", 64 * 32 * 4 * 4, 0)]
    assert data_start % 64 == 0 and data_start >= len(body)
