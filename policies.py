"""Compare cache eviction policies on a recorded router trace (experts.csv from expert-logger).

The real cache is one pool shared by all layers, so this simulates that: a key is (layer, expert), and the pool
holds a fraction of all layer*expert pairs. Requests are replayed in generation order, layer by layer.

usage: .venv/bin/python policies.py results/pc/122b-router 256
"""

import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

FRACTIONS = [0.15, 0.19, 0.22, 0.30]


def trace(data_dir):
    ex = pd.read_csv(Path(data_dir) / "experts.csv")
    ex = ex[ex.gen == 1].sort_values(["prompt", "pos", "layer"])
    ecols = [c for c in ex.columns if c.startswith("e")]
    layers = ex.layer.to_numpy()
    sel = ex[ecols].to_numpy()
    # one request list per op: (layer, expert) keys for this token and layer
    return [[(int(L), int(e)) for e in row] for L, row in zip(layers, sel)]


def lru(reqs, cap):
    c, hits = OrderedDict(), 0
    for op in reqs:
        for k in op:
            if k in c:
                hits += 1
                c.move_to_end(k)
            else:
                c[k] = 1
                if len(c) > cap:
                    c.popitem(last=False)
    return hits


def slru(reqs, cap, protected_frac=0.8):
    """Segmented LRU: new keys go to a probation segment; a second hit promotes them to the protected segment."""
    pcap = int(cap * protected_frac)
    prob, prot, hits = OrderedDict(), OrderedDict(), 0
    for op in reqs:
        for k in op:
            if k in prot:
                hits += 1
                prot.move_to_end(k)
            elif k in prob:
                hits += 1
                del prob[k]
                prot[k] = 1
                if len(prot) > pcap:
                    old, _ = prot.popitem(last=False)
                    prob[old] = 1
            else:
                prob[k] = 1
            while len(prob) + len(prot) > cap:
                (prob if prob else prot).popitem(last=False)
    return hits


def two_q(reqs, cap):
    """2Q: first-time keys go through a small FIFO (A1in); keys seen again recently (A1out ghost list) go to main LRU."""
    kin, kout = max(1, cap // 4), cap // 2
    a1in, a1out, am, hits = OrderedDict(), OrderedDict(), OrderedDict(), 0
    for op in reqs:
        for k in op:
            if k in am:
                hits += 1
                am.move_to_end(k)
                continue
            if k in a1in:
                hits += 1
                continue
            if k in a1out:
                del a1out[k]
                am[k] = 1
            else:
                a1in[k] = 1
            while len(a1in) + len(am) > cap:
                if len(a1in) > kin or not am:
                    old, _ = a1in.popitem(last=False)
                    a1out[old] = 1
                    if len(a1out) > kout:
                        a1out.popitem(last=False)
                else:
                    am.popitem(last=False)
    return hits


def arc(reqs, cap):
    """Adaptive Replacement Cache (Megiddo & Modha): balances recency and frequency automatically."""
    t1, t2, b1, b2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
    p, hits = 0, 0

    def replace(k):
        nonlocal p
        if t1 and (len(t1) > p or (k in b2 and len(t1) == p)):
            old, _ = t1.popitem(last=False)
            b1[old] = 1
        else:
            old, _ = t2.popitem(last=False)
            b2[old] = 1

    for op in reqs:
        for k in op:
            if k in t1:
                hits += 1
                del t1[k]
                t2[k] = 1
            elif k in t2:
                hits += 1
                t2.move_to_end(k)
            elif k in b1:
                p = min(cap, p + max(len(b2) // max(len(b1), 1), 1))
                replace(k)
                del b1[k]
                t2[k] = 1
            elif k in b2:
                p = max(0, p - max(len(b1) // max(len(b2), 1), 1))
                replace(k)
                del b2[k]
                t2[k] = 1
            else:
                if len(t1) + len(b1) == cap:
                    if len(t1) < cap:
                        b1.popitem(last=False)
                        replace(k)
                    else:
                        t1.popitem(last=False)
                elif len(t1) + len(b1) < cap and len(t1) + len(t2) + len(b1) + len(b2) >= cap:
                    if len(t1) + len(t2) + len(b1) + len(b2) == 2 * cap:
                        b2.popitem(last=False)
                    replace(k)
                t1[k] = 1
    return hits


def lfu_decay(reqs, cap, half_life_ops=2000):
    """Evict the key with the lowest exponentially decayed use count (a mix of frequency and recency).
    score * decay^(now - last) orders keys the same at every `now`, so log(score) - last*log(decay) works as a heap key."""
    import heapq, math
    ld = math.log(0.5) / half_life_ops  # log(decay) < 0
    score, last, cache, heap, hits, t = {}, {}, set(), [], 0, 0
    for op in reqs:
        t += 1
        protect = set(op)
        for k in op:
            if k in cache:
                hits += 1
            else:
                if len(cache) >= cap:
                    skipped = []
                    while True:
                        key, victim = heapq.heappop(heap)
                        if victim not in cache or key != math.log(score[victim]) - last[victim] * ld:
                            continue  # stale
                        if victim in protect:
                            skipped.append((key, victim))
                            continue
                        cache.remove(victim)
                        break
                    for item in skipped:
                        heapq.heappush(heap, item)
                cache.add(k)
            prev = score[k] * math.exp(ld * (t - last[k])) if k in score else 0.0
            score[k], last[k] = prev + 1, t
            heapq.heappush(heap, (math.log(score[k]) - t * ld, k))
    return hits


def belady(reqs, cap):
    flat = [k for op in reqs for k in op]
    nxt, seen = [0] * len(flat), {}
    for i in range(len(flat) - 1, -1, -1):
        nxt[i] = seen.get(flat[i], 1 << 60)
        seen[flat[i]] = i
    import heapq
    cache, heap, hits = {}, [], 0
    for i, k in enumerate(flat):
        if k in cache:
            hits += 1
        elif len(cache) >= cap:
            while True:
                negnext, victim = heapq.heappop(heap)
                if cache.get(victim) == -negnext:
                    del cache[victim]
                    break
        cache[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return hits


def main():
    data_dir, n_expert = sys.argv[1], int(sys.argv[2])
    reqs = trace(data_dir)
    n_layers = len({k[0] for op in reqs for k in op})
    total = sum(len(op) for op in reqs)
    print(f"{len(reqs)} ops ({n_layers} layers), {total} expert requests; pool = fraction of {n_layers * n_expert} layer-experts\n")
    policies = {"LRU (current)": lru, "SLRU": slru, "2Q": two_q, "ARC": arc, "LFU-decay": lfu_decay, "optimal": belady}
    print(f"{'cache size':>10}" + "".join(f"{name:>15}" for name in policies))
    for f in FRACTIONS:
        cap = int(f * n_layers * n_expert)
        row = [policies[name](reqs, cap) / total for name in policies]
        print(f"{f:>10.0%}" + "".join(f"{h:>15.1%}" for h in row), flush=True)


if __name__ == "__main__":
    main()
