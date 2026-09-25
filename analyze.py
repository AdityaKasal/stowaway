"""Analyze router logs from expert-logger.

Answers, for the generated (decode) tokens:
  1. popularity  - are a few experts used far more than others?
  2. reuse       - does a token reuse the experts the previous tokens used?
  3. caching     - with room for X% of experts in RAM, how often is the needed expert already there?
  4. prediction  - can layer L's choice predict layer L+1's choice (so we can prefetch it)?
  5. cost        - what that means in SSD reads per token

usage: .venv/bin/python analyze.py data models/Qwen3.5-35B-A3B-Q5_K_M.gguf
"""

import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "llama.cpp" / "gguf-py"))
import gguf  # noqa: E402

CACHE_FRACTIONS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]


def model_info(gguf_path):
    first = Path(gguf_path)
    r = gguf.GGUFReader(first)
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()

    def field(name):
        return int(r.fields[f"{arch}.{name}"].parts[-1][0])

    n_expert = field("expert_count")
    # split models ("-00001-of-00003.gguf"): tensors are spread over all parts, metadata is in the first
    parts = sorted(first.parent.glob(first.name.replace("00001-of", "*-of"))) if "00001-of" in first.name else [first]
    expert_bytes = defaultdict(int)  # layer -> bytes of all routed experts in that layer
    total = 0
    for part in parts:
        for t in gguf.GGUFReader(part).tensors:
            total += int(t.n_bytes)
            if "_exps." in t.name:
                layer = int(t.name.split(".")[1])
                expert_bytes[layer] += int(t.n_bytes)
    per_expert = np.mean(list(expert_bytes.values())) / n_expert  # up+gate+down for one expert, one layer
    return {
        "arch": arch,
        "n_expert": n_expert,
        "k": field("expert_used_count"),
        "n_moe_layers": len(expert_bytes),
        "file_gb": total / 1e9,
        "experts_gb": sum(expert_bytes.values()) / 1e9,
        "expert_mb": per_expert / 1e6,
    }


def load(data_dir):
    ex = pd.read_csv(Path(data_dir) / "experts.csv")
    pr = pd.read_csv(Path(data_dir) / "prompts.csv")
    ex = ex[ex.gen == 1].sort_values(["prompt", "pos", "layer"])
    ecols = [c for c in ex.columns if c.startswith("e")]
    layers = sorted(ex.layer.unique())
    # sel[layer] = (T, k) array of experts for every generated token, in generation order
    sel = {L: ex[ex.layer == L][ecols].to_numpy() for L in layers}
    meta = ex[ex.layer == layers[0]][["prompt", "pos"]].to_numpy()
    cat = pr.set_index("prompt").category.to_dict()
    return sel, meta, cat, layers


# ---------- 1. popularity ----------

def popularity(sel, n_expert):
    top = {0.1: [], 0.25: [], 0.5: []}
    unused = []
    for s in sel.values():
        counts = np.sort(np.bincount(s.ravel(), minlength=n_expert))[::-1]
        for f in top:
            top[f].append(counts[: int(f * n_expert)].sum() / counts.sum())
        unused.append((counts == 0).mean())
    return {f"top_{int(f*100)}pct_share": float(np.mean(v)) for f, v in top.items()} | {
        "never_used_frac": float(np.mean(unused))
    }


# ---------- 2. reuse over recent tokens ----------

def reuse(sel, meta, n_expert, k, windows=(1, 4, 16, 64)):
    out = {}
    prompts = meta[:, 0]
    for W in windows:
        vals = []
        for s in sel.values():
            for t in range(W, len(s)):
                if prompts[t - W] != prompts[t]:
                    continue
                recent = set(s[t - W : t].ravel())
                vals.append(np.isin(s[t], list(recent)).mean())
        out[f"last_{W}"] = float(np.mean(vals))
        out[f"last_{W}_random"] = 1 - (1 - k / n_expert) ** W
    return out


# ---------- 3. cache simulation ----------

def simulate(s, cap, policy):
    """s: (T, k) requests for one layer. Returns hit rate for a cache holding `cap` experts."""
    T, k = s.shape
    hits = 0
    if policy == "lru":
        cache = OrderedDict()
        for row in s:
            for e in row:
                if e in cache:
                    hits += 1
                    cache.move_to_end(e)
                else:
                    cache[e] = True
                    if len(cache) > cap:
                        cache.popitem(last=False)
    elif policy == "lfu":
        freq = defaultdict(int)
        cache = set()
        for row in s:
            for e in row:
                freq[e] += 1
                if e in cache:
                    hits += 1
                else:
                    if len(cache) >= cap:
                        protect = set(row)
                        victim = min((c for c in cache if c not in protect), key=lambda c: freq[c])
                        cache.remove(victim)
                    cache.add(e)
    elif policy == "belady":  # optimal: evict the expert needed furthest in the future
        flat = s.ravel()
        nxt = np.full(len(flat), np.iinfo(np.int64).max)
        last = {}
        for i in range(len(flat) - 1, -1, -1):
            nxt[i] = last.get(flat[i], np.iinfo(np.int64).max)
            last[flat[i]] = i
        cache = {}  # expert -> next use index
        for i, e in enumerate(flat):
            if e in cache:
                hits += 1
            elif len(cache) >= cap:
                victim = max(cache, key=cache.get)
                del cache[victim]
            cache[e] = nxt[i]
    return hits / flat_len(s)


def flat_len(s):
    return s.shape[0] * s.shape[1]


def caching(sel, n_expert):
    res = {}
    for policy in ("lru", "lfu", "belady"):
        res[policy] = {}
        for f in CACHE_FRACTIONS:
            cap = int(f * n_expert)
            res[policy][f] = float(np.mean([simulate(s, cap, policy) for s in sel.values()]))
    return res


# ---------- 4. predicting the next layer ----------

def cross_layer(sel, meta, layers, n_expert, k):
    """Train co-occurrence (layer L choice -> layer L+1 choice) on even prompts, test on odd ones."""
    train = meta[:, 0] % 2 == 0
    test = ~train
    recall = {m: [] for m in (k, 2 * k, 4 * k)}
    recall_prev_tok = []
    for a, b in zip(layers, layers[1:]):
        A, B = sel[a], sel[b]
        M = np.zeros((n_expert, n_expert))
        for ra, rb in zip(A[train], B[train]):
            M[np.ix_(ra, rb)] += 1
        idx = np.where(test)[0]
        for t in idx:
            score = M[A[t]].sum(0)
            order = np.argsort(-score)
            actual = set(B[t])
            for m in recall:
                recall[m].append(len(actual & set(order[:m])) / k)
            # baseline: guess layer L+1 will pick what it picked for the previous token
            if t > 0 and meta[t - 1, 0] == meta[t, 0]:
                recall_prev_tok.append(len(actual & set(B[t - 1])) / k)
    out = {f"cooccur_top{m}": float(np.mean(v)) for m, v in recall.items()}
    out["prev_token_same_layer"] = float(np.mean(recall_prev_tok))
    out["random_top2k"] = 2 * k / n_expert
    return out


# ---------- 5. per-topic differences ----------

def topic_divergence(sel, meta, cat, n_expert):
    cats = sorted(set(cat.values()))
    prompt_cat = np.array([cat[p] for p in meta[:, 0]])

    def js(p, q):
        m = (p + q) / 2
        kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(a / b), 0))  # noqa: E731
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    out = {}
    for c in cats:
        vals = []
        for s in sel.values():
            p = np.bincount(s[prompt_cat == c].ravel(), minlength=n_expert) + 1e-9
            q = np.bincount(s[prompt_cat != c].ravel(), minlength=n_expert) + 1e-9
            vals.append(js(p / p.sum(), q / q.sum()))
        out[c] = float(np.mean(vals))
    return out  # 0 = same experts as everything else, 1 = completely different experts


def main():
    data_dir, gguf_path = sys.argv[1], sys.argv[2]
    info = model_info(gguf_path)
    sel, meta, cat, layers = load(data_dir)
    E, k = info["n_expert"], info["k"]

    print(f"\nmodel: {info['arch']}  {E} experts, {k} per token, {info['n_moe_layers']} MoE layers")
    print(f"file {info['file_gb']:.1f} GB, of which routed experts {info['experts_gb']:.1f} GB "
          f"({info['expert_mb']:.2f} MB per expert per layer)")
    print(f"logged {len(meta)} generated tokens from {len(set(meta[:, 0]))} prompts")
    active_mb = k * info["n_moe_layers"] * info["expert_mb"]
    print(f"routed expert data needed per token: {active_mb:.0f} MB")

    results = {"model": info, "tokens": int(len(meta))}

    print("\n1. POPULARITY (share of all expert picks that go to the most popular experts)")
    results["popularity"] = pop = popularity(sel, E)
    for key, v in pop.items():
        print(f"   {key:22s} {v:6.1%}")

    print("\n2. REUSE (share of this token's experts also used by the last W tokens, same layer)")
    results["reuse"] = ru = reuse(sel, meta, E, k)
    for W in (1, 4, 16, 64):
        print(f"   last {W:3d} tokens: {ru[f'last_{W}']:6.1%}   (random routing would give {ru[f'last_{W}_random']:6.1%})")

    print("\n3. CACHE HIT RATE (cache holds X% of each layer's experts)")
    results["cache"] = ca = caching(sel, E)
    print("   " + "cache".ljust(8) + "".join(f"{p:>9s}" for p in ca))
    for f in CACHE_FRACTIONS:
        print(f"   {f:>5.0%}   " + "".join(f"{ca[p][f]:9.1%}" for p in ca))

    print("\n4. PREDICTING THE NEXT LAYER (share of layer L+1's experts we'd have prefetched)")
    results["prediction"] = cl = cross_layer(sel, meta, layers, E, k)
    for key, v in cl.items():
        print(f"   {key:24s} {v:6.1%}")

    print("\n5. TOPIC DIFFERENCES (Jensen-Shannon divergence of expert usage vs other topics, 0..1)")
    results["topics"] = td = topic_divergence(sel, meta, cat, E)
    for c, v in td.items():
        print(f"   {c:14s} {v:.3f}")

    print("\n6. WHAT IT MEANS: SSD reads per token and the speed limit they set")
    results["cost"] = {}
    for f in CACHE_FRACTIONS:
        row = {}
        for p in ("lru", "belady"):
            miss_mb = (1 - ca[p][f]) * active_mb
            row[p] = {"miss_mb": miss_mb, "tok_s_at_3GBps": 3000 / miss_mb if miss_mb else None}
        results["cost"][f] = row
        print(f"   cache {f:>4.0%}: LRU misses {row['lru']['miss_mb']:6.0f} MB/token (SSD limit ~{row['lru']['tok_s_at_3GBps']:5.1f} tok/s at 3 GB/s)"
              f" | optimal {row['belady']['miss_mb']:5.0f} MB/token")

    out = Path(data_dir) / "summary.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
