from __future__ import annotations

import argparse
import ast
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_sequence(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    return [str(x) for x in parsed]


def great_circle_matrix(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    lon1, lon2 = np.radians(longitude)[:, None], np.radians(longitude)[None, :]
    lat1, lat2 = np.radians(latitude)[:, None], np.radians(latitude)[None, :]
    value = np.sin((lat2 - lat1) / 2) ** 2
    value += np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def row_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=True)
    sums = x.sum(axis=1, keepdims=True)
    return np.divide(x, sums, out=np.zeros_like(x), where=sums > 0)


def topk_rows(scores: np.ndarray, k: int = 10, exclude_self: bool = True) -> np.ndarray:
    work = scores.copy()
    if exclude_self:
        np.fill_diagonal(work, -np.inf)
    idx = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(work, idx, axis=1)
    order = np.argsort(-vals, axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


def build_events(rows):
    events = []
    for route_idx, row in enumerate(rows):
        seq = row["ids"]
        for t in range(len(seq) - 1):
            if seq[t] == seq[t + 1]:
                continue
            events.append((route_idx, seq[t - 1] if t > 0 else -1, seq[t], seq[t + 1], t))
    return events


def collect_train(rows, n_items: int):
    counts = np.zeros((n_items, n_items), dtype=np.float64)
    context_counts = defaultdict(Counter)
    for row in rows:
        seq = row["ids"]
        for t in range(len(seq) - 1):
            o, d = seq[t], seq[t + 1]
            if o == d:
                continue
            counts[o, d] += 1
            if t > 0:
                context_counts[(seq[t - 1], o)][d] += 1
    return counts, context_counts


def bootstrap_route_ci(route_values: np.ndarray, seed: int = 2026, reps: int = 1000):
    rng = np.random.default_rng(seed)
    n = len(route_values)
    if n == 0:
        return [math.nan, math.nan]
    means = np.empty(reps)
    for b in range(reps):
        means[b] = route_values[rng.integers(0, n, n)].mean()
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def summarize_predictions(events, recs: np.ndarray, n_items: int, distances: np.ndarray,
                          long_tail: np.ndarray, route_count: int, route_mask=None):
    if route_mask is not None:
        keep = np.asarray([bool(route_mask[e[0]]) for e in events])
        events = [e for e, flag in zip(events, keep) if flag]
        recs = recs[keep]
    origins = np.asarray([e[2] for e in events], dtype=int)
    targets = np.asarray([e[3] for e in events], dtype=int)
    route_ids = np.asarray([e[0] for e in events], dtype=int)
    matches = recs == targets[:, None]
    any_match = matches.any(axis=1)
    ranks = np.where(any_match, matches.argmax(axis=1) + 1, np.inf)
    hit5 = ranks <= 5
    hit10 = ranks <= 10
    rr = np.where(np.isfinite(ranks), 1.0 / ranks, 0.0)
    ndcg = np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0)
    tail_mask = long_tail[targets]
    valid_routes = np.unique(route_ids)
    route_hit = np.asarray([hit10[route_ids == r].mean() for r in valid_routes])
    selected_distances = distances[origins[:, None], recs]
    return {
        "event_count": int(len(events)),
        "route_count": int(len(valid_routes)),
        "Hit@5": float(hit5.mean()),
        "Hit@10": float(hit10.mean()),
        "MRR": float(rr.mean()),
        "NDCG@10": float(ndcg.mean()),
        "catalog_coverage@10": float(len(np.unique(recs)) / n_items),
        "long_tail_test_count": int(tail_mask.sum()),
        "long_tail_recall@10": float(hit10[tail_mask].mean()) if tail_mask.any() else math.nan,
        "mean_distance_km": float(np.mean(selected_distances)),
        "median_distance_km": float(np.median(selected_distances)),
        "macro_route_Hit@10": float(route_hit.mean()),
        "macro_route_Hit@10_95CI": bootstrap_route_ci(route_hit),
    }


def evaluate_static(name, events, candidate_matrix, *summary_args, route_mask=None):
    origins = np.asarray([e[2] for e in events], dtype=int)
    recs = candidate_matrix[origins]
    result = summarize_predictions(events, recs, *summary_args, route_mask=route_mask)
    result["model"] = name
    return result


def second_order_candidates(context_counts, first_candidates, n_items: int, min_support: int = 5):
    cache = {}
    for context, counter in context_counts.items():
        if sum(counter.values()) < min_support:
            continue
        current = context[1]
        ordered = [item for item, _ in counter.most_common() if item != current]
        for item in first_candidates[current]:
            if int(item) != current and int(item) not in ordered:
                ordered.append(int(item))
            if len(ordered) >= 10:
                break
        cache[context] = np.asarray(ordered[:10], dtype=int)
    return cache


def evaluate_second_order(name, events, cache, first_candidates, *summary_args, route_mask=None):
    recs = np.vstack([cache.get((e[1], e[2]), first_candidates[e[2]]) for e in events])
    result = summarize_predictions(events, recs, *summary_args, route_mask=route_mask)
    result["model"] = name
    return result


def make_prefixes(rows, max_len: int):
    prefixes, meta = [], []
    for route_idx, row in enumerate(rows):
        seq = row["ids"]
        for t in range(len(seq) - 1):
            if seq[t] == seq[t + 1]:
                continue
            prefixes.append(seq[max(0, t - max_len + 1):t + 1])
            meta.append((route_idx, seq[t - 1] if t > 0 else -1, seq[t], seq[t + 1], t))
    return prefixes, meta


def train_neural(model_name: str, train_rows, val_rows, test_rows, n_items: int, long_tail,
                 distances, unseen_template_mask, output_dir: Path, seed: int, epochs: int,
                 max_len: int = 40, dim: int = 64):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    class Segments(Dataset):
        def __init__(self, rows):
            self.segments = []
            for row in rows:
                seq = [x + 1 for x in row["ids"]]
                for start in range(0, len(seq) - 1, max_len):
                    seg = seq[start:start + max_len + 1]
                    if len(seg) >= 2:
                        self.segments.append(seg)
        def __len__(self): return len(self.segments)
        def __getitem__(self, idx): return self.segments[idx]

    def collate(batch):
        x = torch.zeros((len(batch), max_len), dtype=torch.long)
        y = torch.full((len(batch), max_len), -100, dtype=torch.long)
        lengths = torch.zeros(len(batch), dtype=torch.long)
        for i, seg in enumerate(batch):
            m = min(len(seg) - 1, max_len)
            x[i, :m] = torch.tensor(seg[:m])
            y[i, :m] = torch.tensor(seg[1:m + 1])
            lengths[i] = m
        return x, y, lengths

    class GRURec(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
            self.gru = nn.GRU(dim, dim, batch_first=True)
            self.out = nn.Linear(dim, n_items + 1)
        def forward(self, x):
            h, _ = self.gru(self.emb(x))
            return self.out(h)

    class SASRec(nn.Module):
        def __init__(self):
            super().__init__()
            self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
            self.pos = nn.Embedding(max_len, dim)
            layer = nn.TransformerEncoderLayer(dim, 2, dim * 2, dropout=.2, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, 2)
            self.norm = nn.LayerNorm(dim)
            self.out = nn.Linear(dim, n_items + 1)
        def forward(self, x):
            pos = torch.arange(x.size(1), device=x.device)[None, :]
            z = self.item(x) + self.pos(pos)
            causal = torch.triu(torch.ones(x.size(1), x.size(1), dtype=torch.bool, device=x.device), diagonal=1)
            z = self.encoder(z, mask=causal, src_key_padding_mask=(x == 0))
            return self.out(self.norm(z))

    model = GRURec() if model_name == "GRU4Rec" else SASRec()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    loader = DataLoader(Segments(train_rows), batch_size=128, shuffle=True, collate_fn=collate, num_workers=0)

    def predict(rows):
        prefixes, events = make_prefixes(rows, max_len)
        all_recs = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(prefixes), 512):
                batch = prefixes[start:start + 512]
                x = torch.zeros((len(batch), max_len), dtype=torch.long)
                lengths = torch.zeros(len(batch), dtype=torch.long)
                currents = []
                for i, prefix in enumerate(batch):
                    m = len(prefix)
                    x[i, :m] = torch.tensor([v + 1 for v in prefix])
                    lengths[i] = m
                    currents.append(prefix[-1])
                logits = model(x)
                rows_idx = torch.arange(len(batch))
                scores = logits[rows_idx, lengths - 1]
                scores[:, 0] = -torch.inf
                scores[rows_idx, torch.tensor(currents) + 1] = -torch.inf
                recs = torch.topk(scores, 10, dim=1).indices.cpu().numpy() - 1
                all_recs.append(recs)
        return events, np.vstack(all_recs)

    best_ndcg, best_state, history = -1, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        for x, y, _ in loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, n_items + 1), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tokens = int((y != -100).sum())
            total_loss += float(loss) * tokens
            total_tokens += tokens
        val_events, val_recs = predict(val_rows)
        val_result = summarize_predictions(val_events, val_recs, n_items, distances, long_tail, len(val_rows))
        history.append({"epoch": epoch, "loss": total_loss / total_tokens, "validation_NDCG@10": val_result["NDCG@10"], "validation_Hit@10": val_result["Hit@10"]})
        print(model_name, "seed", seed, "epoch", epoch, json.dumps(history[-1]), flush=True)
        if val_result["NDCG@10"] > best_ndcg:
            best_ndcg = val_result["NDCG@10"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_events, test_recs = predict(test_rows)
    result = summarize_predictions(test_events, test_recs, n_items, distances, long_tail, len(test_rows))
    result["model"] = model_name
    result["seed"] = seed
    result["best_validation_NDCG@10"] = best_ndcg
    result["template_unseen"] = summarize_predictions(test_events, test_recs, n_items, distances, long_tail, len(test_rows), unseen_template_mask)
    (output_dir / f"{model_name}_seed{seed}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save(best_state, output_dir / f"{model_name}_seed{seed}.pt")
    return result


def main(args):
    root = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    routes = pd.read_excel(root / "Supplementary_Data_S2_Cleaned_Routes.xlsx", sheet_name="Cleaned_Model_Routes")
    states = pd.read_excel(root / "Supplementary_Data_S3_States_and_Transitions.xlsx", sheet_name="Attraction_States")
    states = states.sort_values("state_id").reset_index(drop=True)
    name_to_id = dict(zip(states.attraction_name_standardized.astype(str), states.state_id.astype(int)))
    prepared = []
    for row in routes.itertuples(index=False):
        ids = []
        for name in parse_sequence(row.attraction_sequence_cleaned):
            if name in name_to_id:
                value = name_to_id[name]
                if not ids or ids[-1] != value:
                    ids.append(value)
        date = pd.to_datetime(row.departure_date, errors="coerce")
        if len(ids) >= 2 and pd.notna(date) and date.year >= 2010:
            prepared.append({"date": date, "record_id": int(row.record_id), "template_id": int(row.route_template_id), "ids": ids})
    prepared.sort(key=lambda x: (x["date"], x["record_id"]))
    a, b = int(len(prepared) * .70), int(len(prepared) * .85)
    train, val, test = prepared[:a], prepared[a:b], prepared[b:]
    n_items = len(states)
    train_counts, second_counts = collect_train(train, n_items)
    observed = row_normalize(train_counts)
    destination_frequency = train_counts.sum(axis=0)
    pull_row = destination_frequency / destination_frequency.sum()
    pull = np.tile(pull_row, (n_items, 1))
    np.fill_diagonal(pull, 0)
    pull = row_normalize(pull)
    distances = great_circle_matrix(states.longitude_wgs84.to_numpy(float), states.latitude_wgs84.to_numpy(float))
    bandwidth = np.median(distances[distances > 0])
    spatial = np.exp(-distances / bandwidth)
    np.fill_diagonal(spatial, 0)
    spatial = row_normalize(spatial)
    two_step = row_normalize(observed @ observed)
    positive = destination_frequency[destination_frequency > 0]
    cutoff = np.quantile(positive, .80)
    long_tail = destination_frequency <= cutoff
    val_events = build_events(val)
    test_events = build_events(test)
    train_templates = {x["template_id"] for x in train}
    unseen_template_mask = np.asarray([x["template_id"] not in train_templates for x in test])
    summary_args = (n_items, distances, long_tail, len(test))

    popularity_scores = np.tile(destination_frequency, (n_items, 1))
    nearest_scores = -distances
    first_scores = observed + 1e-12 * pull
    candidates = {
        "Global popularity": topk_rows(popularity_scores),
        "Nearest distance": topk_rows(nearest_scores),
        "First-order Markov": topk_rows(first_scores),
        "Original global fusion": topk_rows(.8 * observed + .1 * spatial + .1 * pull),
    }

    # Equal weight per unique route template removes repeated-product multiplicity.
    unique_template_rows = []
    seen_templates = set()
    for row in train:
        if row["template_id"] not in seen_templates:
            seen_templates.add(row["template_id"])
            unique_template_rows.append(row)
    template_counts, _ = collect_train(unique_template_rows, n_items)
    candidates["Template-debiased Markov"] = topk_rows(row_normalize(template_counts) + 1e-12 * pull)

    deterministic = []
    unseen_deterministic = []
    for name, matrix in candidates.items():
        deterministic.append(evaluate_static(name, test_events, matrix, *summary_args))
        unseen_deterministic.append(evaluate_static(name, test_events, matrix, *summary_args, route_mask=unseen_template_mask))

    second_cache = second_order_candidates(second_counts, candidates["First-order Markov"], n_items, min_support=5)
    deterministic.append(evaluate_second_order("Second-order Markov (support>=5)", test_events, second_cache, candidates["First-order Markov"], *summary_args))
    unseen_deterministic.append(evaluate_second_order("Second-order Markov (support>=5)", test_events, second_cache, candidates["First-order Markov"], *summary_args, route_mask=unseen_template_mask))

    # Support-adaptive empirical-Bayes transition smoothing.
    best = None
    val_summary_args = (n_items, distances, long_tail, len(val))
    row_totals = train_counts.sum(axis=1, keepdims=True)
    for mu in [.1, .3, 1, 3, 10, 30, 100, 300]:
        for spatial_share in [0, .25, .5, .75, 1.0]:
            prior = spatial_share * spatial + (1 - spatial_share) * pull
            score = (train_counts + mu * prior) / (row_totals + mu)
            cand = topk_rows(score)
            val_result = evaluate_static("tmp", val_events, cand, *val_summary_args)
            key = (val_result["NDCG@10"], val_result["Hit@10"], val_result["long_tail_recall@10"])
            if best is None or key > best[0]:
                best = (key, mu, spatial_share, cand, val_result)
    _, mu, spatial_share, adaptive_candidates, adaptive_val = best
    adaptive = evaluate_static("Support-adaptive empirical Bayes", test_events, adaptive_candidates, *summary_args)
    adaptive["selected_mu"] = mu
    adaptive["selected_spatial_share"] = spatial_share
    adaptive["validation"] = adaptive_val
    deterministic.append(adaptive)
    adaptive_unseen = evaluate_static("Support-adaptive empirical Bayes", test_events, adaptive_candidates, *summary_args, route_mask=unseen_template_mask)
    adaptive_unseen["selected_mu"] = mu
    adaptive_unseen["selected_spatial_share"] = spatial_share
    unseen_deterministic.append(adaptive_unseen)

    # Low-rank transition-factorization baseline selected on validation NDCG.
    if not args.skip_svd:
        start = time.time()
        u, s, vt = np.linalg.svd(np.log1p(train_counts), full_matrices=False)
        print("SVD seconds", time.time() - start, flush=True)
        best_lr = None
        for rank in [16, 32, 64, 128]:
            score = (u[:, :rank] * s[:rank]) @ vt[:rank]
            cand = topk_rows(score)
            val_result = evaluate_static("tmp", val_events, cand, *val_summary_args)
            key = val_result["NDCG@10"]
            if best_lr is None or key > best_lr[0]:
                best_lr = (key, rank, cand, val_result)
        _, rank, lr_candidates, lr_val = best_lr
        lr = evaluate_static("Low-rank transition factorization", test_events, lr_candidates, *summary_args)
        lr["selected_rank"] = rank
        lr["validation"] = lr_val
        deterministic.append(lr)
        lr_unseen = evaluate_static("Low-rank transition factorization", test_events, lr_candidates, *summary_args, route_mask=unseen_template_mask)
        lr_unseen["selected_rank"] = rank
        unseen_deterministic.append(lr_unseen)

    output = {
        "protocol": {
            "training_routes": len(train), "validation_routes": len(val), "test_routes": len(test),
            "test_routes_with_unseen_template": int(unseen_template_mask.sum()),
            "test_unseen_template_fraction": float(unseen_template_mask.mean()),
            "candidate_cutoff": 10,
            "self_candidate_excluded": True,
            "long_tail_frequency_quantile": .80,
            "spatial_bandwidth_km": float(bandwidth),
        },
        "deterministic_full_test": deterministic,
        "deterministic_unseen_template_test": unseen_deterministic,
        "neural": [],
    }
    (output_dir / "tkde_results_deterministic.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)

    if args.neural:
        neural_results = []
        for model_name in ["GRU4Rec", "SASRec"]:
            for seed in args.seeds:
                neural_results.append(train_neural(model_name, train, val, test, n_items, long_tail, distances,
                                                   unseen_template_mask, output_dir, seed, args.epochs,
                                                   max_len=args.max_len, dim=args.dim))
        output["neural"] = neural_results
        (output_dir / "tkde_results_all.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"neural": neural_results}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--neural", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--skip-svd", action="store_true")
    main(parser.parse_args())
