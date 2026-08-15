from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_sequence(value) -> list[str]:
    """Parse a stored ordered attraction sequence without changing its order."""
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass
    return [item.strip().strip("'\"") for item in text.strip("[]").split(",") if item.strip()]


def great_circle_matrix(longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    """Return pairwise great-circle distances in kilometres."""
    lon_1, lon_2 = np.radians(longitude)[:, None], np.radians(longitude)[None, :]
    lat_1, lat_2 = np.radians(latitude)[:, None], np.radians(latitude)[None, :]
    value = np.sin((lat_2 - lat_1) / 2) ** 2
    value += np.cos(lat_1) * np.cos(lat_2) * np.sin((lon_2 - lon_1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def row_normalize(matrix: np.ndarray, add_self_loops_to_zero_rows: bool = False) -> np.ndarray:
    """Normalize positive rows and optionally close zero rows with self-loops."""
    result = matrix.astype(float, copy=True)
    row_sums = result.sum(axis=1)
    if add_self_loops_to_zero_rows:
        zero_rows = np.where(row_sums == 0)[0]
        result[zero_rows, zero_rows] = 1.0
        row_sums = result.sum(axis=1)
    positive_rows = row_sums > 0
    result[positive_rows] /= row_sums[positive_rows, None]
    return result


def collect_counts(sequences: list[list[int]], state_count: int):
    """Construct first-order counts, second-order contexts and held-out events."""
    counts = np.zeros((state_count, state_count), dtype=float)
    second_order = defaultdict(Counter)
    events = []
    for sequence in sequences:
        for position in range(len(sequence) - 1):
            origin, destination = sequence[position], sequence[position + 1]
            if origin == destination:
                continue
            previous = sequence[position - 1] if position > 0 else -1
            counts[origin, destination] += 1
            events.append((previous, origin, destination))
            if previous >= 0:
                second_order[(previous, origin)][destination] += 1
    return counts, second_order, events


def top_candidates(score: np.ndarray, cutoff: int) -> np.ndarray:
    """Return sorted row-specific candidate indices at the requested cutoff."""
    cutoff = min(int(cutoff), score.shape[1])
    indices = np.argpartition(-score, np.arange(cutoff), axis=1)[:, :cutoff]
    values = np.take_along_axis(score, indices, axis=1)
    order = np.argsort(-values, axis=1)
    return np.take_along_axis(indices, order, axis=1)


def evaluate(events, candidates: np.ndarray, state_count: int, distances=None, long_tail=None):
    """Evaluate rankings with explicit five- and ten-item cutoffs."""
    origins = np.asarray([event[1] for event in events], dtype=int)
    targets = np.asarray([event[2] for event in events], dtype=int)
    recommendations = candidates[origins]
    matches = recommendations == targets[:, None]
    positions = np.where(matches, np.arange(1, recommendations.shape[1] + 1), 0).max(axis=1)
    ranks = np.where(positions > 0, positions, np.inf)
    at_10 = recommendations[:, : min(10, recommendations.shape[1])]
    output = {
        "test_transition_count": int(len(events)),
        "Hit@5": float((ranks <= 5).mean()),
        "Hit@10": float((ranks <= 10).mean()),
        "MRR": float(np.where(np.isfinite(ranks), 1 / ranks, 0).mean()),
        "NDCG@10": float(np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0).mean()),
        "catalog_coverage_at_10": float(len(np.unique(at_10)) / state_count),
    }
    if long_tail is not None:
        mask = np.asarray([long_tail[target] for target in targets], dtype=bool)
        output["long_tail_test_count"] = int(mask.sum())
        output["long_tail_recall_at_10"] = float((ranks[mask] <= 10).mean()) if mask.any() else np.nan
    if distances is not None:
        selected = distances[origins[:, None], at_10]
        output["mean_recommendation_distance_km"] = float(np.nanmean(selected))
        output["median_recommendation_distance_km"] = float(np.nanmedian(selected))
    return output


def stationary_distribution(matrix: np.ndarray, tolerance: float = 1e-14, maximum_iterations: int = 20000):
    """Solve pi = pi P by power iteration from the uniform initial distribution."""
    probability = np.ones(matrix.shape[0]) / matrix.shape[0]
    difference = np.inf
    for iteration in range(1, maximum_iterations + 1):
        updated = probability @ matrix
        difference = float(np.abs(updated - probability).sum())
        probability = updated
        if difference < tolerance:
            break
    residual = float(np.abs(probability @ matrix - probability).sum())
    return probability, iteration, difference, residual


def main(package_root: Path, data_dir: Path | None = None) -> None:
    data_dir = data_dir.resolve() if data_dir is not None else package_root / "data"
    output_dir = package_root / "reproduced_outputs"
    output_dir.mkdir(exist_ok=True)

    routes = pd.read_excel(data_dir / "Supplementary_Data_S2_Cleaned_Routes.xlsx", sheet_name="Cleaned_Model_Routes")
    states = pd.read_excel(data_dir / "Supplementary_Data_S3_States_and_Transitions.xlsx", sheet_name="Attraction_States")
    states = states.sort_values("state_id").reset_index(drop=True)
    state_count = len(states)
    name_to_id = dict(zip(states["attraction_name_standardized"].astype(str), states["state_id"].astype(int)))

    prepared = []
    for row in routes.itertuples(index=False):
        identifiers = [name_to_id[name] for name in parse_sequence(row.attraction_sequence_cleaned) if name in name_to_id]
        sequence = []
        for identifier in identifiers:
            if not sequence or sequence[-1] != identifier:
                sequence.append(identifier)
        date = pd.to_datetime(row.departure_date, errors="coerce")
        if pd.notna(date) and date.year < 2010:
            date = pd.NaT
        if len(sequence) >= 2 and pd.notna(date):
            prepared.append((date, int(row.record_id), sequence))
    prepared.sort(key=lambda item: (item[0], item[1]))

    train_end = int(len(prepared) * 0.70)
    validation_end = int(len(prepared) * 0.85)
    train = prepared[:train_end]
    validation = prepared[train_end:validation_end]
    test = prepared[validation_end:]

    train_counts, second_counts, _ = collect_counts([item[2] for item in train], state_count)
    _, _, validation_events = collect_counts([item[2] for item in validation], state_count)
    _, _, test_events = collect_counts([item[2] for item in test], state_count)
    observed = row_normalize(train_counts)

    longitude = states["longitude_wgs84"].astype(float).to_numpy()
    latitude = states["latitude_wgs84"].astype(float).to_numpy()
    distances = great_circle_matrix(longitude, latitude)
    bandwidth = float(np.median(distances[distances > 0]))
    spatial = np.exp(-distances / bandwidth)
    np.fill_diagonal(spatial, 0)
    spatial = row_normalize(spatial)
    two_step = row_normalize(observed @ observed)
    destination_frequency = train_counts.sum(axis=0)
    pull = np.tile(destination_frequency / max(destination_frequency.sum(), 1), (state_count, 1))
    np.fill_diagonal(pull, 0)
    pull = row_normalize(pull)

    positive_frequency = destination_frequency[destination_frequency > 0]
    long_tail_cutoff = np.quantile(positive_frequency, 0.80)
    long_tail = destination_frequency <= long_tail_cutoff

    popularity = np.tile(np.argsort(-destination_frequency)[:10], (state_count, 1))
    nearest = top_candidates(-distances, 10)
    first_order = top_candidates(observed + 1e-12 * pull, 10)
    markov_spatial = top_candidates(0.75 * observed + 0.25 * spatial, 10)

    second_candidates = {}
    for context, counter in second_counts.items():
        candidate_list = [item for item, _ in counter.most_common()]
        for item in first_order[context[1]]:
            if int(item) not in candidate_list:
                candidate_list.append(int(item))
            if len(candidate_list) >= 10:
                break
        second_candidates[context] = np.asarray(candidate_list[:10], dtype=int)

    def evaluate_second_order(events):
        rows = [second_candidates.get((previous, origin), first_order[origin]) for previous, origin, _ in events]
        patched = [(-1, origin, target) for _, origin, target in events]
        return evaluate(patched, np.vstack(rows), state_count, distances, long_tail)

    weights = []
    for observed_weight in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for spatial_weight in [0, 0.1, 0.2, 0.3, 0.4]:
            for two_step_weight in [0, 0.1, 0.2, 0.3, 0.4]:
                pull_weight = round(1 - observed_weight - spatial_weight - two_step_weight, 10)
                if 0 <= pull_weight <= 0.4:
                    weights.append((observed_weight, spatial_weight, two_step_weight, pull_weight))

    validation_rows = []
    best = None
    for weight_tuple in weights:
        score = sum(weight * matrix for weight, matrix in zip(weight_tuple, [observed, spatial, two_step, pull]))
        candidates = top_candidates(score, 10)
        metrics = evaluate(validation_events, candidates, state_count, distances, long_tail)
        row = {
            "observed_weight": weight_tuple[0],
            "spatial_weight": weight_tuple[1],
            "two_step_weight": weight_tuple[2],
            "attraction_pull_weight": weight_tuple[3],
            "candidate_cutoff": 10,
            **metrics,
        }
        validation_rows.append(row)
        selection_key = (metrics["NDCG@10"], metrics["Hit@10"], metrics["long_tail_recall_at_10"])
        if best is None or selection_key > best[0]:
            best = (selection_key, weight_tuple)

    selected_weights = best[1]
    complete_score = sum(weight * matrix for weight, matrix in zip(selected_weights, [observed, spatial, two_step, pull]))
    complete = top_candidates(complete_score, 10)
    performance = pd.DataFrame(
        [
            {"model": "Global popularity", **evaluate(test_events, popularity, state_count, distances, long_tail)},
            {"model": "Nearest distance", **evaluate(test_events, nearest, state_count, distances, long_tail)},
            {"model": "First-order Markov", **evaluate(test_events, first_order, state_count, distances, long_tail)},
            {"model": "Second-order Markov", **evaluate_second_order(test_events)},
            {"model": "Markov plus spatial prior", **evaluate(test_events, markov_spatial, state_count, distances, long_tail)},
            {"model": "Full multi-source model", **evaluate(test_events, complete, state_count, distances, long_tail)},
        ]
    )

    observed_edges = pd.read_excel(data_dir / "Supplementary_Data_S3_States_and_Transitions.xlsx", sheet_name="Observed_Transitions")
    full_counts = np.zeros((state_count, state_count), dtype=float)
    for row in observed_edges.itertuples(index=False):
        full_counts[int(row.origin_state_id), int(row.destination_state_id)] += float(row.transition_count)
    stationary_matrix = row_normalize(full_counts, add_self_loops_to_zero_rows=True)
    stationary, iterations, difference, residual = stationary_distribution(stationary_matrix)
    stationary_table = pd.DataFrame(
        {
            "state_id": states["state_id"].astype(int),
            "attraction_name": states["attraction_name_standardized"],
            "stationary_probability": stationary,
        }
    ).sort_values("stationary_probability", ascending=False)
    diagnostics = pd.DataFrame(
        [
            {"metric": "iterations", "value": iterations},
            {"metric": "last_iteration_l1_difference", "value": difference},
            {"metric": "stationary_equation_l1_residual", "value": residual},
            {"metric": "selected_observed_weight", "value": selected_weights[0]},
            {"metric": "selected_spatial_weight", "value": selected_weights[1]},
            {"metric": "selected_two_step_weight", "value": selected_weights[2]},
            {"metric": "selected_attraction_pull_weight", "value": selected_weights[3]},
            {"metric": "candidate_cutoff", "value": 10},
        ]
    )

    output_path = output_dir / "Reproduced_Model_Results.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        performance.to_excel(writer, sheet_name="Test_Performance", index=False)
        pd.DataFrame(validation_rows).sort_values(["NDCG@10", "Hit@10"], ascending=False).to_excel(
            writer, sheet_name="Validation_Grid", index=False
        )
        stationary_table.head(50).to_excel(writer, sheet_name="Stationary_Top50", index=False)
        diagnostics.to_excel(writer, sheet_name="Diagnostics", index=False)

    summary = {
        "modeling_routes": int(len(routes)),
        "dated_modeling_routes": len(prepared),
        "training_routes": len(train),
        "validation_routes": len(validation),
        "test_routes": len(test),
        "test_transitions": len(test_events),
        "selected_weights": {
            "observed": selected_weights[0],
            "spatial": selected_weights[1],
            "two_step": selected_weights[2],
            "attraction_pull": selected_weights[3],
        },
        "candidate_cutoff": 10,
        "stationary_equation_l1_residual": residual,
    }
    (output_dir / "reproduction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce the transition-probability analyses.")
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing the four supplementary Excel data files.")
    arguments = parser.parse_args()
    main(arguments.package_root.resolve(), arguments.data_dir)
