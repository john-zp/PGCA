"""Paired cluster resampling; seeds are not independent test observations."""
import numpy as np
import pandas as pd

METRICS = ["Accuracy", "MAE", "QWK", "Macro-F1"]
CLASSES = ["2", "3", "4A", "4B", "4C", "5"]


def cm_metrics(cm):
    cm = np.asarray(cm, dtype=float)
    total = cm.sum(axis=(-2, -1))
    diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
    actual, predicted = cm.sum(-1), cm.sum(-2)
    distance = abs(np.arange(6)[:, None] - np.arange(6)[None, :])
    accuracy = diagonal.sum(-1) / total
    mae = (cm * distance).sum(axis=(-2, -1)) / total
    expected = actual[..., :, None] * predicted[..., None, :] / total[..., None, None]
    expected_cost = (expected * distance ** 2).sum(axis=(-2, -1))
    observed_cost = (cm * distance ** 2).sum(axis=(-2, -1))
    qwk = 1 - np.divide(observed_cost, expected_cost, out=np.full_like(observed_cost, np.nan), where=expected_cost != 0)
    f1 = np.divide(2 * diagonal, actual + predicted, out=np.zeros_like(diagonal), where=(actual + predicted) != 0)
    assert np.all(mae >= 1 - accuracy - 1e-12)
    return np.stack([accuracy, mae, qwk, f1.mean(-1)], -1), f1


def hard_metrics(frame):
    y, p = frame.y_true.to_numpy(int), frame.y_pred.to_numpy(int)
    assert np.isin(y, range(1, 7)).all() and np.isin(p, range(1, 7)).all()
    cm = np.bincount((y - 1) * 6 + p - 1, minlength=36).reshape(6, 6)
    values, f1 = cm_metrics(cm)
    return {**dict(zip(METRICS, values.tolist())), **{f"F1_{c}": float(v) for c, v in zip(CLASSES, f1)}}


def classification_pair(left, right, name, replicates=2000):
    seeds = sorted(set(left) & set(right))
    reference = left[seeds[0]].sort_values("image_id")
    groups = sorted(reference.group_id.unique())
    group_index = {g:i for i,g in enumerate(groups)}
    group = np.array([group_index[g] for g in reference.group_id])
    weights = np.random.default_rng(724299).multinomial(len(groups), np.ones(len(groups))/len(groups), size=replicates)
    samples, points = [], []
    for seed in seeds:
        pair_values = []
        pair_points = []
        for frame in [left[seed], right[seed]]:
            frame = frame.sort_values("image_id")
            assert np.array_equal(frame.image_id, reference.image_id)
            assert np.array_equal(frame.y_true, reference.y_true)
            assert np.array_equal(frame.group_id, reference.group_id)
            bins = group*36 + (frame.y_true.to_numpy(int)-1)*6 + frame.y_pred.to_numpy(int)-1
            cubes = np.bincount(bins, minlength=len(groups)*36).reshape(len(groups), 36)
            pair_values.append(cm_metrics((weights @ cubes).reshape(replicates, 6, 6))[0])
            pair_points.append(cm_metrics(cubes.sum(0).reshape(6, 6))[0])
        samples.append(pair_values[0]-pair_values[1])
        points.append(pair_points[0]-pair_points[1])
    result = []
    for seed, point, draws in list(zip(seeds, points, samples)) + [("mean_across_seeds", np.mean(points, 0), np.mean(samples, 0))]:
        assert np.isfinite(draws).all()
        for i, metric in enumerate(METRICS):
            low, high = np.quantile(draws[:, i], [.025, .975])
            result.append(dict(comparison=name, seed=seed, metric=metric, difference=float(point[i]), ci_low=float(low), ci_high=float(high),
                test_images=len(reference), groups=len(groups), replicates=replicates, exploratory=True))
    return result


def scalar_pair(left, right, value, name, replicates=2000):
    joined = left.merge(right, on=["image_id", "group_id", "seed"], suffixes=("_a", "_b"), validate="one_to_one")
    assert len(joined) == len(left) == len(right)
    groups = sorted(joined.group_id.unique())
    weights = np.random.default_rng(724299).multinomial(len(groups), np.ones(len(groups))/len(groups), size=replicates)
    points, samples, seeds = [], [], sorted(joined.seed.unique())
    for seed in seeds:
        frame = joined[joined.seed == seed].copy()
        frame["delta"] = frame[value + "_a"] - frame[value + "_b"]
        grouped = frame.groupby("group_id").delta.agg(["sum", "count"]).reindex(groups, fill_value=0)
        samples.append((weights @ grouped["sum"]) / (weights @ grouped["count"]))
        points.append(frame.delta.mean())
    output = []
    for seed, point, draws in list(zip(seeds, points, samples)) + [("mean_across_seeds", np.mean(points), np.mean(samples, 0))]:
        low, high = np.quantile(draws, [.025, .975])
        output.append(dict(comparison=name, seed=str(seed), metric=value, difference=float(point), ci_low=float(low), ci_high=float(high),
            test_images=len(joined)//len(seeds), groups=len(groups), replicates=replicates, exploratory=True))
    return output


def with_mean_sd(frame, by, metrics):
    rows = []
    for values, part in frame.groupby(by, sort=False):
        values = values if isinstance(values, tuple) else (values,)
        context = dict(zip(by, values))
        for name, operation in [("mean", lambda s:s.mean()), ("sample_SD", lambda s:s.std(ddof=1))]:
            rows.append({**context, "seed": name, "record_type": name, **{m:float(operation(part[m])) for m in metrics}, "completed_seeds": len(part)})
    result = frame.copy()
    result["record_type"] = "seed"
    return pd.concat([result, pd.DataFrame(rows)], ignore_index=True)
