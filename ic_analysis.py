"""IC integration and NaI calibration, independent of the Streamlit interface.

Uses upper-median noise estimation, local peak boundaries, and a continuous
origin-constrained fit. No calibration measurements or fitted coefficients are bundled.
"""
from __future__ import annotations

import csv
import io
import math
import re
from collections import defaultdict

import numpy as np


def upper_median(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return 0.0
    k = values.size // 2
    return float(np.partition(values, k)[k])


def parse_chromatogram(text: str, name: str = "sample.txt") -> dict:
    lines = text.lstrip("\ufeff").splitlines()
    start = next((i for i, line in enumerate(lines)
                  if re.match(r"^Time\s*\(min\)", line, re.I)), -1)
    chromeleon = start >= 0 and "Step" in lines[start]
    meta = dict(line.split("\t", 1) for line in lines[:max(start, 0)] if "\t" in line)
    rows = []
    for line in lines[start + 1:]:
        parts = re.split(r"[\t,; ]+", line.strip())
        try:
            t, value = float(parts[0]), float(parts[2 if chromeleon else 1])
        except (ValueError, IndexError):
            continue
        if math.isfinite(t) and math.isfinite(value):
            rows.append([t, value])
    if len(rows) < 20:
        raise ValueError("Insufficient data: upload Chromeleon TXT or two-column time (min), signal (uS) TXT/CSV.")
    xy = np.asarray(rows)
    if np.any(np.diff(xy[:, 0]) <= 0):
        raise ValueError("Time must be strictly increasing.")
    unit = meta.get("Signal Unit")
    if unit and unit not in ("µS", "μS", "uS"):
        raise ValueError("Signal units must be uS.")
    try:
        volume = float(meta["Volume (µl)"])
        if not math.isfinite(volume) or volume <= 0:
            raise ValueError("Injection volume must be positive.")
    except KeyError:
        volume = None
    return {"name": name, "xy": xy, "method": meta.get("Instrument Method", ""),
            "volume": volume}


def integrate(xy, left: float, right: float) -> dict:
    xy = np.asarray(xy, dtype=float)
    if not (math.isfinite(left) and math.isfinite(right)
            and xy[0, 0] <= left < right <= xy[-1, 0]):
        raise ValueError("Integration bounds must lie within the data range, with start before end.")
    a, b = np.interp([left, right], xy[:, 0], xy[:, 1])
    points = np.vstack(([left, a], xy[(xy[:, 0] > left) & (xy[:, 0] < right)], [right, b]))
    baseline = a + (b - a) * (points[:, 0] - left) / (right - left)
    area = float(np.trapezoid(points[:, 1] - baseline, points[:, 0]))
    if area <= 0:
        raise ValueError("Net peak area must be positive; adjust integration bounds.")
    apex = int(np.argmax(points[:, 1]))
    return {"left": float(left), "right": float(right), "area": area,
            "rt": float(points[apex, 0]), "height": float(points[apex, 1] - baseline[apex]),
            "baseline": [float(a), float(b)], "clipped": False}


def detect_peaks(xy, window=(2.0, math.inf)) -> list[dict]:
    xy = np.asarray(xy, dtype=float)
    left, right = window
    if not math.isfinite(left) or left < 0 or right <= left:
        raise ValueError("Invalid peak search window.")
    t, y = xy.T
    dt = (t[-1] - t[0]) / (len(t) - 1)
    radius = max(1, math.floor(.035 / dt + .5))
    indexes = np.arange(len(y))
    l, r = np.maximum(0, indexes - radius), np.minimum(len(y), indexes + radius + 1)
    sums = np.concatenate(([0.0], np.cumsum(y)))
    smoothed = (sums[r] - sums[l]) / (r - l)
    valid = indexes[(t >= left) & (t <= right)]
    if len(valid) < 20:
        raise ValueError("Insufficient data inside the peak search window.")
    base = float(np.sort(smoothed[valid])[math.floor(len(valid) * .15)])
    noise = upper_median(np.abs(np.diff(y))) / .954
    top = valid[np.argmax(smoothed[valid])]
    amplitude = smoothed[top] - base
    if amplitude < max(8 * noise, .002):
        raise ValueError("No significant positive peak detected.")
    candidates = [int(i) for i in valid if 0 < i < len(y) - 1
                  and smoothed[i] >= smoothed[i - 1] and smoothed[i] > smoothed[i + 1]
                  and smoothed[i] - base > max(8 * noise, amplitude * .08, .002)]
    candidates.sort(key=lambda i: smoothed[i], reverse=True)
    chosen = []
    for i in candidates:
        if all(abs(t[i] - t[j]) > .6 for j in chosen):
            chosen.append(i)
        if len(chosen) >= 12:
            break
    hold = max(3, math.floor(.08 / dt + .5))
    peaks = []
    for i in chosen:
        pre = smoothed[valid[(t[valid] >= t[i] - 3) & (t[valid] <= t[i] - 1)]]
        local_base = upper_median(pre) if len(pre) >= 20 else base
        local_noise = 1.4826 * upper_median(np.abs(pre - local_base)) if len(pre) >= 20 else noise
        if smoothed[i] - local_base < max(8 * noise, 6 * local_noise, amplitude * .08, .002):
            continue
        threshold = local_base + max(3 * noise, 3 * local_noise, (smoothed[i] - local_base) * .001)
        l = r = i
        count = 0
        while l > valid[0]:
            l -= 1
            count = count + 1 if smoothed[l] <= threshold else 0
            if count >= hold:
                break
        count = 0
        while r < valid[-1]:
            r += 1
            count = count + 1 if smoothed[r] <= threshold else 0
            if count >= hold:
                break
        if any(p["li"] <= i <= p["ri"] for p in peaks):
            continue
        try:
            peak = integrate(xy, float(t[l]), float(t[r]))
        except ValueError:
            continue
        peaks.append({**peak, "li": l, "ri": r,
                      "clipped": bool(l == valid[0] or r == valid[-1]),
                      "snr": peak["height"] / max(noise, 1e-9)})
    if not peaks:
        raise ValueError("Automatic peak detection failed; set integration bounds manually.")
    return sorted(peaks, key=lambda p: p["area"], reverse=True)


def calibration_group(standards):
    groups = defaultdict(list)
    for trace in standards:
        groups[(trace["method"], trace["volume"])].append(trace)
    group = max(groups.values(), key=len, default=[])
    if len(group) < 2:
        raise ValueError("At least two standards with the same instrument method and injection volume are required.")
    return group


def predict(model: dict, concentration):
    c = np.asarray(concentration, dtype=float)
    if model["kind"] == "piecewise":
        x = c / model["breakpoint"]
        low = model["joinArea"] * x + model["q"] * (x * x - x)
        high = model["joinArea"] + model["high"]["coeff"][1] * (c - model["breakpoint"])
        result = np.where(c <= model["breakpoint"], low, high)
    else:
        result = np.polynomial.polynomial.polyval(c, model["coeff"])
    return float(result) if result.ndim == 0 else result


def _stats(points, fn):
    y = np.asarray([p["area"] for p in points])
    predicted = np.asarray([fn(p["concentration"]) for p in points])
    sse, sst = float(np.sum((y - predicted) ** 2)), float(np.sum((y - np.mean(y)) ** 2))
    return {"r2": 1 - sse / sst if sst else 0.0,
            "rmse": math.sqrt(sse / len(points)), "count": len(points)}


def fit(points, kind="piecewise") -> dict:
    if any(not math.isfinite(p["concentration"]) or not math.isfinite(p["area"])
           or p["concentration"] <= 0 or p["area"] <= 0 for p in points):
        raise ValueError("Fitting requires finite positive concentrations and areas; the origin is a separate constraint.")
    if len(points) < 2:
        raise ValueError("At least two standards are required.")
    if kind != "piecewise":
        degree = 2 if kind == "quadratic" else 1
        if degree == 2 and len(points) < 4:
            raise ValueError("Quadratic fitting requires at least four standards.")
        c = np.asarray([p["concentration"] for p in points])
        scale = float(np.max(c))
        design = np.vander(c / scale, degree + 1, increasing=True)
        if np.linalg.matrix_rank(design) < degree + 1:
            raise ValueError("Insufficient distinct concentration levels for fitting.")
        coeff = np.linalg.lstsq(design, [p["area"] for p in points], rcond=None)[0] / scale ** np.arange(degree + 1)
        model = {"kind": kind, "coeff": coeff.tolist(), "min": float(min(c)), "max": scale,
                 "measuredMin": float(min(c))}
        model["monotonic"] = bool(coeff[1] > 0 and (degree == 1 or
                    min(coeff[1] + 2 * coeff[2] * min(c), coeff[1] + 2 * coeff[2] * scale) > 0))
    else:
        boundary = [p for p in points if abs(p["concentration"] - .001) < 1e-12]
        below = [p for p in points if p["concentration"] < .001 - 1e-12]
        above = [p for p in points if p["concentration"] > .001 + 1e-12]
        if not boundary or len(below) < 2 or len(above) < 2:
            raise ValueError("Piecewise fitting requires a 0.001 M boundary standard, two lower and two higher standards.")
        join = sum(p["area"] for p in boundary) / len(boundary)
        x = [p["concentration"] / .001 for p in below]
        q = sum((v * v - v) * (p["area"] - join * v) for p, v in zip(below, x)) / sum((v * v - v) ** 2 for v in x)
        slope = sum((p["concentration"] - .001) * (p["area"] - join) for p in above) / sum((p["concentration"] - .001) ** 2 for p in above)
        if min(join - q, join + q) < 0 or slope <= 0:
            raise ValueError("Cannot obtain a monotonically increasing piecewise curve; review standards and integration.")
        maximum = max(p["concentration"] for p in points)
        model = {"kind": kind, "breakpoint": .001, "joinArea": join, "q": q,
                 "min": 0.0, "measuredMin": min(p["concentration"] for p in points),
                 "max": maximum, "monotonic": True,
                 "origin": {"concentration": 0, "area": 0, "source": "constraint", "measured": False},
                 "low": {"kind": "quadratic", "coeff": [0.0, (join - q) / .001, q / .001 ** 2],
                         "min": 0, "max": .001, "originConstrained": True, "boundaryConstrained": True},
                 "high": {"kind": "linear", "coeff": [join - slope * .001, slope],
                          "min": .001, "max": maximum, "boundaryConstrained": True}}
        model["low"].update(_stats(below + boundary, lambda c: predict(model, c)))
        model["high"].update(_stats(boundary + above, lambda c: predict(model, c)))
    model.update(_stats(points, lambda c: predict(model, c)))
    return model


def invert(model, area: float) -> dict:
    if not math.isfinite(area) or area < 0:
        raise ValueError("Peak area must be finite and nonnegative.")
    if not model["monotonic"]:
        raise ValueError("The fitted curve is not monotonic; unique inversion is unavailable.")
    segment = "global"
    if model["kind"] == "piecewise":
        if area <= model["joinArea"]:
            l, r = 0.0, model["breakpoint"]
            for _ in range(70):
                mid = (l + r) / 2
                if predict(model, mid) < area:
                    l = mid
                else:
                    r = mid
            concentration = 0.0 if area == 0 else model["breakpoint"] if area == model["joinArea"] else (l + r) / 2
            segment = "low"
        else:
            concentration = model["breakpoint"] + (area - model["joinArea"]) / model["high"]["coeff"][1]
            segment = "high"
    else:
        a, b, *quadratic = model["coeff"]
        c = quadratic[0] if quadratic else 0
        if not c:
            concentration = (area - a) / b
        else:
            discriminant = b * b - 4 * c * (a - area)
            if discriminant < 0:
                raise ValueError("This area cannot be inverted with the current curve.")
            roots = [r for r in ((-b + math.sqrt(discriminant)) / (2 * c), (-b - math.sqrt(discriminant)) / (2 * c))
                     if r >= 0 and b + 2 * c * r > 0]
            if len(roots) != 1:
                raise ValueError("No unique nonnegative concentration solution.")
            concentration = roots[0]
    if concentration < 0 or not math.isfinite(concentration):
        raise ValueError("Area is below the fitted intercept; no nonnegative concentration is available.")
    below = area < predict(model, model["measuredMin"])
    return {"concentration": concentration, "segment": segment,
            "inRange": not below and area <= predict(model, model["max"]), "belowMeasuredRange": below}


def quantify(trace, peak, model, reference, dilution=1.0, confirmed=False):
    if not peak:
        raise ValueError("No valid peak selected.")
    if peak.get("clipped"):
        raise ValueError("Peak may be truncated; review integration bounds or upload the complete trace.")
    if trace["method"] and trace["method"] != reference["method"]:
        raise ValueError("Instrument method differs from the calibration method.")
    if trace["volume"] is not None and abs(trace["volume"] - reference["volume"]) > 1e-6:
        raise ValueError("Injection volume differs from the calibration standards.")
    if (not trace["method"] or trace["volume"] is None) and not confirmed:
        raise ValueError("Missing method or injection volume; confirm matching measurement conditions.")
    if not math.isfinite(dilution) or dilution <= 0:
        raise ValueError("Dilution factor must be finite and positive.")
    result = invert(model, peak["area"])
    original = result["concentration"] * dilution
    return {**result, "original_M": original, "dilution": dilution,
            "NaI_mg_L": original * 149894, "iodide_mg_L": original * 126904.47}


def csv_bytes(rows):
    if not rows:
        return b""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        # Keep arbitrary uploaded filenames from becoming spreadsheet formulas.
        writer.writerow({k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v
                         for k, v in row.items()})
    return buffer.getvalue().encode("utf-8-sig")
