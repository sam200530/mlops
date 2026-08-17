"""Replay real transactions against a running API.

Monitoring needs traffic, and this project has no users. Rather than invent
numbers, this script replays **real rows from the dataset** at the live service,
which exercises the whole serving path end to end: Pydantic validation, the Redis
velocity store, the model, calibration, the prediction log, and the metrics
counters.

Rows are replayed in chronological order, because the velocity features depend on
arrival order — shuffling would produce trailing-window values that could never
occur in reality.

Every report this generates is labelled as simulated traffic. Usage:

    python scripts/simulate_traffic.py --n 200
    python scripts/simulate_traffic.py --n 500 --batch-size 50 --explain 5
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.schema import TARGET  # noqa: E402
from src.utils.logging_config import setup_logging  # noqa: E402
from src.utils.paths import PROCESSED_DIR, REPORTS_DIR, ensure_dir  # noqa: E402

logger = logging.getLogger("simulate_traffic")

#: Fields the API schema names explicitly; everything else goes to extra_features.
NAMED_FIELDS = {
    "TransactionAmt": "transaction_amt",
    "ProductCD": "product_cd",
    "card1": "card1",
    "card2": "card2",
    "card3": "card3",
    "card4": "card4",
    "card5": "card5",
    "card6": "card6",
    "addr1": "addr1",
    "addr2": "addr2",
    "dist1": "dist1",
    "dist2": "dist2",
    "P_emaildomain": "p_emaildomain",
    "R_emaildomain": "r_emaildomain",
    "DeviceType": "device_type",
    "DeviceInfo": "device_info",
    "TransactionDT": "transaction_dt",
}

#: Raw columns forwarded through extra_features. Engineered columns are excluded —
#: the service computes those itself, which is the point of replaying raw rows.
EXTRA_PREFIXES = ("C", "D", "M", "V", "id_")


def _clean(value: Any) -> Any:
    """Convert pandas/NumPy values into JSON-safe primitives."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            return str(value)
    if isinstance(value, float) and value != value:
        return None
    return value


def row_to_payload(row: pd.Series, include_extras: bool) -> dict[str, Any]:
    """Convert a dataset row into an API request body."""
    payload: dict[str, Any] = {}
    for column, field in NAMED_FIELDS.items():
        if column in row.index:
            cleaned = _clean(row[column])
            if cleaned is not None:
                payload[field] = cleaned

    payload.setdefault("transaction_amt", 100.0)
    payload.setdefault("product_cd", "W")

    if include_extras:
        extras: dict[str, Any] = {}
        for column in row.index:
            name = str(column)
            if name in NAMED_FIELDS or name == TARGET or name.startswith("_"):
                continue
            if name.startswith(EXTRA_PREFIXES) and _is_raw_column(name):
                cleaned = _clean(row[column])
                if cleaned is not None:
                    extras[name] = cleaned
        if extras:
            payload["extra_features"] = extras
    return payload


def _is_raw_column(name: str) -> bool:
    """True for original dataset columns, false for engineered ones."""
    if name.startswith("id_"):
        return len(name) == 5 and name[3:].isdigit()
    prefix, rest = name[0], name[1:]
    return prefix in {"C", "D", "M", "V"} and rest.isdigit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay dataset rows against the API.")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200, help="Transactions to send.")
    parser.add_argument("--batch-size", type=int, default=0, help="0 = single /predict calls.")
    parser.add_argument("--explain", type=int, default=3, help="Number of /explain calls.")
    parser.add_argument("--source", default="holdout", choices=["holdout", "test", "modelling"])
    parser.add_argument("--no-extras", action="store_true", help="Send only the named fields.")
    parser.add_argument(
        "--repeat-fraction",
        type=float,
        default=0.1,
        help="Fraction of requests resent verbatim, to exercise the cache.",
    )
    args = parser.parse_args()

    setup_logging()
    import httpx

    path = PROCESSED_DIR / f"{args.source}_prepared.parquet"
    if not path.is_file():
        logger.error("%s missing — run scripts/build_dataset.py first", path)
        return 1

    df = pd.read_parquet(path)
    # Chronological order matters: velocity features depend on arrival sequence.
    if "TransactionDT" in df.columns:
        df = df.sort_values("TransactionDT")
    df = df.head(args.n)
    payloads = [row_to_payload(row, include_extras=not args.no_extras) for _, row in df.iterrows()]
    logger.info("Prepared %d payloads from %s", len(payloads), path.name)

    latencies: list[float] = []
    probabilities: list[float] = []
    errors: list[str] = []
    risk_counts: dict[str, int] = {}
    started = time.perf_counter()

    with httpx.Client(base_url=args.url, timeout=60.0) as client:
        try:
            health = client.get("/health").json()
            logger.info("API health: %s", health)
            if not health.get("model_loaded"):
                logger.error("API reports no model loaded — run scripts/train.py first")
                return 1
        except Exception as error:  # noqa: BLE001
            logger.error("Cannot reach %s: %s", args.url, error)
            return 1

        if args.batch_size > 0:
            for start in range(0, len(payloads), args.batch_size):
                chunk = payloads[start : start + args.batch_size]
                response = client.post("/predict/batch", json={"transactions": chunk})
                if response.status_code != 200:
                    errors.append(f"batch {start}: {response.status_code} {response.text[:120]}")
                    continue
                body = response.json()
                latencies.append(body["latency_ms"])
                for prediction in body["predictions"]:
                    probabilities.append(prediction["fraud_probability"])
                    risk_counts[prediction["risk_level"]] = (
                        risk_counts.get(prediction["risk_level"], 0) + 1
                    )
        else:
            repeat_every = int(1 / args.repeat_fraction) if 0 < args.repeat_fraction < 1 else 0
            for index, payload in enumerate(payloads):
                response = client.post("/predict", json=payload)
                if response.status_code != 200:
                    errors.append(f"row {index}: {response.status_code} {response.text[:120]}")
                    continue
                body = response.json()
                latencies.append(body["latency_ms"])
                probabilities.append(body["fraud_probability"])
                risk_counts[body["risk_level"]] = risk_counts.get(body["risk_level"], 0) + 1

                # Resend verbatim to exercise the prediction cache.
                if repeat_every and index % repeat_every == 0:
                    client.post("/predict", json=payload)

        explanations: list[dict[str, Any]] = []
        for payload in payloads[: max(0, args.explain)]:
            response = client.post("/explain", json=payload)
            if response.status_code == 200:
                explanations.append(response.json())
            else:
                errors.append(f"explain: {response.status_code} {response.text[:120]}")

        metrics = client.get("/monitoring/metrics").json()

    elapsed = time.perf_counter() - started
    report = {
        "traffic_source": "SIMULATED — real dataset rows replayed locally, not production traffic",
        "source_partition": args.source,
        "requests_sent": len(payloads),
        "successful": len(probabilities),
        "errors": errors[:10],
        "n_errors": len(errors),
        "elapsed_seconds": round(elapsed, 2),
        "throughput_rps": round(len(probabilities) / elapsed, 2) if elapsed else None,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else None,
            "p50": round(statistics.median(latencies), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
        "fraud_probability": {
            "mean": round(statistics.fmean(probabilities), 6) if probabilities else None,
            "max": round(max(probabilities), 6) if probabilities else None,
        },
        "risk_levels": risk_counts,
        "service_metrics": metrics,
        "example_explanations": explanations[:2],
    }

    out = ensure_dir(REPORTS_DIR / "monitoring") / "simulated_traffic.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Sent %d requests (%d ok, %d errors) in %.1fs | mean latency %s ms | risk %s",
        len(payloads),
        len(probabilities),
        len(errors),
        elapsed,
        report["latency_ms"]["mean"],
        risk_counts,
    )
    logger.info("Wrote %s", out)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
