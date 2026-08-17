"""API contract tests.

These run against a real FastAPI TestClient with a real (small) trained model, so
they exercise the full scoring path: Pydantic validation, frame construction, the
velocity store, the feature pipeline, the model and calibration. Postgres and
Redis are disabled, which also verifies the service degrades cleanly without them.
"""

from __future__ import annotations

import pytest

from api.velocity_store import VelocityStore, compute_entity_keys


class TestHealth:
    def test_health_reports_model_loaded(self, api_client) -> None:
        response = api_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["model_loaded"] is True
        assert body["redis"] == "disabled"
        assert body["database"] == "disabled"
        assert body["uptime_seconds"] >= 0

    def test_root_points_at_docs(self, api_client) -> None:
        assert api_client.get("/").json()["docs"] == "/docs"


class TestModelInfo:
    def test_returns_provenance_and_thresholds(self, api_client) -> None:
        response = api_client.get("/model-info")
        assert response.status_code == 200
        body = response.json()
        assert body["model_name"] == "lightgbm"
        assert body["n_features"] > 0
        assert body["n_train_rows"] > 0
        assert set(body["risk_thresholds"]) == {"medium", "high"}
        assert 0.0 <= body["decision_threshold"] <= 1.0


class TestPredict:
    def test_returns_a_valid_probability_and_risk_band(
        self, api_client, valid_transaction_payload
    ) -> None:
        response = api_client.post("/predict", json=valid_transaction_payload)
        assert response.status_code == 200, response.text
        body = response.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["risk_level"] in {"low", "medium", "high"}
        assert body["request_id"]
        assert body["latency_ms"] > 0
        assert body["features_supplied"] > 0
        assert isinstance(body["flagged"], bool)

    def test_risk_band_is_consistent_with_the_probability(
        self, api_client, valid_transaction_payload
    ) -> None:
        body = api_client.post("/predict", json=valid_transaction_payload).json()
        probability = body["fraud_probability"]
        expected = "high" if probability >= 0.70 else "medium" if probability >= 0.30 else "low"
        assert body["risk_level"] == expected

    def test_works_with_only_the_required_fields(self, api_client) -> None:
        # Everything omitted becomes NaN, which the model handles natively.
        response = api_client.post("/predict", json={"transaction_amt": 50.0, "product_cd": "W"})
        assert response.status_code == 200, response.text
        assert 0.0 <= response.json()["fraud_probability"] <= 1.0

    def test_response_carries_request_id_header(
        self, api_client, valid_transaction_payload
    ) -> None:
        response = api_client.post("/predict", json=valid_transaction_payload)
        assert response.headers["x-request-id"]
        assert float(response.headers["x-process-time-ms"]) >= 0


class TestPredictValidation:
    @pytest.mark.parametrize(
        "payload,reason",
        [
            ({}, "missing required fields"),
            ({"transaction_amt": 100.0}, "missing product_cd"),
            ({"transaction_amt": -5.0, "product_cd": "W"}, "negative amount"),
            ({"transaction_amt": 0.0, "product_cd": "W"}, "zero amount"),
            ({"transaction_amt": "abc", "product_cd": "W"}, "non-numeric amount"),
            ({"transaction_amt": 10.0, "product_cd": ""}, "empty product code"),
            ({"transaction_amt": 10.0, "product_cd": "W", "card1": -1}, "negative card1"),
            (
                {"transaction_amt": 10.0, "product_cd": "W", "unknown_field": 1},
                "unknown field rejected by extra='forbid'",
            ),
        ],
    )
    def test_invalid_payloads_return_422(self, api_client, payload, reason) -> None:
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 422, f"{reason}: {response.text}"
        body = response.json()
        assert body["error_type"] == "RequestValidationError"
        assert "errors" in body

    def test_validation_failures_are_counted(self, api_client) -> None:
        from api import dependencies

        before = dependencies.state.metrics.counters().get("validation_failures", 0)
        api_client.post("/predict", json={"product_cd": "W"})
        after = dependencies.state.metrics.counters().get("validation_failures", 0)
        assert after == before + 1


class TestPredictBatch:
    def test_scores_every_transaction(self, api_client, valid_transaction_payload) -> None:
        payload = {"transactions": [valid_transaction_payload for _ in range(5)]}
        response = api_client.post("/predict/batch", json=payload)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["count"] == 5
        assert len(body["predictions"]) == 5
        for prediction in body["predictions"]:
            assert 0.0 <= prediction["fraud_probability"] <= 1.0

    def test_request_ids_are_unique_within_a_batch(
        self, api_client, valid_transaction_payload
    ) -> None:
        payload = {"transactions": [valid_transaction_payload for _ in range(4)]}
        body = api_client.post("/predict/batch", json=payload).json()
        ids = {prediction["request_id"] for prediction in body["predictions"]}
        assert len(ids) == 4

    def test_predictions_stay_aligned_with_request_order(self, api_client) -> None:
        """Regression: velocity requires chronological processing, so the frame is
        sorted internally. Responses must still come back in submitted order."""
        # Submitted newest-first, i.e. deliberately not chronological.
        transactions = [
            {"transaction_amt": 10.0, "product_cd": "W", "transaction_dt": 9_000_000},
            {"transaction_amt": 20.0, "product_cd": "C", "transaction_dt": 3_000_000},
            {"transaction_amt": 30.0, "product_cd": "R", "transaction_dt": 6_000_000},
        ]
        batch = api_client.post("/predict/batch", json={"transactions": transactions})
        assert batch.status_code == 200, batch.text
        batch_predictions = batch.json()["predictions"]
        assert len(batch_predictions) == 3

        # features_supplied is a per-record property, so it pins each response to
        # the record it came from regardless of internal reordering.
        for submitted, prediction in zip(transactions, batch_predictions):
            single = api_client.post("/predict", json=submitted).json()
            assert prediction["features_supplied"] == single["features_supplied"]

    def test_empty_batch_is_rejected(self, api_client) -> None:
        assert api_client.post("/predict/batch", json={"transactions": []}).status_code == 422

    def test_oversized_batch_is_rejected(self, api_client, valid_transaction_payload) -> None:
        payload = {"transactions": [valid_transaction_payload] * 501}
        assert api_client.post("/predict/batch", json=payload).status_code == 422

    def test_batch_with_one_invalid_row_is_rejected_entirely(
        self, api_client, valid_transaction_payload
    ) -> None:
        payload = {"transactions": [valid_transaction_payload, {"transaction_amt": -1.0}]}
        assert api_client.post("/predict/batch", json=payload).status_code == 422


class TestExplain:
    def test_returns_ranked_shap_contributions(self, api_client, valid_transaction_payload) -> None:
        response = api_client.post("/explain", json=valid_transaction_payload)
        assert response.status_code == 200, response.text
        body = response.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["top_factors"], "expected at least one contributing feature"
        magnitudes = [abs(factor["shap_value"]) for factor in body["top_factors"]]
        assert magnitudes == sorted(magnitudes, reverse=True)
        for factor in body["top_factors"]:
            assert factor["direction"] in {"increases_risk", "decreases_risk"}
            assert factor["feature"]

    def test_top_n_is_respected(self, api_client, valid_transaction_payload) -> None:
        body = api_client.post("/explain?top_n=3", json=valid_transaction_payload).json()
        assert len(body["top_factors"]) == 3

    def test_invalid_top_n_is_rejected(self, api_client, valid_transaction_payload) -> None:
        response = api_client.post("/explain?top_n=999", json=valid_transaction_payload)
        assert response.status_code == 422


class TestMonitoringEndpoints:
    def test_metrics_reports_counters_and_backends(
        self, api_client, valid_transaction_payload
    ) -> None:
        api_client.post("/predict", json=valid_transaction_payload)
        body = api_client.get("/monitoring/metrics").json()
        assert body["traffic_source"] == "local_or_simulated"
        assert body["counters"]["predict_requests"] >= 1
        assert body["latency_ms"]["count"] >= 1
        assert body["velocity_backend"] == "in_memory_fallback"

    def test_prediction_drift_reports_insufficient_data_early(self, api_client) -> None:
        body = api_client.get("/monitoring/prediction-drift").json()
        assert body["status"] in {"insufficient_data", "ok"} or "n_recent_scores" in body

    def test_prediction_summary_reports_disabled_logging(self, api_client) -> None:
        assert api_client.get("/monitoring/predictions/summary").json()["status"] == "disabled"


class TestVelocityStore:
    def test_entity_keys_match_the_offline_definition(self) -> None:
        keys = compute_entity_keys({"card1": 13926, "addr1": 315, "card2": 404})
        assert keys["_entity_card"] == 13926
        assert keys["_entity_card_addr"] == 13926 * 1000 + 315
        assert keys["_entity_card_full"] == 13926 * 1_000_000 + 315 * 1000 + 404

    def test_missing_parts_map_to_zero(self) -> None:
        keys = compute_entity_keys({"card1": None, "addr1": float("nan"), "card2": 404})
        assert keys["_entity_card"] == 0
        assert keys["_entity_card_full"] == 404

    def test_first_transaction_sees_no_history(self) -> None:
        store = VelocityStore(None)
        features = store.features_for({"card1": 1, "TransactionAmt": 10.0}, timestamp=1000)
        assert features["entity_card_txn_count_1h"] == 0.0
        assert features["entity_card_amt_sum_1h"] == 0.0

    def test_history_accumulates_across_calls(self) -> None:
        store = VelocityStore(None)
        record = {"card1": 42, "TransactionAmt": 10.0}
        store.features_for(record, timestamp=1000)
        store.features_for(record, timestamp=1060)
        third = store.features_for(record, timestamp=1120)
        assert third["entity_card_txn_count_1h"] == 2.0
        assert third["entity_card_amt_sum_1h"] == pytest.approx(20.0)
        assert third["entity_card_seconds_since_prev"] == 60.0

    def test_transactions_outside_the_window_are_excluded(self) -> None:
        store = VelocityStore(None)
        record = {"card1": 99, "TransactionAmt": 5.0}
        store.features_for(record, timestamp=0)
        later = store.features_for(record, timestamp=7200)  # 2 hours later
        assert later["entity_card_txn_count_1h"] == 0.0
        assert later["entity_card_txn_count_24h"] == 1.0

    def test_different_entities_are_isolated(self) -> None:
        store = VelocityStore(None)
        store.features_for({"card1": 1, "TransactionAmt": 1.0}, timestamp=100)
        other = store.features_for({"card1": 2, "TransactionAmt": 1.0}, timestamp=200)
        assert other["entity_card_txn_count_1h"] == 0.0


class TestModelUnavailable:
    def test_predict_returns_503_without_a_model(
        self, api_client, valid_transaction_payload
    ) -> None:
        from api import dependencies

        original = dependencies.state.artifact
        dependencies.state.artifact = None
        try:
            response = api_client.post("/predict", json=valid_transaction_payload)
            assert response.status_code == 503
            assert "not loaded" in response.json()["detail"]
        finally:
            dependencies.state.artifact = original
