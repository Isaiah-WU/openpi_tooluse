from __future__ import annotations

import pytest

from openpi_client.server_capabilities import add_rtc_server_capability
from openpi_client.server_capabilities import validate_rtc_server_capability


def test_baseline_client_accepts_legacy_metadata() -> None:
    validate_rtc_server_capability({}, rtc_requested=False)


def test_rtc_client_accepts_enabled_server() -> None:
    metadata = add_rtc_server_capability(
        {"policy_name": "ur10e"},
        rtc_enabled=True,
        execution_horizon=10,
        prefix_attention_schedule="exp",
        max_guidance_weight=10.0,
        warmup_complete=True,
        warmup_inferences=2,
    )
    validate_rtc_server_capability(
        metadata,
        rtc_requested=True,
        fixed_prefix_shape_required=True,
        warmup_complete_required=True,
    )
    assert metadata["policy_name"] == "ur10e"
    assert metadata["openpi_server"]["rtc"]["fixed_prefix_shape"] is True
    assert metadata["openpi_server"]["rtc"]["warmup_complete"] is True
    assert metadata["openpi_server"]["rtc"]["warmup_inferences"] == 2


def test_warmup_aware_client_rejects_server_without_completed_warmup() -> None:
    metadata = add_rtc_server_capability(
        {},
        rtc_enabled=True,
        execution_horizon=10,
        prefix_attention_schedule="exp",
        max_guidance_weight=10.0,
    )

    with pytest.raises(RuntimeError, match="completed server warm-up"):
        validate_rtc_server_capability(
            metadata,
            rtc_requested=True,
            fixed_prefix_shape_required=True,
            warmup_complete_required=True,
        )


def test_server_rejects_inconsistent_completed_warmup_metadata() -> None:
    with pytest.raises(ValueError, match="at least two"):
        add_rtc_server_capability(
            {},
            rtc_enabled=True,
            execution_horizon=10,
            prefix_attention_schedule="exp",
            max_guidance_weight=10.0,
            warmup_complete=True,
            warmup_inferences=1,
        )


def test_fixed_prefix_client_rejects_pre_feature_rtc_server() -> None:
    metadata = add_rtc_server_capability(
        {},
        rtc_enabled=True,
        execution_horizon=10,
        prefix_attention_schedule="exp",
        max_guidance_weight=10.0,
    )
    del metadata["openpi_server"]["rtc"]["fixed_prefix_shape"]

    with pytest.raises(RuntimeError, match="fixed-prefix-shape"):
        validate_rtc_server_capability(
            metadata,
            rtc_requested=True,
            fixed_prefix_shape_required=True,
        )


def test_legacy_rtc_client_accepts_server_without_fixed_prefix_feature() -> None:
    metadata = add_rtc_server_capability(
        {},
        rtc_enabled=True,
        execution_horizon=10,
        prefix_attention_schedule="exp",
        max_guidance_weight=10.0,
    )
    del metadata["openpi_server"]["rtc"]["fixed_prefix_shape"]

    validate_rtc_server_capability(metadata, rtc_requested=True)


def test_rtc_client_rejects_disabled_server() -> None:
    metadata = add_rtc_server_capability(
        {},
        rtc_enabled=False,
        execution_horizon=10,
        prefix_attention_schedule="exp",
        max_guidance_weight=10.0,
    )
    with pytest.raises(RuntimeError, match="not enabled"):
        validate_rtc_server_capability(metadata, rtc_requested=True)


def test_rtc_client_rejects_legacy_server() -> None:
    with pytest.raises(RuntimeError, match="protocol_version"):
        validate_rtc_server_capability({}, rtc_requested=True)
