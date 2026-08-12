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
    )
    validate_rtc_server_capability(
        metadata,
        rtc_requested=True,
        fixed_prefix_shape_required=True,
    )
    assert metadata["policy_name"] == "ur10e"
    assert metadata["openpi_server"]["rtc"]["fixed_prefix_shape"] is True


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
