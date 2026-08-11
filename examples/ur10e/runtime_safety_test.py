from __future__ import annotations

import pytest

from runtime_safety import ARM_PHRASE
from runtime_safety import request_execution_arm


def test_dry_run_does_not_prompt_or_arm() -> None:
    assert not request_execution_arm(
        False,
        input_fn=lambda _: pytest.fail("dry-run must not prompt"),
    )


def test_exact_phrase_arms_execution() -> None:
    assert request_execution_arm(True, input_fn=lambda _: ARM_PHRASE)


def test_wrong_phrase_rejects_execution() -> None:
    with pytest.raises(RuntimeError, match="no actions were sent"):
        request_execution_arm(True, input_fn=lambda _: "yes")
