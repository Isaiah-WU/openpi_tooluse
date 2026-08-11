"""Compatibility imports for the packaged UR10e execution gate."""

from openpi_client.rtc_safety import ARM_PHRASE
from openpi_client.rtc_safety import request_execution_arm

__all__ = ["ARM_PHRASE", "request_execution_arm"]
