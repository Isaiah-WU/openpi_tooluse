"""Compatibility entry point for packaged RTC calibration."""

from openpi_client.rtc_calibration import RTCRecommendation
from openpi_client.rtc_calibration import RTCDelayTracker
from openpi_client.rtc_calibration import load_baseline_policy_delays
from openpi_client.rtc_calibration import main
from openpi_client.rtc_calibration import recommend_rtc_parameters
from openpi_client.rtc_calibration import validate_rtc_runtime_parameters

__all__ = [
    "RTCRecommendation",
    "RTCDelayTracker",
    "load_baseline_policy_delays",
    "recommend_rtc_parameters",
    "validate_rtc_runtime_parameters",
]

if __name__ == "__main__":
    main()
