"""EEG analysis tool implementations.

Tool set (10 tools):
  get_recording_info          — basic recording metadata
  get_channel_list            — channel names, types, regions
  compute_noise_metrics       — per-channel noise indicators
  compute_psd                 — power spectral density, alpha peak
  compute_band_power          — absolute/relative band power, ratios
  detect_transients           — candidate peak detection + morphology (not confirmed events)
  compute_temporal_features   — statistical and Hjorth features
  compute_asymmetry           — left-right hemispheric differences
  compute_windowed_features   — sliding-window batch feature time-series
  compute_channel_correlation — inter-channel Pearson correlation matrix

Design principles:
  - Tools only compute numerical metrics, never clinical categories.
  - Input is record_id + parameters, output is deterministic.
  - Channel selection is controlled by the model via the channels parameter.
"""

from eeg_agent_bench.tools.implementations.get_recording_info import GetRecordingInfoTool
from eeg_agent_bench.tools.implementations.get_channel_list import GetChannelListTool
from eeg_agent_bench.tools.implementations.compute_signal_quality import ComputeNoiseMetricsTool
from eeg_agent_bench.tools.implementations.compute_psd import ComputePSDTool
from eeg_agent_bench.tools.implementations.compute_band_power import ComputeBandPowerTool
from eeg_agent_bench.tools.implementations.detect_epileptiform_discharges import DetectTransientsTool
from eeg_agent_bench.tools.implementations.compute_temporal_features import ComputeTemporalFeaturesTool
from eeg_agent_bench.tools.implementations.compute_asymmetry import ComputeAsymmetryTool
from eeg_agent_bench.tools.implementations.compute_windowed_features import ComputeWindowedFeaturesTool
from eeg_agent_bench.tools.implementations.compute_channel_correlation import ComputeChannelCorrelationTool

ALL_TOOLS = [
    GetRecordingInfoTool,
    GetChannelListTool,
    ComputeNoiseMetricsTool,
    ComputePSDTool,
    ComputeBandPowerTool,
    DetectTransientsTool,
    ComputeTemporalFeaturesTool,
    ComputeAsymmetryTool,
    ComputeWindowedFeaturesTool,
    ComputeChannelCorrelationTool,
]
