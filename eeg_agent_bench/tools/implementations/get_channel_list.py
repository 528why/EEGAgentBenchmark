"""Tool: get_channel_list — channel names, types, regions, hemisphere."""

from __future__ import annotations

from typing import Any

from eeg_agent_bench.tools.base import BaseTool
from eeg_agent_bench.types import ToolSpec


class GetChannelListTool(BaseTool):

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="get_channel_list",
            version="0.1.0",
            description=(
                "Return the list of EEG channels with their type (EEG/EOG/EMG/etc.), "
                "hemisphere (L/R/M), brain region (frontal/temporal/central/parietal/"
                "occipital), and bad-channel flag. "
                "Applies to any recording (incl. single-channel); takes only "
                "``record_id``. Output is bounded (one row per channel)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "record_id": {
                        "type": "string",
                        "description": "EEG record identifier.",
                    },
                },
                "required": ["record_id"],
            },
        )

    def execute(self, arguments: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        from eeg_agent_bench.tools.implementations._data_utils import (
            load_raw, HEMISPHERE_MAP, REGION_MAP, attach_response_meta,
        )

        record_id = arguments["record_id"]
        raw = load_raw(record_id, context)
        bads = set(raw.info.get("bads", []))

        channels = []
        for ch_name in raw.ch_names:
            ch_info = raw.info["chs"][raw.ch_names.index(ch_name)]
            # MNE channel type
            import mne
            ch_type = mne.channel_type(raw.info, raw.ch_names.index(ch_name))

            # Normalise name for lookup (strip EEG prefix, spaces)
            clean = ch_name.replace("EEG ", "").replace("eeg ", "").strip()
            # Handle bipolar montage names like "F7-T7"
            base = clean.split("-")[0] if "-" in clean else clean

            hemisphere = HEMISPHERE_MAP.get(base, "unknown")
            region = REGION_MAP.get(base, "unknown")

            channels.append({
                "name": ch_name,
                "type": ch_type,
                "hemisphere": hemisphere,
                "region": region,
                "marked_bad_in_file": ch_name in bads,
            })

        return attach_response_meta(
            {
                "record_id": record_id,
                "n_channels": len(channels),
                "channels": channels,
                "file_header_bads": sorted(bads),
            },
            n_rows=len(channels),
        )
