# EEGAgentBench Frozen Evaluation Signals

This directory contains the signal files referenced by the 1,072-instance
EEGAgentBench evaluation set.

- T1 has 38 knowledge-QA instances and does not use signal files.
- T2--T6 contain 1,034 instances and reference 1,034 unique signal files.
- The copied files occupy approximately 23 GB (24,244,274,188 bytes).
- `../benchmark/runtime/` maps evaluation instances to release-relative signal
  paths without exposing those paths to the model.
- `../benchmark/manifests/files.jsonl` records every signal file's expected
  path and size for download verification.

## License and Attribution

These signals are redistributed under each source's own upstream terms; the
files retain their source-specific licenses and attribution requirements. In
particular, the Bonn set (T3) is provided by its authors for research and
education only, with commercial and military use prohibited. See the dataset
card and `LICENSES/README.md` in the Hugging Face dataset release before reuse.
