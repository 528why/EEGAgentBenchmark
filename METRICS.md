# Official Metrics

The implementation in `eeg_agent_bench/reporting/official.py` is the sole
source of corpus-level benchmark scores. It recomputes every metric from the
saved prediction and independent answer key. Values previously cached in a
run's `score` or `score_details` fields are not used.

## Completeness

An official task summary requires exactly the frozen scenario set: T1 38, T2
300, T3 188, T4 69, T5 280, and T6 197. Missing, extra, duplicate, or
non-completed run rows cause scoring to fail instead of silently changing a
denominator. A completed row with an empty or malformed model prediction stays
in the denominator and is scored according to the task rules below.

## T1 Knowledge QA

T1 reports Accuracy over all 38 questions. A recoverable A/B/C/D option is
matched to the reference option. Empty or unrecoverable answers are incorrect.

## T2--T4 Classification

T2--T4 report Accuracy and corpus-level Macro-F1. Their fixed reference class
sets are:

- T2: clean, ocular_contaminated, muscle_contaminated
- T3: normal, epileptic
- T4: AD, FTD, HC

For each reference class, F1 is `2TP / (2TP + FP + FN)`. Macro-F1 is the
unweighted mean over only the task's fixed reference classes. Empty outputs and
arbitrary labels such as `artifact`, `poor quality`, or
`non_epileptic_non_normal` map to `__invalid__`: they count as errors and false
negatives for their gold class, remain in N, and never create extra averaged
classes. T2 additionally normalizes documented aliases such as EOG to
ocular_contaminated and EMG to muscle_contaminated.

## T5 Seizure Detection

The primary T5 metric is subject-averaged SzCORE-style Event-F1. For each of 280
records, event boundaries are clipped to the recording, quantized at 10 Hz,
events separated by less than 90 s are merged, and events longer than 300 s are
split. Reference events are extended 30 s before onset and 60 s after offset.
A positive-duration overlap is a match. Record TP, FP, FN and duration are
summed within each CHB-MIT subject; Event-F1 is then computed per subject and
averaged equally across 24 subjects.

Dice-S is the mean direct time-axis Dice overlap over the 140 seizure-positive
records only. It does not use tolerance windows, event merging, or splitting.
The summary also reports subject-averaged sensitivity, precision and FP/24h,
plus record-level miss and over-alarm rates.

## T6 Sleep Staging

Each predicted run-length encoding is expanded to the recording's complete
30-second epoch sequence. Gaps and missing tails are filled with W, later
segments overwrite overlaps, out-of-range tails are truncated, and malformed
segments are skipped. Empty predictions therefore become an all-W sequence and
are not excluded. Per-record Macro-F1 over W/N1/N2/N3/REM and Cohen's kappa are
computed, then averaged equally over all 197 recordings.

## Overall

Overall is the unweighted mean of six task scores: T1 Accuracy, T2 Macro-F1,
T3 Macro-F1, T4 Macro-F1, T5 subject-averaged Event-F1, and T6 Macro-F1.
