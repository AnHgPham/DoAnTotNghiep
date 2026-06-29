# Troubleshooting EN

## Git Not In PATH

Symptom: Git commands fail.

Cause: Git is not installed or PATH was not loaded.

Fix: install Git for Windows and reopen the terminal.

## Colab Reset

Symptom: `/content` dataset disappears and training stops.

Cause: runtime reset or timeout.

Fix: rerun dataset setup and resume from Drive checkpoint.

Prevention: save every epoch, save latest every epoch, and package artifacts to Drive.

## Colab Units Exhausted

Symptom: run stops mid-training and GPU is unavailable.

Fix: use the latest saved epoch checkpoint, for example the current Top500 `epoch_13.pt`, and continue later.

## Missing Checkpoint

Symptom: model card shows missing.

Cause: checkpoint was not downloaded or path changed.

Fix: copy the checkpoint into the expected `server/final_kws_artifacts_package/checkpoints/...` path or update the model profile.

## No Unknown GSC Audio Found

Symptom: Open-set test skips unknown words.

Cause: local GSC folder is missing or incomplete.

Fix: set up GSC v2 under `data/gsc_v2`.

## No Enrolled Keywords

Symptom: detection fails or returns only unknown.

Fix: enroll a preset such as GSC 17 known.

## Label Count Mismatch

Symptom: long-audio expected count differs from detection count.

Cause: VAD/energy segmentation skipped or split words differently.

Fix: provide timing JSON and inspect missed expected cards.

## Threshold Miss

Symptom: correct top-1 candidate is rejected because distance is above threshold.

Fix: calibrate threshold. Avoid raising threshold blindly because it increases false accepts.

## Guard Miss

Symptom: distance is acceptable but top-1/top-2 margin is too small.

Fix: use calibration. Turning the guard off is acceptable for keyword-only demos but usually hurts open-set balance.

## Low Unknown Rejection

Symptom: high false accept rate in open-set.

Fix: use Guard ON, Per-class OFF, accept margin 0.05 as the current default demo policy, then run calibration.

## Partial Top500 Cache

Symptom: fewer than expected word folders or low coverage.

Fix: rebuild the session-first dataset and avoid reusing partial Drive cache.

## Worker 100 Freezes

Symptom: PyTorch DataLoader warning and slow/frozen training.

Fix: reduce workers to 12 or 20. Worker 100 is only for the Top500 Colab runbook, not a global default.

## Drive Copy Too Slow

Symptom: setup stalls while copying WAV cache to Drive.

Fix: keep dataset in the Colab session and save only checkpoints/results/packages to Drive.
