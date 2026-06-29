# UI Demo Guide EN

## Start

Build UI:

```bash
cd src/demo/ui
npm install
npm run build
```

Start backend:

```bash
python -m src.demo.api_server
```

Open:

```text
http://127.0.0.1:8000/
```

If `src/demo/ui/dist` exists, FastAPI serves the React UI. Otherwise it falls back to the legacy static UI.

## Recommended Demo Flow

### 1. Select Model

Use:

- `Microset epoch05` for the thesis anchor;
- `Top500 epoch13` for broader local demo coverage.

When switching models, choose rebuild enrollment if waveform cache exists, or clear enrollment if you want a clean session.

### 2. Enroll GSC 17 Known

Use the GSC 17 known preset:

```text
yes, stop, happy, bird, dog, tree, marvin, four, learn, wow, sheila, zero, down, left, right, off, three
```

This matches the open-set 17/17 demo preset.

### 3. Single Detection

Upload a short audio file. Inspect:

- predicted keyword or unknown;
- L2 distance;
- threshold;
- margin;
- top-3 candidates;
- backend-confirmed policy settings.

### 4. Long Audio

Upload long audio and optionally labels/timing files. The UI shows summary metrics, expected/detected timelines, detection cards, and missed expected cards.

Read miss reasons before judging model accuracy:

- no overlap;
- threshold reject;
- guard reject;
- wrong prediction;
- outside enrollment;
- segmentation skip.

### 5. Open-Set

Use the 17 known / 17 unknown preset. Current recommended policy:

- Guard ON;
- Per-class OFF;
- accept margin 0.05.

Report balanced score, keyword accuracy, unknown reject accuracy, FAR, and false reject rate. This is demo-level sampled evaluation, not a replacement for GSC test100.

### 6. Reports

Use the Reports tab to export a Markdown session report and inspect artifact status.

## Demo Notes

- Restart the backend after building React if port 8000 is still serving the old UI.
- If a model profile is missing, inspect the local `server` artifact folder.
- If open-set words are skipped, verify local GSC setup.
- If long-audio accuracy is low, inspect timing overlap and missed expected cards.
