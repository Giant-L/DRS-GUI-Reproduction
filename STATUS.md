# Status

Last updated: 2026-09-11 (Asia/Shanghai)

## Completed

- Read DRS-GUI Methodology 3.1-3.4 and Experimental Settings from the supplied PDF.
- Confirmed the Stage 1 boundary: full-screen grounding only, with no DRS components.
- Implemented the ScreenSpot-Pro sample abstraction and loader, including original image dimensions, metadata, and absolute-pixel GT boxes.
- Implemented the original-pixel evaluator and explicit conversions for normalized, resized, and crop-local coordinates.
- Implemented a common `GroundingModel` interface, a DeepSeek Vision smoke adapter, and an OpenAI-compatible UGround-V1-2B adapter.
- Implemented bounded single-sample and batch CLIs plus JSON/JSONL result persistence. Failed samples remain in the denominator.
- Downloaded the full official `likaixin/ScreenSpot-Pro` snapshot to the ignored `data/` directory.
- Resolved dataset revision: `210e78d3844251110bff86c95835ebd37a6930fa`.
- Parsed all 1,581 annotations and confirmed that all 1,581 referenced screenshots exist.
- Opened and dimension-checked ten evenly spaced real samples spanning ten application/platform sets.
- Passed 36 offline unit/integration tests on Python 3.11.5.
- Kept dataset files, outputs, checkpoints, private tasks, `.env`, and paper PDFs out of Git.

## Working

- Final Stage 1 documentation, clean-tree verification, and `main` push.

## Failed / Blocked

- No blocker for the offline Stage 1 framework.
- Real grounding inference has not been run: no valid DeepSeek key was read from `.env`, and no UGround inference server is available.
- Local UGround loading was deliberately not attempted on this 8 GB Apple M2 host. No checkpoint was downloaded.
- No paid DeepSeek API request was launched automatically.
- Four PNG files are present in the official snapshot but are not referenced by any annotation; they are retained as upstream data rather than deleted.

## Experiment Results

- No benchmark results yet.
- Dataset integrity result: 1,581/1,581 referenced images present; 10/10 evenly spaced real images opened with dimensions matching annotations.
- Test result: 36/36 offline tests passed.
- The integrity and test results above are not grounding accuracy. Synthetic/mock unit-test outputs are never reported as benchmark results.

## Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/download_dataset.py --output data/ScreenSpot-Pro --verify-samples 10 --max-workers 16
python -m pytest -q
```

## Next Step

1. Put a newly issued DeepSeek key only in `.env`, or provide a reachable UGround server.
2. Run one or two explicitly authorized real-model smoke samples and inspect their persisted records.
3. Treat Stage 1 as fixed only after smoke inference is verified.
4. Create `reproduction` from `main` only when Stage 2 begins.
