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
- Fixed the Stage 1 baseline on `main` and created `reproduction` from `main@20a8f73`.
- Implemented `UIElement`, cached UI-perceptor, semantic-scorer, and optional lazy Instructor-large embedding abstractions.
- Implemented Focus, Shift, and Scatter with paper-specified top fractions, area cap, and IoU cap.
- Implemented the three paper reward terms and final 0.4/0.4/0.2 combination.
- Implemented deterministic MCTS selection, expansion, reward evaluation, and backpropagation with `N=8`, `H=3`, and `c=1`.
- Implemented Best Region cropping, explicit crop-local coordinate labeling, and restoration to original screenshot pixels.
- Implemented full MCTS trace/result persistence and search visualization.
- Passed 64 offline unit/integration tests on Python 3.11.5.
- Kept dataset files, outputs, checkpoints, private tasks, `.env`, and paper PDFs out of Git.

## Working

- Stage 2 core code and bounded proxy demos are complete on `reproduction`.
- Paper-aligned OmniParser V2 + Instructor-large + real grounding-model integration remains pending compatible model resources.

## Failed / Blocked

- No blocker for the model-independent DRS core.
- Real grounding inference has not been run: no valid DeepSeek key was read from `.env`, and no UGround inference server is available.
- Local UGround loading was deliberately not attempted on this 8 GB Apple M2 host. No checkpoint was downloaded.
- No paid DeepSeek API request was launched automatically.
- OmniParser V2 and Instructor-large were not downloaded or executed on the 8 GB Apple M2 host.
- Public CVF/arXiv searches and the arXiv source archive contain no accessible DRS-GUI implementation or supplementary parameter file. The source package repeats the main-paper specification but does not provide `lambda`, `tau`, Focus thresholds, or the prefix template.
- Four PNG files are present in the official snapshot but are not referenced by any annotation; they are retained as upstream data rather than deleted.

## Experiment Results

- No benchmark results yet.
- Dataset integrity result: 1,581/1,581 referenced images present; 10/10 evenly spaced real images opened with dimensions matching annotations.
- Stage 1 checkpoint result: 36/36 offline tests passed at the baseline commit.
- DRS test result: 64/64 offline tests passed.
- Six real ScreenSpot-Pro screenshots were used for search-only demos with `tesseract_ocr_proxy_non_paper` elements and `token_overlap_proxy_non_paper` relevance. No grounding model was called.
- Post-search GT-center diagnostic, with all cases retained:
  - `fruitloops_windows_8`: false
  - `quartus_windows_19`: false
  - `word_macos_83`: false
  - `eviews_windows_3`: true
  - `eviews_windows_6`: true
  - `eviews_windows_21`: false
- The 2/6 diagnostic is not benchmark accuracy. It demonstrates that the core runs on real screenshots and also shows that OCR/token overlap is not a valid substitute for OmniParser V2 + Instructor-large.
- The integrity and test results above are not grounding accuracy. Synthetic/mock unit-test outputs are never reported as benchmark results.

## Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/download_dataset.py --output data/ScreenSpot-Pro --verify-samples 10 --max-workers 16
python -m pytest -q
python scripts/run_drs_search_demo.py --index 232 --elements tasks/drs_demo_cache/eviews_windows_3.json
python scripts/run_drs_single.py --backend uground --index 232 --elements tasks/drs_demo_cache/eviews_windows_3.json
```

## Next Step

1. Obtain OmniParser V2 output and Instructor-large relevance for the same small fixed sample set; do not use GT to construct elements or scores.
2. Run one or two samples with a reachable UGround-V1-2B server and persist both baseline and DRS results.
3. Inspect search traces and crop/global coordinate overlays before increasing to 20 samples.
4. Ask the authors for the unspecified values and behaviors listed in `REPRODUCTION_GAPS.md`.
