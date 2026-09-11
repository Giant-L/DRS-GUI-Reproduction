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
- Verified the configured DeepSeek `deepseek-flash` endpoint with one real ScreenSpot-Pro full-screen grounding request.
- Implemented a private remote OmniParser V2 + Instructor-large GPU service contract and GT-free local cache client.
- Deployed OmniParser V2 and `hkunlp/instructor-large` on one rented RTX 4090 behind a localhost-only FastAPI service and private SSH tunnel.
- Verified simultaneous offline model loading on the 4090; Instructor-large returns finite 1,024-dimensional embeddings.
- Added explicit local model paths, a reproducible AutoDL launcher, pinned GPU-service dependencies, and tracked third-party compatibility patches.
- Passed 71 offline unit/integration tests on Python 3.11.5.
- Kept dataset files, outputs, checkpoints, private tasks, `.env`, and paper PDFs out of Git.

## Working

- Stage 2 core code and bounded proxy demos are complete on `reproduction`.
- Two disjoint fixed 20-sample paired DeepSeek smoke rounds are complete (40 samples total), with all model and search failures retained.

## Failed / Blocked

- No blocker for the model-independent DRS core.
- No UGround inference server is currently available.
- Local UGround loading was deliberately not attempted on this 8 GB Apple M2 host. No checkpoint was downloaded.
- OmniParser V2 and Instructor-large remain intentionally remote rather than installed on the 8 GB Apple M2 host.
- Direct Hugging Face and hf-mirror Xet downloads timed out from AutoDL. The official AutoDL academic proxy was used for Instructor-large; ModelScope's `AI-ModelScope/OmniParser-v2.0` mirror was used for OmniParser V2 weights.
- Public CVF/arXiv searches and the arXiv source archive contain no accessible DRS-GUI implementation or supplementary parameter file. The source package repeats the main-paper specification but does not provide `lambda`, `tau`, Focus thresholds, or the prefix template.
- Four PNG files are present in the official snapshot but are not referenced by any annotation; they are retained as upstream data rather than deleted.

## Experiment Results

- No benchmark results yet.
- Dataset integrity result: 1,581/1,581 referenced images present; 10/10 evenly spaced real images opened with dimensions matching annotations.
- Stage 1 checkpoint result: 36/36 offline tests passed at the baseline commit.
- DRS test result: 71/71 offline tests passed.
- Real DeepSeek baseline smoke test on `eviews_windows_3`: prediction `(1305, 528)` lies inside GT bbox `[1275, 521, 1340, 549]`; latency 2.93 seconds; 1,084 total tokens. This is one real sample, not benchmark accuracy.
- The first request returned a truncated JSON fragment because DeepSeek thinking mode consumed the 64-token output allowance. The adapter now explicitly disables thinking; the same-sample regression request completed successfully, and the behavior is locked by an offline test.
- Real paper-component-aligned perception smoke test on `eviews_windows_3`: OmniParser V2 produced 67 UI elements; Instructor-large scored them without GT input; parser latency 1.40 seconds and scoring latency 0.12 seconds on the remote 4090.
- Real paired result on `eviews_windows_3` (one sample, not benchmark accuracy):
  - full-screen DeepSeek baseline: `(1305, 528)`, correct, 1,084 tokens, 2.93-second grounding call;
  - OmniParser V2 + Instructor-large + DRS + the same DeepSeek backend: best region `[1274.59, 517.58, 1344.57, 549.69]`, final global point `(1309, 533)`, correct, 286 tokens, 1.23-second grounding call;
  - perception took another 1.52 seconds and must be included for end-to-end latency comparisons. The one paired sample shows a 73.6% grounding-token reduction, not an accuracy improvement.
- Remote deployment regression: `AutoModel` incorrectly constructed a full T5 encoder-decoder from the encoder-only Instructor checkpoint. It was replaced with `T5EncoderModel`, eliminating random decoder initialization and the missing `decoder_input_ids` failure.
- Private-tunnel regression: Requests inherited a transparent proxy in the elevated execution environment and returned an empty HTTP 502 before reaching Uvicorn. The private perception client now disables environment proxies; a regression assertion locks this behavior.
- Fixed 20-sample paired real-data smoke comparison using the same `deepseek-flash` grounding backend:
  - baseline: 2/20 correct (10%);
  - DRS reproduction: 2/20 correct (10%); accuracy delta 0 percentage points;
  - paired outcomes: 1 both correct, 1 DRS-only correct, 1 baseline-only correct, 17 both wrong;
  - Best Region contained the GT center in 5/20 samples;
  - DRS failure decomposition: 15 search-region misses and 3 grounding failures despite a region containing GT;
  - grounding tokens: 21,503 baseline versus 9,130 DRS, a 57.5% reduction;
  - mean grounding-call latency: 2.96 seconds baseline versus 1.63 seconds after DRS cropping; DRS additionally required 5.78 seconds mean remote perception latency;
  - zero API/parse errors in either arm. This fixed, non-random subset is a smoke diagnostic, not ScreenSpot-Pro benchmark accuracy.
- Second disjoint fixed 20-sample paired smoke comparison:
  - baseline: 2/20 correct (10%);
  - DRS reproduction: 2/20 correct (10%); accuracy delta 0 percentage points;
  - paired outcomes: 2 DRS-only correct, 2 baseline-only correct, 16 both wrong;
  - Best Region contained the GT center in 6/20 samples; DRS was correct on 2/6 region hits;
  - grounding tokens: 21,543 baseline versus 9,460 DRS, a 56.1% reduction;
  - mean grounding-call latency: 5.77 seconds baseline versus 1.82 seconds after DRS cropping; DRS additionally required 7.21 seconds mean remote perception latency;
  - baseline had zero errors; DRS retained three failures (two empty model responses and one crop-local coordinate outside the crop). Token reduction is partly affected by the two empty responses, which report no usage.
- Combined fixed 40-sample smoke diagnostic:
  - baseline: 4/40 correct (10%); DRS reproduction: 4/40 correct (10%); accuracy delta 0 percentage points;
  - paired outcomes: 1 both correct, 3 DRS-only correct, 3 baseline-only correct, 33 both wrong;
  - Best Region recall: 11/40 (27.5%); DRS accuracy conditioned on a region hit: 4/11 (36.4%);
  - DRS failures: 29 search-region misses and 7 grounding failures inside a region containing GT;
  - mean Best Region area was 10.5% of the screenshot, an 89.5% mean image-area reduction;
  - grounding tokens: 43,046 baseline versus 18,590 DRS, a 56.8% reduction;
  - mean grounding-call latency: 4.36 seconds baseline versus 1.72 seconds after cropping, plus 6.50 seconds mean remote perception latency;
  - this is a fixed, non-random, DeepSeek-based smoke subset—not paper-aligned UGround benchmark accuracy and not evidence of an accuracy gain.
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
python scripts/cache_remote_perception.py --index 232
bash deployment/perception_service/start_autodl.sh
```

## Next Step

1. Inspect the 29 search-region misses from the combined fixed 40-sample comparison before increasing the sample count.
2. Run one or two samples with a reachable UGround-V1-2B server and persist both baseline and DRS results.
3. Compare the exact OmniParser checkpoint and unspecified search parameters with the authors' environment.
4. Only after the region-recall gap is understood, predeclare a larger 100-sample subset.
