# DRS-GUI Reproduction

A research-oriented paper reimplementation of **DRS-GUI: Dynamic Region Search for Training-Free GUI Grounding** (CVPR 2026).

This repository is deliberately split by Git branch:

- `main`: a reproducible full-screen GUI-grounding baseline.
- `reproduction`: DRS-GUI search components, created only after the baseline is stable.

The current `reproduction` branch inherits the tested Stage 1 baseline and adds a paper-guided implementation of the DRS-GUI core. The `main` branch remains the baseline-only comparison point.

## Stage 1 pipeline

```text
Instruction + Full Screenshot
             |
             v
       Grounding Model
       (UGround / DeepSeek smoke backend)
             |
             v
   Original-pixel point (x, y)
             |
             v
          Evaluator
```

The evaluation rule follows ScreenSpot-Pro: a prediction is correct when the predicted point lies inside or on the boundary of the ground-truth bounding box.

## DRS-GUI pipeline (`reproduction`)

```text
Instruction + Full Screenshot
             |
             v
 UI Element Perceptor + Semantic Relevance
             |
             v
      Focus / Shift / Scatter
             |
             v
      MCTS Action Planner
             |
             v
          Best Region
             |
             v
    Same Grounding Model on Crop
             |
             v
 Crop-local Point -> Original-pixel Point
             |
             v
          Evaluator
```

The implemented core follows Methodology 3.1-3.4 and the paper's experimental settings:

- OmniParser-style elements contain global-pixel `bbox`, `description`, and `interactive` fields.
- Instruction and element descriptions share a domain prefix, final-hidden-state mean pooling, and cosine similarity. The optional Instructor backend is lazy and not part of the default lightweight installation.
- Focus keeps the top 15% relevant visible elements.
- Scatter considers the top 10% external elements and never exceeds 1.5x the current region area.
- Shift considers the top 15% external anchors and enforces IoU <= 0.3.
- MCTS uses rollout budget 8, post-initial-Focus depth 3, and UCT constant 1.
- Reward implements interaction-weighted relevance, UI coverage consistency, and semantic concentration with weights 0.4/0.4/0.2.

The paper does not specify several code-level choices. Every selected default is centralized in `DRSConfig`, serialized with search results, and documented in `REPRODUCTION_GAPS.md` rather than being presented as an official value.

## Repository layout

```text
.
├── README.md
├── STATUS.md
├── REPRODUCTION_GAPS.md
├── .env.example
├── requirements.txt
├── pyproject.toml
├── src/drsgui/
│   ├── baseline.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluator.py
│   ├── drs/
│   │   ├── actions.py
│   │   ├── config.py
│   │   ├── geometry.py
│   │   ├── mcts.py
│   │   ├── perception.py
│   │   ├── pipeline.py
│   │   ├── reward.py
│   │   ├── search.py
│   │   ├── types.py
│   │   └── visualization.py
│   └── models/
│       ├── base.py
│       ├── deepseek.py
│       └── uground.py
├── scripts/
│   ├── download_dataset.py
│   ├── run_drs_search_demo.py
│   ├── run_drs_single.py
│   ├── run_single.py
│   └── eval_baseline.py
└── tests/
```

Dataset files, outputs, private tasks, model checkpoints, caches, `.env`, and paper PDFs are ignored by Git.

## Setup

Python 3.10 or newer is required. Python 3.11 is recommended for later model compatibility.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
cp .env.example .env
```

Never commit `.env` or API keys.

## Download ScreenSpot-Pro

The dataset is downloaded from the official Hugging Face release, [`likaixin/ScreenSpot-Pro`](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro). The full snapshot is approximately 3.4 GB and contains 1,581 grounding tasks.

```bash
python scripts/download_dataset.py \
  --output data/ScreenSpot-Pro \
  --verify-samples 10
```

The post-download check loads the official annotations and opens ten real screenshots to verify their dimensions. It performs no model inference and reports no accuracy.

For reproducibility, the script resolves the requested branch/tag to an immutable Hugging Face commit and writes the ignored local file `drsgui_download_metadata.json`. The Stage 1 snapshot verified on 2026-09-11 resolved to `210e78d3844251110bff86c95835ebd37a6930fa`.

The public dataset can be downloaded without credentials. If Hugging Face rate limits make it slow, set an optional `HF_TOKEN` in `.env`; the token is never committed.

Expected local layout:

```text
data/ScreenSpot-Pro/
├── annotations/*.json
└── images/<application_platform>/*.png
```

The loader treats each annotation's `img_filename` as the source of truth. It exposes the instruction, original screenshot, absolute-pixel GT bbox, image dimensions, UI type, application, platform, and category.

## Coordinates

The repository-wide contract is strict:

- A backend returns pixels relative to the exact image it receives.
- In Stage 1 that image is the full screenshot, so the result is already in original pixels.
- In DRS the backend receives the Best Region crop; the pipeline explicitly labels its output `crop_local_pixels`, then restores and labels `original_image_pixels` before evaluation.
- The evaluator never guesses the model coordinate system.
- UGround's official `[0, 1000)` output is converted inside `UGroundModel`.
- Invalid, ambiguous, non-finite, or out-of-image coordinates produce explicit errors.
- Failed samples stay in the evaluation denominator.

The official dataset contains one GT box (`inventor_windows_60`) that extends one pixel above the screenshot. The loader preserves this annotation exactly and accepts partially clipped boxes, while rejecting boxes that do not intersect the image.

## Grounding backends

### UGround-V1-2B (paper-aligned baseline)

`UGroundModel` connects to an OpenAI-compatible vLLM or SGLang server. It uses the official model id `osunlp/UGround-V1-2B`, temperature zero, and the official normalized-coordinate convention.

On a CUDA host, start a compatible server separately, for example:

```bash
vllm serve osunlp/UGround-V1-2B \
  --served-model-name osunlp/UGround-V1-2B \
  --api-key EMPTY
```

Then configure:

```dotenv
UGROUND_API_BASE=http://localhost:8000/v1
UGROUND_API_KEY=EMPTY
UGROUND_MODEL=osunlp/UGround-V1-2B
```

No UGround checkpoint is downloaded by the dataset script.

The backend interface is intentionally independent of UGround, so a Qwen2.5-VL adapter can be added later without changing the dataset, evaluator, or result format.

### DeepSeek Vision (smoke test only)

DeepSeek Vision is provided only to check the end-to-end transport and persistence pipeline. It is **not** a paper-aligned baseline and its result must not be compared to the paper as UGround/Qwen performance.

```dotenv
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash-vision-exp
```

API calls may incur cost. Run them explicitly:

```bash
python scripts/run_single.py --backend deepseek --index 0
```

## Run a bounded baseline

The evaluator requires an explicit limit to prevent an accidental full benchmark:

```bash
python scripts/eval_baseline.py \
  --backend uground \
  --limit 10 \
  --offset 0
```

Each run creates an ignored directory under `outputs/` containing:

- `config.json`: timestamp, model, backend, dataset, sample count, and coordinate contract;
- `predictions.jsonl`: prediction, GT bbox, correctness, latency, and explicit error for every sample;
- `summary.json`: sample count, correct count, error count, and accuracy.

Results produced by synthetic unit-test models are test artifacts, not benchmark results. Real results are never fabricated or silently filtered.

## Run the DRS core with cached elements

The paper core is deliberately testable without OmniParser, Instructor-large, a GPU, or network access. A cache must declare its provenance and use original screenshot pixel coordinates:

```json
{
  "coordinate_space": "original_image_pixels",
  "sample_id": "sample_id",
  "source": "omniparser_v2_or_explicit_proxy_name",
  "relevance_source": "instructor_large_or_explicit_proxy_name",
  "elements": [
    {
      "element_id": "element-0",
      "bbox": [10, 20, 80, 50],
      "description": "Save",
      "interactive": true,
      "relevance": 0.91
    }
  ]
}
```

Run search and visualization for exactly one ScreenSpot-Pro sample:

```bash
python scripts/run_drs_search_demo.py \
  --index 232 \
  --elements tasks/drs_demo_cache/eviews_windows_3.json
```

This command performs region search only. It stores the complete MCTS trace, component rewards, selected element IDs, assumptions, and visualization. GT is added only after search as a diagnostic overlay and is never passed to the perceptor, actions, reward, or planner.

With a configured backend, run one crop-grounding sample explicitly:

```bash
python scripts/run_drs_single.py \
  --backend uground \
  --index 232 \
  --elements tasks/drs_demo_cache/eviews_windows_3.json
```

The single-sample runner persists model failures and exits non-zero; it does not silently discard them. A fair benchmark requires OmniParser V2 and Instructor-large caches or live outputs, plus the same grounding backend/configuration for baseline and DRS.

## Remote paper-aligned perception

`deployment/perception_service` provides a private GPU service that composes the
official OmniParser V2 output with `hkunlp/instructor-large` relevance scoring.
It converts OmniParser's normalized boxes to original screenshot pixels and never
accepts ground truth. Keep the service bound to the GPU host's `127.0.0.1:8010`
and access it through an SSH tunnel.

After the tunnel and ignored `.env` variables are configured, cache one sample:

```bash
python scripts/cache_remote_perception.py --index 232
```

Then pass `tasks/perception_cache/eviews_windows_3.json` to the existing DRS
search or crop-grounding command. Full GPU installation and tunnel instructions
are in `deployment/perception_service/README.md`.

## Tests

The default suite is offline and requires neither a GPU nor an API key:

```bash
pytest
```

It covers dataset parsing, image-size validation, bbox evaluation, coordinate conversions, invalid model responses, Focus, Shift, Scatter, all reward terms, MCTS, cached and remote perception contracts, semantic cosine scoring, visualization, crop grounding, and result persistence.

The current `reproduction` suite passes 71 offline tests. The downloaded snapshot has 1,581 annotated samples and 1,581 referenced screenshots. The real-screenshot search demos described in `STATUS.md` use clearly labeled non-paper OCR/relevance proxies and are not grounding results. A separate one-sample paper-component-aligned smoke result using remote OmniParser V2 and Instructor-large is documented there and must not be interpreted as benchmark accuracy.

## Project status

See [STATUS.md](STATUS.md) for verified progress and commands, and [REPRODUCTION_GAPS.md](REPRODUCTION_GAPS.md) for paper details that must not be presented as specified defaults.

## References

- Yichao Liu, Huawen Shen, Liu Yu, Shiyu Liu, Zeyu Chen, and Yu Zhou. *DRS-GUI: Dynamic Region Search for Training-Free GUI Grounding*. CVPR 2026.
- [ScreenSpot-Pro official repository](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)
- [ScreenSpot-Pro official dataset](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro)
- [UGround official repository](https://github.com/OSU-NLP-Group/UGround)
- [UGround-V1-2B model](https://huggingface.co/osunlp/UGround-V1-2B)
