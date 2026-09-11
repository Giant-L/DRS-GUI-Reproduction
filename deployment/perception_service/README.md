# GPU Perception Service

This private service runs the paper-aligned perception path:

```text
Screenshot -> OmniParser V2 -> UI elements -> Instructor-large -> relevance
```

It never accepts or reads ground-truth boxes. Bind it to `127.0.0.1` and reach it
through an SSH tunnel; do not expose the port publicly.

## GPU host setup

Use the provider's persistent data disk for all repositories and model caches.
The following configuration was verified on one AutoDL RTX 4090 using its
PyTorch 2.5.1 / Python 3.12 / CUDA 12.4 image. Put repositories and weights on
the persistent `/root/autodl-tmp` disk:

```bash
git clone --branch reproduction https://github.com/Giant-L/DRS-GUI-Reproduction.git
git clone https://github.com/microsoft/OmniParser.git

cd OmniParser
git checkout 354021201345a96178360b28733573e27269f2de
git apply ../DRS-GUI-Reproduction/patches/omniparser-lazy-paddleocr.patch
git apply ../DRS-GUI-Reproduction/patches/omniparser-local-processor.patch

cd ../DRS-GUI-Reproduction
python -m pip install -e .
python -m pip install -r deployment/perception_service/requirements.txt
```

The deployed smoke test used the `icon_detect/model.pt` and
`icon_caption_florence` files mirrored at
`AI-ModelScope/OmniParser-v2.0`. The paper names OmniParser V2 but does not pin
its detector checkpoint; this checkpoint choice is therefore recorded in
`REPRODUCTION_GAPS.md`. Instructor-large is downloaded from the original
`hkunlp/instructor-large` Hugging Face repository. Set explicit local paths so
runtime inference is offline:

```bash
# OmniParser V2 mirror used by the recorded smoke test.
python -m pip install "modelscope==1.40.0"
modelscope download AI-ModelScope/OmniParser-v2.0 \
  icon_detect/model.pt icon_detect/model.yaml icon_detect/train_args.yaml \
  icon_caption/config.json icon_caption/generation_config.json \
  icon_caption/model.safetensors --local-dir /root/autodl-tmp/OmniParser/weights
mv /root/autodl-tmp/OmniParser/weights/icon_caption \
  /root/autodl-tmp/OmniParser/weights/icon_caption_florence

# AutoDL's academic proxy was required for direct Hugging Face access.
source /etc/network_turbo
hf download hkunlp/instructor-large \
  config.json pytorch_model.bin special_tokens_map.json spiece.model \
  tokenizer.json tokenizer_config.json \
  --local-dir /root/autodl-tmp/models/instructor-large
```

Start the private service from the reproduction repository:

```bash
export OMNIPARSER_ROOT=/root/autodl-tmp/OmniParser
export OMNIPARSER_DETECTOR_PATH=/root/autodl-tmp/OmniParser/weights/icon_detect/model.pt
export OMNIPARSER_CAPTION_PATH=/root/autodl-tmp/OmniParser/weights/icon_caption_florence
export OMNIPARSER_PROCESSOR_PATH=/root/autodl-tmp/models/florence-2-base
export INSTRUCTOR_MODEL_PATH=/root/autodl-tmp/models/instructor-large
bash deployment/perception_service/start_autodl.sh
```

The launcher binds only to `127.0.0.1`, writes its PID and log under
`/root/autodl-tmp`, and defaults Transformers to offline mode. An optional
`PERCEPTION_API_KEY` may be supplied before starting.

## Private local access

On the local Mac, create the tunnel using the provider's actual SSH host and port:

```bash
ssh -N -L 8010:127.0.0.1:8010 -p <ssh-port> <user>@<host>
```

Configure the ignored local `.env`:

```dotenv
PERCEPTION_API_BASE=http://127.0.0.1:8010
PERCEPTION_API_KEY=<same-random-token>
PERCEPTION_TIMEOUT_SECONDS=300
```

Cache exactly one real sample locally:

```bash
python scripts/cache_remote_perception.py --index 232
```

The resulting ignored JSON can be passed directly to `run_drs_search_demo.py` or
`run_drs_single.py`.
