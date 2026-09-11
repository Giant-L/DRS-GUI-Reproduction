# GPU Perception Service

This private service runs the paper-aligned perception path:

```text
Screenshot -> OmniParser V2 -> UI elements -> Instructor-large -> relevance
```

It never accepts or reads ground-truth boxes. Bind it to `127.0.0.1` and reach it
through an SSH tunnel; do not expose the port publicly.

## GPU host setup

Use the provider's persistent data disk for all repositories and model caches.
After identifying that mount with `df -h`, set `GPU_WORKDIR` to it and run:

```bash
git clone --branch reproduction https://github.com/Giant-L/DRS-GUI-Reproduction.git
git clone --depth 1 https://github.com/microsoft/OmniParser.git

cd OmniParser
python -m pip install -r requirements.txt
huggingface-cli download microsoft/OmniParser-v2.0 icon_detect_v3/model.pt \
  --revision refs/pr/37 --local-dir weights
for file in config.json generation_config.json model.safetensors; do
  huggingface-cli download microsoft/OmniParser-v2.0 "icon_caption/$file" \
    --local-dir weights
done
mv weights/icon_caption weights/icon_caption_florence

cd ../DRS-GUI-Reproduction
python -m pip install -e .
python -m pip install -r deployment/perception_service/requirements.txt
```

Start the private service from the reproduction repository:

```bash
export OMNIPARSER_ROOT=/absolute/path/to/OmniParser
export HF_HOME=/absolute/path/on/data-disk/huggingface
export PERCEPTION_API_KEY='<random-token>'
export INSTRUCTOR_DEVICE=cuda
python -m uvicorn deployment.perception_service.app:app \
  --host 127.0.0.1 --port 8010
```

The first perception request downloads `hkunlp/instructor-large`. Check health on
the GPU host with an `X-API-Key` header, then keep the service terminal running.

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
