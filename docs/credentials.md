# Credentials: the `api-keys` secret

`full-baseline.yaml` reads three values out of a Kubernetes Secret named
`api-keys` (see `full-baseline.yaml:53-67`). The Secret is built from a local
env file (that should never get committed to this repo).

## 1. Create the file

```bash
mkdir -p ~/.config/nc-baseline
```

Create `~/.config/nc-baseline/credentials.env` with exactly these three
fields, `KEY=value` per line, no `export`, no quotes:

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_ID=<your-huggingface-username-or-org>
```

Then lock it down, since it's a plaintext credential file:

```bash
chmod 600 ~/.config/nc-baseline/credentials.env
```

Where each value comes from:

- **`HF_TOKEN`** — a Hugging Face access token with  **write** access
- **`WANDB_API_KEY`** — from wandb.ai/authorize
- **`HF_ID`** — your Hugging Face username

## 2. Load it into the cluster

```bash
kubectl create secret generic api-keys \
  --from-env-file="$HOME/.config/nc-baseline/credentials.env" \
  --dry-run=client -o yaml | kubectl apply -f -
```