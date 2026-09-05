# -*- coding: utf-8 -*-
"""用 huggingface_hub 下载 SE-600M（比 git clone 更抗代理）。"""
import os

# 走国内镜像，避开 huggingface.co 经代理的 SSL 问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

from huggingface_hub import snapshot_download

from common import STATE_MODEL_DIR

out = STATE_MODEL_DIR
out.mkdir(parents=True, exist_ok=True)
print("downloading arcinstitute/SE-600M ->", out)
path = snapshot_download(
    repo_id="arcinstitute/SE-600M",
    local_dir=out,
    resume_download=True,
)
print("DONE", path)
