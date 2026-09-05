# -*- coding: utf-8 -*-
"""从 Lightning ckpt 抽出可加载的瘦文件。ckpt 里夹了训练时代码路径（vci），需要 monkeypatch。"""
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import STATE_MODEL_DIR


class Dummy:
    def __init__(self, *a, **k):
        pass

    def __getattr__(self, name):
        return Dummy

    def __call__(self, *a, **k):
        return Dummy()


def install_dummy(modname):
    if modname in sys.modules:
        return
    m = types.ModuleType(modname)
    m.__getattr__ = lambda n: Dummy  # type: ignore
    sys.modules[modname] = m


for name in [
    "vci",
    "vci.model",
    "vci.models",
    "vci.nn",
    "vci.emb",
    "vci.emb.nn",
    "vci.emb.nn.model",
]:
    install_dummy(name)

# catch-all import hook
class Finder:
    def find_module(self, fullname, path=None):
        if fullname.startswith("vci"):
            return self
        return None

    def load_module(self, fullname):
        install_dummy(fullname)
        return sys.modules[fullname]

    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("vci"):
            import importlib.machinery
            return importlib.util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        install_dummy(spec.name)
        return sys.modules[spec.name]

    def exec_module(self, module):
        return None


sys.meta_path.insert(0, Finder())

src = STATE_MODEL_DIR / "se600m_epoch4.ckpt"
dst = STATE_MODEL_DIR / "se600m_epoch4_slim.ckpt"
if not src.exists():
    raise FileNotFoundError(f"missing legacy checkpoint: {src}; use 09_state_embed_safetensors.py with the available safetensors weights")
print("loading ckpt on CPU...")
ckpt = torch.load(src, map_location="cpu", weights_only=False)
print("type", type(ckpt))
if not isinstance(ckpt, dict):
    raise SystemExit("not a dict")
print("top keys:", list(ckpt.keys())[:30])
keep = {}
for k in ("state_dict", "hyper_parameters", "cfg_yaml", "pytorch-lightning_version", "epoch", "global_step", "protein_embeds_dict"):
    if k in ckpt:
        keep[k] = ckpt[k]
        print(" kept", k, type(ckpt[k]))
if "state_dict" in keep:
    n = sum(v.numel() * v.element_size() for v in keep["state_dict"].values() if torch.is_tensor(v))
    print("state_dict MB", n / 1e6)
torch.save(keep, dst)
print("saved", dst, "bytes", os.path.getsize(dst))
