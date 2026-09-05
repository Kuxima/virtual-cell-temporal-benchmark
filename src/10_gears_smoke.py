# -*- coding: utf-8 -*-
"""GEARS 冒烟测试：下载 Norman 数据 → 训练 5 个 epoch → 预测一个扰动。
对齐官方 model_tutorial，缩短训练轮数以便快速验证。"""
import os
import time

import torch
from gears import GEARS, PertData

from common import DATA_ROOT, RESULTS

OUT = str(RESULTS)
DATA_DIR = DATA_ROOT / "raw" / "gears"
RESULTS.mkdir(parents=True, exist_ok=True)

t0 = time.time()

# 1) 数据（首次运行自动下载 Norman 2019）
pert_data = PertData(str(DATA_DIR))
pert_data.load(data_name="norman")
# Windows 兼容：GEARS 0.1.2 的 prepare_split 不会自建嵌套目录
(DATA_DIR / "norman" / "splits" / "gears" / "norman").mkdir(parents=True, exist_ok=True)
pert_data.prepare_split(split="simulation", seed=1)
pert_data.get_dataloader(batch_size=64)

print(f"[smoke] data ready in {time.time()-t0:.0f}s; perturbations={len(pert_data.set2conditions)}")

# 2) 模型（官方教程默认超参，epoch 压到 5）
# Windows 兼容：model_initialize 与 train 都会往这些目录写文件，但不会自建
(DATA_DIR / "gears" / "norman").mkdir(parents=True, exist_ok=True)
device = os.environ.get("GEARS_DEVICE", "cuda")
epochs = int(os.environ.get("GEARS_EPOCHS", "5"))
model = GEARS(pert_data, device=device)
model.model_initialize(hidden_size=64)
model.train(epochs=epochs)

# 3) 预测一个单基因扰动（predict 接受基因名列表）
gene = model.pert_list[0]
pred, _ = model.predict([[gene]])
print(f"[smoke] prediction for {gene}: {'OK' if pred is not None else 'FAILED'}")

print("GEARS SMOKE TEST PASSED")
