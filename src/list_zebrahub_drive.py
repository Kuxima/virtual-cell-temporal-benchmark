# -*- coding: utf-8 -*-
"""枚举 Zebrahub Google Drive 文件夹树（只列不下载），输出所有文件的路径与大小。"""
import os

import gdown

URL = "https://drive.google.com/drive/folders/1SIkI6FAOtNp3hCeJYxfi19EgnXCDCgOu"

files = gdown.download_folder(URL, skip_download=True, quiet=True)
print(f"total entries: {len(files)}")
for f in files:
    # gdown 返回 GoogleDriveFile 对象：path 为相对路径
    try:
        size = f.file_size or 0
    except Exception:
        size = 0
    name = getattr(f, "path", None) or f.title
    print(f"{size/1e6:>10.1f} MB  {name}  [id={f.id}]")
