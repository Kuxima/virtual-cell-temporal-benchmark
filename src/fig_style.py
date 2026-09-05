# -*- coding: utf-8 -*-
"""论文图形规范共享样式（全部绘图脚本统一引用；作者指定大写面板编号）。

规范要点（基于 Nature 排版指南，面板编号按作者要求用大写 A/B/C）：
- 无衬线 Arial/Helvetica；最终印刷尺寸下正文/刻度 5-7 pt，面板编号 8 pt 粗体大写字母
- 双栏宽 183 mm（7.205 in），单栏 89 mm（3.504 in）；图形即最终印刷尺寸渲染
- 图内不放总标题（suptitle）与描述性面板标题——说明文字进 legend
- PDF/PS 内嵌 Type 42（TrueType）字体，文字可编辑
- 色板：Okabe-Ito 色盲友好色（MODEL_COLORS 见 common.py）

用法：
    from fig_style import setup_nature, panel_label, panel_header, save_fig, DOUBLE_COL
    setup_nature()
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.0))
    panel_label(ax, "A")
    save_fig(fig, RESULTS, "fig2_multimodel_benchmark")
"""
import os

import matplotlib as mpl

MM2IN = 1 / 25.4
SINGLE_COL = 89 * MM2IN    # 3.50 in
DOUBLE_COL = 183 * MM2IN   # 7.21 in

FONT_SANS = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def setup_nature():
    mpl.use("Agg")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_SANS,
        # 最终印刷尺寸字号：正文/刻度 5-7 pt，轴标签 7 pt
        "font.size": 6,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        # 线宽不低于 Nature 最小值（0.3 pt 上下）
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "errorbar.capsize": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "axes.grid": False,
        # 字体以 TrueType(Type 42) 内嵌进 PDF/PS，文字保持可编辑
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "savefig.dpi": int(os.environ.get("FIG_DPI", "600")),
    })


def fig_env():
    """(dpi, fmt)；FIG_FORMAT 支持 png/pdf/both（默认 both，一次出评审位图+矢量）。"""
    return int(os.environ.get("FIG_DPI", "600")), os.environ.get("FIG_FORMAT", "both")


def save_fig(fig, results_dir, stem):
    dpi, fmt = fig_env()
    fmts = ["png", "pdf"] if fmt == "both" else [fmt]
    outs = []
    for f in fmts:
        out = results_dir / f"{stem}.{f}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        outs.append(out)
    print("saved", ", ".join(str(o) for o in outs))
    return outs


def panel_label(ax, letter, x=0.0, y=1.02):
    """面板编号：8 pt 粗体大写字母（A, B, C...，作者指定风格），面板左上角外侧。"""
    ax.text(x, y, str(letter).upper(), transform=ax.transAxes,
            fontsize=8, fontweight="bold", ha="left", va="bottom")


def panel_header(ax, letter, name, x=0.0, y=1.02):
    """编号 + 面板名（UMAP 网格 / 混淆矩阵等需要标注模型名的面板）。

    名字偏移按坐标轴物理宽度折算（8 pt 粗体字母 ~0.07 in + ~0.05 in 间隙），
    保证任何面板宽度下字母与名字都不粘连。"""
    ax_w_in = ax.get_position().width * ax.figure.get_figwidth()
    dx = 0.12 / max(ax_w_in, 0.5)
    ax.text(x, y, str(letter).upper(), transform=ax.transAxes,
            fontsize=8, fontweight="bold", ha="left", va="bottom")
    ax.text(x + dx, y + 0.008, name, transform=ax.transAxes,
            fontsize=7, ha="left", va="bottom")
