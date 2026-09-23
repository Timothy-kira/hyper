#!/usr/bin/env python3
"""Draw docs/s3t_architecture.svg, the S3T-DETR structure diagram.

    python3 tools/draw_s3t_arch.py
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
W, H = 1500, 1110
FONT = 'font-family="PingFang SC, Noto Sans CJK SC, Microsoft YaHei, WenQuanYi Zen Hei, Helvetica, Arial, sans-serif"'
COL = {"in": ("#f1f5f9", "#64748b"), "enc": ("#eff6ff", "#2563eb"), "det": ("#f0fdf4", "#16a34a"),
       "new": ("#fff7ed", "#c2410c"), "mae": ("#faf5ff", "#7c3aed")}
out: list[str] = []


def box(x, y, w, h, lines, kind, fs=13):
    f, s = COL[kind]
    out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{f}" stroke="{s}" stroke-width="1.6"/>')
    lh = fs + 5
    y0 = y + h / 2 - (len(lines) - 1) * lh / 2 + fs / 3
    for i, t in enumerate(lines):
        wt, col, size = ("700", "#0f172a", fs) if i == 0 else ("400", "#334155", fs - 1)
        out.append(f'<text x="{x + w / 2}" y="{y0 + i * lh}" font-size="{size}" font-weight="{wt}" '
                   f'fill="{col}" text-anchor="middle">{t}</text>')


def arrow(x1, y1, x2, y2, c="#475569", m="a"):
    out.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="1.8" marker-end="url(#{m})"/>')


def path(d, c="#475569", m="a", dash=False):
    extra = ' stroke-dasharray="6 4"' if dash else ""
    out.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="1.8" marker-end="url(#{m})"{extra}/>')


def text(x, y, t, c="#475569", fs=12, anchor="middle", weight="400"):
    out.append(f'<text x="{x}" y="{y}" font-size="{fs}" font-weight="{weight}" fill="{c}" text-anchor="{anchor}">{t}</text>')


def section(y, title, sub=""):
    text(40, y, title, c="#0f172a", fs=17, anchor="start", weight="700")
    if sub:
        text(40, y + 20, sub, c="#64748b", fs=13, anchor="start")


def main() -> None:
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" {FONT}>')
    out.append('<defs>'
               '<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#475569"/></marker>'
               '<marker id="o" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#c2410c"/></marker>'
               '</defs>')
    out.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    text(40, 44, "S3T-DETR 模型结构", c="#0f172a", fs=26, anchor="start", weight="700")
    text(40, 70, "光谱 Transformer（自研，MAE 预训练）＋ COCO 预训练 RT-DETR-L（负责空间）。"
         "S3M 的光谱 Mamba 在这里换成了 Transformer。", fs=14, anchor="start")
    for i, (k, t) in enumerate([("in", "固定预处理"), ("enc", "S3T 编码器"), ("new", "新加的零初始化层"),
                                ("det", "COCO 预训练 DETR"), ("mae", "仅预训练用")]):
        f, s = COL[k]
        xx, yy = 1040 + (i % 3) * 150, 92 + (i // 3) * 24
        out.append(f'<rect x="{xx}" y="{yy - 12}" width="16" height="14" rx="3" fill="{f}" stroke="{s}"/>')
        text(xx + 22, yy, t, fs=12, anchor="start")

    # ① input
    section(108, "① 输入：固定预处理（无可学习参数，检测和预训练共用）")
    y1, w1, g1 = 135, 240, 42
    xs1 = [40 + i * (w1 + g1) for i in range(5)]
    items = [["官方 PNG", "16-bit，单通道 (4H, 4W)", "4×4 光谱滤光片阵列"],
             ["官方 X2Cube", "→ (H, W, 16)", "460–600 nm，16 个窄带"],
             ["log1p ＋ 每帧 P2–P98 缩放", "去曝光差异，不截断", "帧内亮度比例全保留"],
             ["逐波段亚像素对齐", "第 k 波段平移 (k//4, k%4)/4 px", "16 个波段对齐到块中心"],
             ["每个波段 3 个特征", "level：亮度（保留）", "shape：去亮度光谱形状", "contrast：31/63 环形局部对比"]]
    for i, (x, it) in enumerate(zip(xs1, items)):
        box(x, y1, w1, 96, it, "in")
        if i:
            arrow(xs1[i - 1] + w1, y1 + 48, x - 4, y1 + 48)
    text(xs1[4] + w1 / 2, y1 + 116, "张量 (B, 3, 16, H, W)", c="#0f172a", fs=13)

    # ② encoder
    section(284, "② S3T 光谱编码器（0.25M 参数，权重来自 MAE 预训练）",
            "token 布局 (B, S, C=16, D=64)：S 是空间位置，每个位置有 16 个波段 token")
    y2, w2, g2 = 340, 178, 27
    xs2 = [40 + i * (w2 + g2) for i in range(7)]
    enc = [["Stem（16 波段共享）", "Conv 3×3, stride 2", "BatchNorm → GELU", "Conv 1×1 → D=64", "不做 LayerNorm"],
           ["波段 token", "(B, S, 16, 64)", "＋ 可学习光谱", "位置编码"],
           ["SpectralBlock ×2", "16 个波段 token", "之间自注意力"],
           ["SpatialMix", "depthwise 7×7", "（16 波段共享）", "＋ MLP"],
           ["SpectralBlock ×2", "16 个波段 token", "之间自注意力"],
           ["SpatialMix", "depthwise 7×7", "＋ MLP"],
           ["SpectralPool", "对 16 个波段做", "注意力加权求和", "→ (B, 64, H/2, W/2)"]]
    for i, (x, it) in enumerate(zip(xs2, enc)):
        box(x, y2, w2, 110, it, "enc")
        if i:
            arrow(xs2[i - 1] + w2, y2 + 55, x - 4, y2 + 55)
    # tensor -> stem, routed below the section subtitle
    path(f"M {xs1[4] + w1 / 2} {y1 + 124} L {xs1[4] + w1 / 2} 326 L {xs2[0] + w2 / 2} 326 L {xs2[0] + w2 / 2} {y2 - 4}")

    iy = 474
    out.append(f'<rect x="{xs2[2] - 10}" y="{iy}" width="{w2 * 2 + g2 + 225}" height="112" rx="10" fill="#ffffff" '
               f'stroke="#93c5fd" stroke-dasharray="5 4"/>')
    text(xs2[2], iy + 22, "SpectralBlock 内部（取代 S3M 的 Spectral Mamba 块）", c="#1d4ed8", fs=13, anchor="start")
    parts = [("x (B·S, 16, 64)", 0, 112), ("LN", 1, 40), ("MHSA 4 头 (SDPA)", 1, 138), ("× γ₁ ＋ 残差", 0, 98),
             ("LN", 1, 40), ("MLP 64→256→64", 1, 118), ("× γ₂ ＋ 残差", 0, 98)]
    cx = xs2[2]
    for j, (t, k, ww) in enumerate(parts):
        out.append(f'<rect x="{cx}" y="{iy + 40}" width="{ww}" height="34" rx="6" '
                   f'fill="{"#dbeafe" if k else "#f8fafc"}" stroke="#60a5fa"/>')
        text(cx + ww / 2, iy + 62, t, c="#0f172a", fs=12)
        if j < len(parts) - 1:
            arrow(cx + ww, iy + 57, cx + ww + 10, iy + 57)
        cx += ww + 12
    text(xs2[2], iy + 98, "γ 是 LayerScale（初值 0.1）；LN 只在分支里，残差主干保留绝对亮度；"
         "各空间位置互不依赖，所以 MAE 只算可见位置", fs=12, anchor="start")
    path(f"M {xs2[2] + w2 / 2} {y2 + 110} L {xs2[2] + w2 / 2} {iy - 2}", c="#93c5fd", dash=True)

    # ③ detection
    section(622, "③ 检测：COCO 预训练 RT-DETR-L（负责空间）")
    y3 = 668
    box(40, y3, 230, 76, ["head 1×1：64 → 3", "零初始化，上采样回 (H, W)"], "new")
    box(40, y3 + 96, 230, 76, ["base 1×1：16 → 3", "LDA 初始化", "输入 16 波段 level 图"], "new")
    out.append(f'<circle cx="320" cy="{y3 + 86}" r="17" fill="#fff" stroke="#c2410c" stroke-width="1.8"/>')
    text(320, y3 + 92, "＋", c="#c2410c", fs=18)
    arrow(270, y3 + 38, 306, y3 + 74, c="#c2410c", m="o")
    arrow(270, y3 + 134, 306, y3 + 98, c="#c2410c", m="o")
    box(365, y3 + 16, 215, 140, ["HGStem + HGNetv2", "stage 1–4", "COCO 预训练", "输出 P3 / P4 / P5"], "det")
    arrow(337, y3 + 86, 361, y3 + 86)
    box(620, y3, 245, 172, ["input_proj（第 19/14/10 层）", "P3 /8、P4 /16、P5 /32", "",
                            "＋ 侧注入 Inject：", "自适应池化 → 1×1 零初始化"], "det")
    arrow(580, y3 + 86, 616, y3 + 86)
    box(905, y3 + 16, 245, 140, ["Hybrid encoder", "AIFI（P5 自注意力）", "CCFM 跨尺度融合"], "det")
    arrow(865, y3 + 86, 901, y3 + 86)
    box(1190, y3 + 16, 270, 140, ["Transformer decoder", "300 个 query，去噪训练", "→ 框 ＋ 18 类"], "det")
    arrow(1150, y3 + 86, 1186, y3 + 86)
    sp = xs2[6] + w2 / 2
    path(f"M {sp} {y2 + 110} L {sp} 646 L 155 646 L 155 {y3 - 4}", c="#c2410c", m="o")
    path(f"M 742 646 L 742 {y3 - 4}", c="#c2410c", m="o")
    text(sp - 8, 640, "S3T 特征 (B, 64, H/2, W/2)", c="#c2410c", fs=12, anchor="end")
    text(40, y3 + 198, "橙色 = 新加的零初始化层：训练第 0 步，检测器看到的只是 16→3 投影，光谱特征靠梯度逐步接入。"
         "编码器在 0.5× 输入尺度上跑（回到原生像素尺度），梯度检查点 ＋ fp16。", c="#9a3412", fs=12, anchor="start")

    # ④ MAE
    section(912, "④ MAE 预训练（只训练光谱编码器，DETR 不参与；数据 = 官方 3000 训练 + 1000 测试图，不读标签）")
    y4, w4, g4 = 935, 250, 40
    xs4 = [40 + i * (w4 + g4) for i in range(5)]
    mae = [["两种掩码", "tube 空间掩码 75%", "（4×4 token 为单位，16 波段一起遮）", "连续 2 个波段（按波长顺序）"],
           ["编码器（同 ②）", "只计算可见位置", "→ token 数只剩 1/4"],
           ["轻量解码器", "SpatialMix ＋ SpectralBlock", "宽度 32，预训练后丢弃"],
           ["重建", "每个 token 的 2×2 像素", "× 16 波段的 level"],
           ["损失", "归一化 MSE", "＋ 0.5 × L1（防平坦光谱）", "＋ 0.5 × 光谱梯度"]]
    for i, (x, it) in enumerate(zip(xs4, mae)):
        box(x, y4, w4, 110, it, "mae")
        if i:
            arrow(xs4[i - 1] + w4, y4 + 55, x - 4, y4 + 55)
    text(40, y4 + 140, "双卡 T4：DDP (NCCL) · fp16 + GradScaler · SDPA mem-efficient（T4 不支持 FlashAttention）"
         " · fused AdamW · 逐 block torch.compile（可选 CUDA Graphs，失败不回退）", c="#6d28d9", fs=12, anchor="start")
    out.append("</svg>")
    dst = REPO / "docs" / "s3t_architecture.svg"
    dst.write_text("\n".join(out))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
