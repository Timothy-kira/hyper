#!/usr/bin/env python3
"""Draw docs/s3t_architecture.svg, the S3T-DETR structure diagram.

    python3 tools/draw_s3t_arch.py
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
W, H = 1500, 1150
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
    text(40, 44, "S3T-DETR 模型结构（S3T-X 版）", c="#0f172a", fs=26, anchor="start", weight="700")
    text(40, 70, "光谱：自研 S3T-X 编码器（通道协方差注意力，MAE v3 预训练）；空间：COCO 预训练 RT-DETR-L；"
         "检测头：RT-DETR decoder ＋ D-FINE 分布回归。所有新层零初始化，第 0 步 = COCO RT-DETR。", fs=14, anchor="start")
    for i, (k, t) in enumerate([("in", "固定预处理"), ("enc", "S3T-X 编码器"), ("new", "新加的零初始化层"),
                                ("det", "COCO 预训练 DETR"), ("mae", "仅预训练用")]):
        f, s_ = COL[k]
        xx, yy = 1040 + (i % 3) * 150, 92 + (i // 3) * 24
        out.append(f'<rect x="{xx}" y="{yy - 12}" width="16" height="14" rx="3" fill="{f}" stroke="{s_}"/>')
        text(xx + 22, yy, t, fs=12, anchor="start")

    # ① input
    section(108, "① 输入：固定预处理（无可学习参数，检测和预训练共用同一个函数）")
    y1, w1, g1 = 135, 240, 42
    xs1 = [40 + i * (w1 + g1) for i in range(5)]
    items = [["官方 PNG", "16-bit，单通道 (4H, 4W)", "4×4 光谱滤光片阵列"],
             ["官方 X2Cube", "→ (H, W, 16)", "460–600 nm，16 个窄带"],
             ["log1p ＋ 每帧 P2–P98 缩放", "逐波段亚像素对齐", "→ 16 通道 uint8（检测输入）"],
             ["0.5× 缩放", "回到原生像素尺度", "（ultralytics 放大到 1024）"],
             ["observed_features", "level：亮度（保留）", "shape：减去存在波段的均值", "contrast：可见像素的 31/63 环"]]
    for i, (x, it) in enumerate(zip(xs1, items)):
        box(x, y1, w1, 96, it, "in")
        if i:
            arrow(xs1[i - 1] + w1, y1 + 48, x - 4, y1 + 48)
    text(xs1[4] + w1 / 2, y1 + 116, "(B, 3, 16, H/2, W/2) → 48 通道", c="#0f172a", fs=13)

    # ② encoder
    section(284, "② S3T-X 光谱编码器（0.21M 参数，MAE v3 预训练）",
            "每个位置只保留一个 64 维向量（旧版是 16 个 token × 64 维）；全程 channels_last；没有 BatchNorm")
    y2, w2, g2 = 340, 250, 40
    xs2 = [40 + i * (w2 + g2) for i in range(5)]
    enc = [["Stem", "Conv 3×3 stride 2：48 → 64", "GELU → Conv 1×1", "不做逐像素归一化（保留亮度）"],
           ["XCA block ×2", "16×16 窗口（≈32 原生像素）", "局部背景的光谱协方差", "≈ 可学习的局部白化（RX）"],
           ["XCA block ×2", "全图", "场景级归一化（光照）"],
           ["LayerNorm", "→ (B, 64, H/4, W/4)", "输入的 stride 4"],
           ["速度（T4，1024²，b2）", "0.50 s/step（编译）", "旧编码器 3.39 s/step", "纯 RT-DETR 0.32 s/step"]]
    for i, (x, it) in enumerate(zip(xs2, enc)):
        box(x, y2, w2, 110, it, "enc" if i < 4 else "in")
        if 0 < i < 4:
            arrow(xs2[i - 1] + w2, y2 + 55, x - 4, y2 + 55)
    path(f"M {xs1[4] + w1 / 2} {y1 + 124} L {xs1[4] + w1 / 2} 326 L {xs2[0] + w2 / 2} 326 L {xs2[0] + w2 / 2} {y2 - 4}")

    iy = 474
    out.append(f'<rect x="{xs2[1] - 10}" y="{iy}" width="1060" height="112" rx="10" fill="#ffffff" '
               f'stroke="#93c5fd" stroke-dasharray="5 4"/>')
    text(xs2[1], iy + 22, "XCA block 内部（XCiT / Restormer MDTA / MST++）", c="#1d4ed8", fs=13, anchor="start")
    parts = [("x (B,64,h,w)", 0, 100), ("LN", 1, 36), ("qkv 1×1 ＋ dw 3×3", 1, 136),
             ("Q̂,K̂ 沿像素 L2 归一化", 1, 168), ("softmax(K̂ᵀQ̂·τ)·V", 1, 150), ("× γ₁ ＋ 残差", 0, 96),
             ("LN → GDFN", 1, 100), ("× γ₂ ＋ 残差", 0, 96)]
    cx = xs2[1]
    for j, (t, k, ww) in enumerate(parts):
        out.append(f'<rect x="{cx}" y="{iy + 40}" width="{ww}" height="34" rx="6" '
                   f'fill="{"#dbeafe" if k else "#f8fafc"}" stroke="#60a5fa"/>')
        text(cx + ww / 2, iy + 62, t, c="#0f172a", fs=12)
        if j < len(parts) - 1:
            arrow(cx + ww, iy + 57, cx + ww + 10, iy + 57)
        cx += ww + 12
    text(xs2[1], iy + 98, "注意力矩阵是 16×16（每头 16 个通道），计算量对像素数线性；γ 是 LayerScale；"
         "MAE 时 q、k 和分支输入在被遮位置置 0，可见输出读不到被遮位置", fs=12, anchor="start")
    path(f"M {xs2[1] + w2 / 2} {y2 + 110} L {xs2[1] + w2 / 2} {iy - 2}", c="#93c5fd", dash=True)

    # ③ detection
    section(628, "③ 检测：COCO 预训练 RT-DETR-L ＋ D-FINE 分布回归（从 COCO 开始训练）")
    y3 = 706
    box(40, y3 + 30, 200, 90, ["base 1×1：16 → 3", "LDA 初始化", "16 波段 level 图"], "new")
    box(270, y3 + 16, 190, 118, ["HGStem", "COCO 预训练", "→ 48 通道，stride 4"], "det")
    arrow(240, y3 + 75, 266, y3 + 75)
    out.append(f'<circle cx="490" cy="{y3 + 75}" r="15" fill="#fff" stroke="#c2410c" stroke-width="1.8"/>')
    text(490, y3 + 80, "＋", c="#c2410c", fs=16)
    arrow(460, y3 + 75, 473, y3 + 75)
    box(525, y3 + 16, 190, 118, ["HGNetv2", "COCO 预训练", "→ P3 /8、P4 /16、P5 /32"], "det")
    arrow(505, y3 + 75, 521, y3 + 75)
    box(750, y3, 230, 150, ["input_proj（第 19/14/10 层）", "＋ 光谱金字塔注入", "（stride-2 卷积 ×3，零初始化 1×1）",
                            "P3/P4 先对 AIFI 全局 token", "做交叉注意力"], "det")
    arrow(715, y3 + 75, 746, y3 + 75)
    box(1010, y3 + 16, 200, 118, ["Hybrid encoder", "AIFI 全局自注意力", "CCFM 跨尺度融合"], "det")
    arrow(980, y3 + 75, 1006, y3 + 75)
    path(f"M 1010 {y3 + 118} L 984 {y3 + 118}", c="#15803d")
    box(1240, y3, 230, 150, ["Decoder（6 层，300 query）", "COCO box head 保留", "＋ 零初始化 FDR 头：", "每条边 33-bin 分布",
                             "框边移动分布期望"], "det")
    arrow(1210, y3 + 75, 1236, y3 + 75)
    # spectral features: stem fusion at stride 4, pyramid to the injections
    # spectral features leave the encoder to the right of the XCA detail box
    sp = xs2[3] + w2 / 2
    path(f"M {sp} {y2 + 110} L {sp} {y2 + 122} L 1440 {y2 + 122} L 1440 664 L 490 664 L 490 {y3 + 58}",
         c="#c2410c", m="o")
    path(f"M 865 664 L 865 {y3 - 4}", c="#c2410c", m="o")
    text(500, 686, "stem 融合：零初始化 1×1（64→48），同在 stride 4", c="#c2410c", fs=12, anchor="start")
    text(875, 686, "光谱金字塔 → P3/P4/P5", c="#c2410c", fs=12, anchor="start")
    text(1430, 656, "S3T-X 特征 (B, 64, H/4, W/4)", c="#c2410c", fs=12, anchor="end")
    text(40, y3 + 176, "损失：MAL（正样本全权重，目标 IoU^1.5）· GIoU ＋ log-size L1 · FGL 0.15（分布 vs GT，两 bin）"
         " · DDF/GO-LSD 1.5（最后一层框蒸馏给前面各层）· 去噪 query 同样计入", c="#9a3412", fs=12, anchor="start")
    text(40, y3 + 196, "加速：AMP fp16（loss 与匈牙利匹配 fp32）· RT-DETR 注意力走 SDPA · S3T-X block 逐个 torch.compile"
         " · fused AdamW · cudnn.benchmark · 双卡 DDP", c="#9a3412", fs=12, anchor="start")

    # ④ MAE
    section(955, "④ MAE v3 预训练（只训练编码器；官方 3000 训练 ＋ 1000 测试图，不读标签，不增强）")
    y4, w4, g4 = 980, 250, 40
    xs4 = [40 + i * (w4 + g4) for i in range(5)]
    mae = [["先 mask，再算特征", "2×2 token 单位，0.5 → 0.75", "1–4 个波段（70% 按波长连续）", "修掉 v1/v2 的 shape/contrast 泄漏"],
           ["编码器（同 ②）", "稠密计算，被遮位置恒为 0", "缺失波段 ＋ 可学习向量"],
           ["解码器（预训练后丢弃）", "2 层全局 attention ＋ 2D sincos", "＋ 1 个窗口 XCA block"],
           ["重建", "每个位置 16 波段 × 2×2 px"],
           ["损失", "空间补全 ＋ 2 × 光谱补全", "归一化 MSE ＋ L1 ＋ 光谱梯度", "参照：波长插值 / 可见均值"]]
    for i, (x, it) in enumerate(zip(xs4, mae)):
        box(x, y4, w4, 110, it, "mae")
        if i:
            arrow(xs4[i - 1] + w4, y4 + 55, x - 4, y4 + 55)
    text(40, y4 + 140, "双卡 T4：DDP · fp16 ＋ GradScaler · SDPA mem-efficient · 逐 block torch.compile（失败不回退）· fused AdamW"
         " · 114 crops/s；1400 步时 光谱补全/插值 0.45，空间补全/均值 0.59", c="#6d28d9", fs=12, anchor="start")
    out.append("</svg>")
    dst = REPO / "docs" / "s3t_architecture.svg"
    dst.write_text("\n".join(out))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
