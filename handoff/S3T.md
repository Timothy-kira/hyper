# S3T-DETR 接力：光谱 Transformer + COCO RT-DETR，双卡 T4

这是一条**独立于 `README.md` 那条续训路线**的新模型线。README 里的 epoch-21 续训依然有效，两条线可以二选一，也可以都跑（额度够的话）。

## 一句话

在 COCO 预训练的 RT-DETR-L 前面，加一个自己设计的**逐像素光谱 Transformer**（S3T），用 MAE 在官方 4000 张图上预训练过。空间部分交给 DETR，光谱部分由我们自己的编码器负责。

## 已经完成的

| 环节 | 状态 | 在哪里 |
| --- | --- | --- |
| MAE 预训练（只训光谱编码器，DETR 不参与） | **已完成**，结果见下 | 公开 notebook `qwyi123/hod26-s3t-mae-pretrain`，输出 `s3t_mae/pretrain_mae.pt` |
| 检测微调代码（S3T 前端 + 侧注入 + 数据增强） | **已写好，CPU 上测试通过，还没在 GPU 上跑过** | `handoff/s3t/hod26_round.py`（由 `tools/s3t_round.py` 生成） |

预训练结果：**运行中**（2026-09-23 约 06:45 UTC 结束，结束后本行更新）。

## 你要做的（和 README 那条线几乎一样）

1. **检查额度**：需要约 11–12 小时 GPU。
2. **确认能打开数据集** `xishengfeng/hod26-planar`（私有；打不开就先找 xishengfeng 加你为协作者）。
3. kaggle.com → Create → New Notebook → File → Import Notebook，上传：
   <https://raw.githubusercontent.com/Timothy-kira/hyper/claude/kaggle-cli-setup-handoff-9w0v7x/handoff/s3t/hod26_round.py>
4. 右侧面板 → **Input → Add Input**，添加**两项**：
   - Datasets：`hod26-planar`
   - Notebooks：`qwyi123/hod26-s3t-mae-pretrain`（公开 notebook，里面有 `pretrain_mae.pt`）

   **不要**再挂 `hod26-ckpt-s4` / `-s2`：这条线从 COCO 开始，不续训旧 checkpoint。
5. **Accelerator → GPU T4 x2**，**Internet → On**。
6. **Save Version → Save & Run All (Commit)**。

### 前两分钟应该看到

```
preflight ok: 2x GPU Tesla T4
preflight ok: S3T MAE encoder /kaggle/input/.../pretrain_mae.pt
preflight passed
  S3T encoder: MAE weights from ... (step N)
  S3T front: 0.25M-param spectral Transformer at 0.5x input scale, 16->3 into the pretrained HGStem, zero-init side injections at layers [19, 14, 10]
  2 GPU(s) visible; DDP across [0, 1], batch 4 (2/card)
```

- 如果看到 `PREFLIGHT FAILED: spectral_stem=s3t but no MAE checkpoint`：第 4 步的 notebook 没挂上。**还没花额度**，挂上再提交。
- 如果看到 `S3T encoder: NO pretrained weights`：说明没找到权重，**立刻停掉**并告诉我们。
- 数据渲染（含每帧 1 份增强副本）需要一段时间，第一条 epoch 日志会晚一些才出来。

**第一次在 GPU 上跑这份检测代码**：CPU 测试覆盖了安装、前向、反向、EMA 深拷贝、保存与加载、验证前的算子融合（fuse），但显存和速度只能在 T4 上才看得到。建议看到**第一条 epoch 日志**再离开。若出现 OOM，把 `train.batch` 从 2 改成 1 重新生成（`python3 tools/s3t_round.py`），或者直接告诉我们。

## 数据增强（正式训练里都开着）

| 类型 | 做法 | 为什么 |
| --- | --- | --- |
| 离线·光谱 | **Savitzky-Golay 沿波长顺序**平滑（窗口 7，2 阶） | mosaic 编号 ≠ 波长顺序（band 4、11 错位），旧代码按编号平滑会把不相邻的波长混在一起；新增 `sg_chain` 修正 |
| 离线·光谱 | 同类 SMOTE（α=0.3） | 同类目标之间插值，不改标签 |
| 离线·空间 | 超像素 CutMix（p=0.4） | 按超像素整块移植，不切断目标 |
| 离线 | 每帧额外生成 1 份增强副本，训练集变成 2 倍 | |
| 在线（ultralytics） | mosaic、水平翻转、缩放/平移 | 最后 3 个 epoch 关闭 mosaic |
| 不做 | HSV、mixup | 16 通道上 HSV 无意义；mixup 跨类插值，理由同 S3M 报告 |

验证集（600 张）**永远不增强**，保证分数衡量的是真实分布。

## 模型结构（详细见 `docs/S3T.md`）

- 输入：官方 X2Cube → log 辐亮度 → 每帧 P2–P98 缩放 → **逐波段亚像素对齐** → 16 通道 uint8。
- S3T 前端：每个波段 token 带 3 个特征（亮度 level、去亮度后的光谱形状 shape、63px 环形背景的局部对比 contrast），经过 4 层**光谱自注意力**（每个像素内 16 个波段 token 之间做 attention）和 2 次局部空间混合，再对波段做注意力池化。
- 两条通路接入 DETR：① 1×1 卷积投影到 3 通道，送进 COCO stem；② 零初始化的侧注入，加到 P3/P4/P5 的输入投影上。
- 零初始化意味着训练第 0 步时，检测器看到的就是一个普通的 16→3 投影，光谱特征靠梯度逐步接入，不会一上来就把预训练的 DETR 冲乱。

## 怎么判断它有没有用

和 README 那条线用**同一把尺子**：held-out 600 张上的 pycocotools mAP50-95。

| 参照 | held-out | LB |
| --- | --- | --- |
| 当前队伍最佳（epoch-52 微调） | 0.69528 | 0.62994 |

要比参照高出 **0.01 以上**才算有效（run-to-run 噪声约 0.0017）。这条线是从 COCO 开始训的，不是在我们 52 个 epoch 的模型上续训，所以 11 小时内**不一定能追上**。它真正的价值是看弱类（stone_block / people / e-bike / car）的 AP 有没有涨，日志末尾会打印每一类的 AP。

## 可选：再跑一轮更长的 MAE 预训练（先做加速冒烟）

现有的 `pretrain_mae.pt` 只在一个账号剩下的额度里训了约 1 小时。如果你的额度有富余，想再训一个更长的版本，**先花约 25 分钟跑一次加速冒烟测试**：

```bash
python3 tools/build_s3t_smoke.py --slug <你的账号>/hod26-s3t-mae-smoke
kaggle kernels push -p kernels/s3t_smoke/build
```

它先用单卡跑一遍，作为速度参照；再用双卡把同一个预训练分别按三种方式各跑 4 分钟：

| 方案 | 做法 |
| --- | --- |
| `eager` | 不编译，作为对照 |
| `fused` | 用 `torch.compile` 逐个 block 编译，把 LayerNorm、GELU、残差相加这些小 kernel 合并成少数几个 Triton kernel |
| `graphs` | 在 `fused` 的基础上再加 CUDA Graphs（`mode="reduce-overhead"`），把每个 block 录成一张图，省掉逐个 kernel 启动的开销 |

日志末尾会给出每种方案的吞吐（crops/s）、GPU 利用率和显存，并写出 `fastest: dual_xxx`。

**背景**：这个模型只有 0.25M 参数，每一步是大量很小的 kernel。第一轮双卡只比单卡快约 1.2 倍，而 `torch.compile` 两次都在 DDP 下触发 inductor 的 stride 断言，退回了不编译的普通模式。新代码做了三处改动：
- 改为逐个 block 原地编译，DDP 包装的仍然是普通模型；
- 关掉 dynamo 的 `optimize_ddp`；
- 前 3 步编译失败时自动退回普通模式。

CPU 上两进程 DDP 已验证逐 block 编译能正常训练、checkpoint 能完整存取；**GPU 上的效果还没测过**，冒烟测试就是为了测这个。

然后按最快的那种方案跑正式预训练。例如 `graphs` 最快时：

```bash
python3 tools/build_s3t_smoke.py --mode pretrain --public --slug <你的账号>/hod26-s3t-mae-pretrain \
  --dual-minutes 600 --config "--batch 20 --crops-per-frame 10 --crop 128 --dim 64 --depth 4 --heads 4 --compile 1 --compile-mode reduce-overhead"
kaggle kernels push -p kernels/s3t_smoke/build
```

跑完后，把检测 kernel 的 `--mae-kernel` 指向你的新 notebook，重新生成检测 kernel：`python3 tools/s3t_round.py --mae-kernel <你的账号>/hod26-s3t-mae-pretrain`。

## 相关文件

- `src/hod26/s3t/`：`preprocess.py`（对齐、先验特征、uint8 编码）、`spectral.py`（光谱 Transformer 编码器）、`mae.py`（MAE 预训练）、`front.py`（接到 DETR 上的前端和侧注入）
- `tools/train_s3t_mae.py`：MAE 训练（torchrun，单卡或多卡）
- `tools/build_s3t_smoke.py`：MAE 冒烟测试 / 正式预训练 kernel 的生成器
- `tools/s3t_round.py`：检测微调 kernel 的生成器
- `tests/test_s3t.py`、`tests/test_s3t_detr.py`：CPU 测试
