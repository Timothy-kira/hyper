# S3T-DETR 接力：S3T-X 光谱编码器 + COCO RT-DETR + D-FINE，双卡 T4

这是一条**独立于 `README.md` 那条续训路线**的新模型线：从 COCO 开始训练，不接旧 checkpoint。README 里的 epoch-21 续训依然有效。

## 一句话

在 COCO 预训练的 RT-DETR-L 前面，加一个自己设计的**光谱编码器 S3T-X**（逐像素的通道协方差注意力），用 MAE 在官方 4000 张图上预训练。检测头在 RT-DETR 自己的 decoder 上加了 **D-FINE 的分布回归**，分类用 **MAL**。所有新加的层都零初始化，训练第 0 步就是原封不动的 COCO RT-DETR。

## 现状（2026-09-23 16:00 UTC）

| 环节 | 状态 | 在哪里 |
| --- | --- | --- |
| 速度探针（T4，1024²，真实 loss） | **完成**，S3T-X 通过，见下 | `zetaoxia/hod26-s3t-detr-probe` v5–v7 |
| MAE v3 预训练（S3T-X，修掉特征泄漏） | **完成**：6358 步，90.6 分钟，6 项检查全部 PASS | 公开 notebook `zetaoxia/hod26-s3t-mae-pretrain3`，输出 `s3t_mae/pretrain3_mae.pt` |
| 检测：S3T-X 前端 + D-FINE + MAL + 数据增强 | **代码完成**，CPU 上端到端跑通（渲染 → 训练 → 保存 best/last → 重载 → submission.csv） | `tools/s3t_round.py` 生成 `handoff/s3t/hod26_round.py` |
| 全流程冒烟（smoke_only） | **通过**（15:55，512 loader + 分阶段解冻版）：**0.550 s/it**（之前 1024 loader 是 0.682），加速全开，训练 / 验证 / best+last / fp16 验证 / 预测 / submission 全部走通，用时 217 秒 | `zetaoxia/hod26-s3t-detr` v1 |
| 正式检测训练 | **运行中**：15:56 UTC 开始，从 COCO 训练 44 epoch，512 loader，分阶段解冻，BN 冻结，预计约 10.4 小时 | `zetaoxia/hod26-s3t-detr` v2 |
| 上一次正式训练（1024 loader） | 按用户要求删掉了：第 1 个 epoch 用了 1055 秒，GPU 利用率只有 53% 和 61%，瓶颈在数据加载；而且没有保护预训练权重。见下面两节 | |

旧的 band-token 编码器（MAE v1/v2）已经停用。v2 那一轮是按用户要求中途删除的，原因见"为什么换成 S3T-X"。

## 为什么换成 S3T-X（速度探针）

旧编码器每个位置保存 16 个波段 token × 64 维 = 1024 个数，是一个像素原始信息（16 波段 × 3 个特征 = 48 个数）的 20 多倍。1024² 的输入有约 105 万个 token，一步训练里 3.3 s 花在它身上；attention 本身只占约 2% 的计算，所以换线性注意力解决不了问题，要减掉的是状态本身。

S3T-X 每个位置只保留一个 64 维向量，波段之间的内容相关混合改用**通道协方差注意力**（XCiT / Restormer 的 MDTA / MST++ 的 spectral-wise attention）：

```
XCA(Q, K, V) = V · softmax(K̂ᵀ Q̂ / τ)，Q̂、K̂ 沿像素做 L2 归一化
```

在**局部窗口**上算时，K̂ᵀQ̂ 就是局部背景的光谱协方差，softmax 后作用在 V 上，相当于一个可学习的局部白化，也就是 local RX 检测器背后的运算。弱类（灰色目标对灰色背景）在 CPU 扫描里正好只有相对局部背景才可分。所以前两层用 16×16 的窗口（约 32 原生像素，对应之前有效的 31px 环形窗口），后两层用全图，做场景级的归一化，比如光照。

探针结果（单张 T4，1024²，AMP，loss 和匈牙利匹配用 fp32，每卡 batch 2，同一个 session 里对比）：

| 配置 | s/step | 峰值显存 |
| --- | --- | --- |
| 纯 RT-DETR b2 | 0.323 | 10.5 GB* |
| 纯 RT-DETR b4 | 0.542 | 12.6 GB* |
| S3T-X eager + checkpoint b2 | 0.612 | 3.8 GB |
| S3T-X eager b2 | 0.587 | 5.6 GB |
| **S3T-X 编译 b2（正式训练用这个）** | **0.500** | 5.3 GB |
| S3T-X 编译 + checkpoint b4 | 0.945 | 7.1 GB |
| 旧 band-token 编码器 编译 b2 | 3.39 | 9.8 GB |

\* 纯 RT-DETR 排在第一个跑，峰值里含 cudnn benchmark 试算法时的工作区。

- S3T-X 比旧编码器**快 6.8 倍**，是纯 RT-DETR 的 1.55 倍。
- 没有 NaN。attention 实际走的是 `fmha_cutlassF/B`（mem-efficient 融合 kernel）。
### 第二轮探针：正式配置（S3T-X + D-FINE/MAL，编译，AMP，b2）

| 配置 | s/step | 峰值显存 |
| --- | --- | --- |
| 纯 RT-DETR + D-FINE/MAL，逐层算 loss | 0.422 | — |
| 纯 RT-DETR + D-FINE/MAL，批量 loss | 0.373 | — |
| S3T-X + D-FINE，逐层算 loss | 0.565 | 5.3 GB |
| S3T-X + D-FINE，逐层算 loss，**channels_last** | 0.586 | **12.5 GB** |
| **S3T-X + D-FINE，批量 loss（正式配置）** | **0.508** | 5.3 GB |

- **channels_last 要关**：ultralytics 8.4 在 CUDA 上默认把整个模型转成 channels_last。在这个模型上它更慢，峰值显存还翻了一倍多，离 14.6 GB 上限很近。S3T 这条线显式设成 `channels_last=False`。
- **D-FINE 的 loss 改成跨层批量计算**，去掉逐层循环和 `.any()` 的同步，数值和原实现一致（误差 1e-7）。原先它每步多花约 0.09 s，现在基本没有额外开销。
- **`deterministic=False`**：确定性模式会限制 cudnn 的算法选择，grid_sample 的反向也会变慢。
- **估算**：双卡每步 4 张图，训练集 2400 张加 1 份增强副本共 4800 张，**约 12.3 分钟一个 epoch**。45 个 epoch 约 9.2 小时，留约 1 小时余量，保证最后 3 个关闭 mosaic 的 epoch 一定能跑完。旧流程 52 个 epoch 到了 held-out 0.695。

## 模型结构

- **输入**：官方 X2Cube → log 辐亮度 → 每帧 P2–P98 缩放 → 逐波段亚像素对齐 → 16 通道 uint8。
- **S3T-X 编码器**（0.21M 参数，`src/hod26/s3t/xca.py`）：
  - 输入是每个波段的 3 个特征（level、shape、contrast），缩到 0.5 倍（回到原生像素尺度）。
  - stem 是 3×3 stride 2 卷积，接 4 个 XCA block（窗口 16、16、全图、全图），每个 block 是 XCA 加 GDFN（门控深度卷积 FFN），全程 channels_last。
  - **没有 BatchNorm**：MAE 时 75% 的位置被遮，BN 的统计量会被大量 0 带偏；检测时每卡只有 2 张图，BN 统计量也不稳。
- **接入 DETR**（`S3TXFront`，`src/hod26/s3t/front.py`），在两边网格本来就对齐的地方接：
  - 编码器输出是输入的 stride 4，和 RT-DETR 的 HGStem 输出（48 通道，stride 4）正好是同一个网格，用一个零初始化的 1×1 卷积（64→48）加上去。不再上采样到原图，也不再加宽全分辨率的 stem。
  - 用可学习的 stride-2 卷积金字塔（stride 8/16/32）给 P3/P4/P5 的输入投影做零初始化注入，代替之前的平均池化：几个像素大的目标，平均池化会把它和背景混在一起。
  - P3/P4 注入之前，先对 AIFI 的全局 token 做一次交叉注意力，从空间方向把全局信息送进光谱特征。
  - 16→3 投影照常送进 COCO stem。
- **检测头**：RT-DETR 自己的 decoder（COCO 权重，6 层，300 query，含 denoising），外加 D-FINE 分布回归，见下。

## MAE v3：修掉 v1/v2 的特征泄漏

v1/v2 的特征是在 mask **之前**算好的，有两处泄漏：

1. **shape = level 减去全部 16 个波段的均值**。只遮 1 个波段时，从同一像素任意一个可见波段的 level 和 shape 就能反推出这个均值，被遮波段的值可以**精确算出来**（测试里用这个公式复现了，误差 < 1e-4）。遮 k 个波段时，它们的和也泄漏了。光谱补全本来是要逼编码器学波段之间的关系，结果做算术就能解。
2. **contrast 的环形均值**包含被遮的像素，被遮区域的盒子和会传给旁边的可见像素。

v3 **先 mask，再算特征**（`observed_features`）：
- shape 只减**存在的波段**的均值；
- contrast 用**归一化卷积**，只对环形窗口里**看得见的像素**取均值；
- 被遮的波段额外加一个可学习的"缺失"向量，编码器能分清"这个波段被遮了"和"这个波段本来就是 0"。

检测前端用的是**同一个函数**，只是没有任何遮挡，所以微调时看到的输入和预训练时完全一致。

其他沿用 v2：
- 2×2 token 的掩码单位，遮挡比例 0.5 → 0.75 逐步加大；
- 每次遮 1–4 个波段，70% 是按波长连续的一段；
- 空间补全和光谱补全的 loss 分开算，并各有参照；
- 解码器是 2 层全局 attention 加正弦位置编码，再加 1 个窗口 XCA block。

**v3 结果**（双 T4，90.6 分钟，6358 步 × 每卡 48 个裁块 × 2 卡 ≈ 61 万个裁块，114.8 crops/s，每卡峰值 4.8 GB，GradScaler 2048–32768，无 NaN）。band/interp 和 spat/mean 都是模型误差 ÷ 平凡填充的误差，低于 1 就说明比平凡填充好：

| 步数（中位数） | loss | band/interp（光谱补全 vs 按波长线性插值） | spat/mean（空间补全 vs 可见均值） | grey（整块灰的比例） | 遮挡比例 |
| --- | --- | --- | --- | --- | --- |
| 0–200 | 2.01 | 1.01 | 1.12 | 0.00 | 0.55 |
| 500–1000 | 0.72 | 0.55 | 0.59 | 0.00 | 0.65 |
| 1500–2000 | 0.53 | 0.44 | 0.51 | 0.01 | 0.75 |
| 3000–4000 | 0.44 | 0.37 | 0.50 | 0.02 | 0.75 |
| 5800–6358 | **0.41** | **0.34** | **0.47** | 0.02 | 0.75 |

光谱补全的误差只有波长插值的 34%，而且特征已经没有泄漏，这是编码器靠光谱注意力学出来的，正是弱类需要的能力。v2 报的 0.60 里有泄漏的功劳。重建图（左：原图，中：遮掉 75% 后，右：重建，伪彩用 5/8/13 波段）：

![MAE v3 重建](../docs/s3t_mae3_recon.png)

石块的轮廓和明暗都能补出来，没有 v1 那种整块的灰色。

## 检测头与 loss：D-FINE 分布回归 + MAL + log-size L1

误差分析显示，匹配上的框中位 IoU 是 0.864，**扣分主要在 IoU 0.9 以上的高阈值**，所以重点放在框的精度上。

- **D-FINE 分布回归（FDR）**（`FDRDecoder`，`kernels/hod26_round/driver.py`）：
  - 原版 D-FINE 会把 box head 换成新的，这里**不替换**：COCO 预训练的 box head 保留，每层旁边再加一个零初始化的分布头。
  - 分布头对每条边预测 33 个偏移量上的分布；偏移量用 D-FINE 的非均匀 W(n)，越靠近 0 越密，最小一格约是边长的 2%。
  - 框 = 预训练 head 给出的框，每条边再移动分布的期望。
  - 第 0 步分布是均匀的，W 是奇函数，期望恰好是 0，所以输出和 COCO decoder **完全一致**（测试验证）。
- **FGL**（权重 0.15）：用匹配到的 GT 训练每层的分布（目标拆到相邻两个 bin 上，按 IoU 加权）。直接复用 loss 自己的匈牙利匹配，包括 denoising query。
- **GO-LSD / DDF**（权重 1.5）：把最后一层的框蒸馏给前面几层的分布，正负样本按 D-FINE 的方式平衡。
- **MAL**（DEIM）：匹配上的 query 全权重，目标是 IoU^1.5。低 IoU 的匹配会被教成低分，而不是像 VFL 那样被降权、在 loss 里被忽略。
- **log-size L1**：宽高在 log 空间做 L1。
- DEIM 的 Dense O2O 就是 mosaic，本来就开着。

## 保留 COCO 和 MAE 的预训练权重：分阶段解冻（2026-09-23 晚）

之前的正式训练**没有**保护预训练权重：
- 所有参数用同一个 lr。ultralytics 的 `optimizer=auto` 选了 AdamW，lr 4.55e-4；
- 骨干网络的 lr 是官方 RT-DETR 微调配置的 45 倍，官方配置里骨干是 0.1×，stem 冻结；
- BatchNorm 在 train 模式下，每卡只有 2 张图，COCO 的 running statistics 几百步之后就被覆盖了。

现在（`train.unfreeze`、`train.frozen_bn`）：

| 部分 | 包括 | 从第几个 epoch 开始（从 1 数） | lr 倍数 |
| --- | --- | --- | --- |
| head | 分类头和框回归头（decoder 6 层 + encoder query 选择），denoising 的类别 embedding | 1 | 1 |
| new | 所有零初始化的新层：S3T 融合、金字塔、P3/P4/P5 注入、D-FINE 分布头 | 1 | 1 |
| mixer | 16→3 波段投影 | 1 | 1 |
| decoder | 预训练的 decoder 层、input_proj、query_pos_head、enc_output | 3（第 3 个 epoch 0.5，第 4 个起 1） | 1 |
| neck | 预训练的 hybrid encoder（AIFI + CCFM） | 3（同上，爬升 2 个 epoch） | 1 |
| s3t_enc | MAE 预训练的 S3T-X 编码器 | 3（同上） | 1 |
| backbone | HGNetv2 stage 1–4（COCO） | 6（分 3 个 epoch 爬到 0.1） | 0.1 |
| stem、骨干里的 BN 缩放和偏移 | | 不训练 | 0 |

- **BatchNorm 全部固定用 COCO 的 running statistics**（`FrozenBatchNorm2d`，一直是 eval 模式）。它的缩放和偏移仍按所在部分的 lr 训练，骨干里的除外。
- 实现方式是**给每个 part 的 lr 乘一个倍数**，不用 `requires_grad`：
  - DDP 只在包装模型时给需要梯度的参数注册 hook；
  - ultralytics 看到被冻结的浮点参数，还会把 `requires_grad` 改回 True。
- 倍数只在 `optimizer.step()` 那一步生效（step 前的 pre-hook 乘上，post-hook 还原），warmup 和 scheduler 写入的 lr 都不受影响。
- 在 AdamW 下，倍数为 0 就是精确冻结：更新量和 decoupled weight decay 都乘以 lr。
- 第 1–5 个 epoch 本来就在 warmup 里，lr 还在往上爬，所以解冻是渐进的。
- 日志里：开头有一行 `staged unfreezing (part: first epoch, ramp epochs, LR x; params)`，每个 epoch 有一行 `unfreeze epoch N: head 1, new 1, ..., backbone 0.0333, stem 0, frozen_norm 0`。
- 测试 `tests/test_unfreeze.py`：
  - 每个参数恰好属于一个 part；
  - 每个阶段只有该动的部分在动，冻结的部分变化量精确为 0.0，动的幅度不超过 lr × 倍数；
  - lr 能还原，LambdaLR 正常；
  - optimizer state_dict 能往返（续训）；
  - BN 的统计量不变；
  - deepcopy、pickle、fuse 都正常。

## 数据加载：loader 用 512，GPU 上放大 2 倍（2026-09-23 晚）

上一次正式训练第 1 个 epoch 用了 1055 秒，两张卡的 GPU 利用率只有 53% 和 61%，瓶颈在 CPU（4 个 vCPU）。
- 16 通道的 mosaic 在 1024 下，单核每个样本 145 ms，在 512 下是 41 ms。
- 原始 cube 是 493×241，1024 本来就是把它放大约 2 倍，所以 loader 用 512 **不丢任何原始像素**。
- `S3TXFront(upsample=2)`：
  - 先做 16→3 投影，只把 3 个通道在 GPU 上双线性放大 2 倍（1×1 卷积和双线性插值可以交换）；
  - 检测器看到的仍然是 1024；
  - S3T 编码器直接读 512 的输入（scale × upsample = 1，不做任何缩放）。
- 训练、验证和预测都是 imgsz 512（`predict_kwargs` 会传入）。渲染好的数据集不用改，指纹和 imgsz 无关。
- 日志：`S3T-X input: loader at 1/2 of the detector's resolution ...`。

## 保留 best.pt 和 last.pt

输出目录（`/kaggle/working`）里会有：

- `final_last.pt`：**每个 epoch** 都复制一份，带 optimizer 和 EMA，可以直接续训。
- `final_best.pt`：**每次** held-out 分数创新高、ultralytics 重写 best.pt 时都复制一份。训练中途出异常也不会丢；训练结束后再换成去掉 optimizer 的版本。
- `final_results.csv`、`final_metrics.jsonl`：每个 epoch 的记录。

ultralytics 的 `runs/` 目录在 scratch 盘上，session 结束不会保存，所以要靠上面这几份副本。

## 这次修掉的问题（端到端 CPU 测试查出来的）

新增了 `tests/test_s3t_e2e.py`：用合成数据在 CPU 上把 `run_submission` 完整跑一遍。

- **"fused AdamW" 以前其实没开**：重建 optimizer 时，每个 param group 里已经带着 `fused=None`，它会覆盖构造函数传的 `fused=True`，但加速表照样显示 ON。现在逐个 group 设置，并且检查每个 group 都真的是 fused，否则直接报错。
- 以前 best.pt 只在训练正常返回后才复制到输出目录，现在每次更新都复制。

## 两次只在 GPU 上出现的问题，以及怎么在一开始就拦住（2026-09-23 下午）

1. **第一次正式训练**（11:28）：加速表里 fp32 loss、SDPA、cudnn.benchmark 显示 off，速度只有 1.2 it/s。
   - 原因：ultralytics 8.4 的 DDP 是**父进程**调用 `get_model` 建模，再用 cloudpickle 把模型传给 worker，worker 不会再调用 `get_model`。
   - 查下来 SDPA 和 fp32 loss 其实跟着模型带过去了，加速表只是读了一个父进程里的标志；但 **S3T 的 `torch.compile` 在 pickle 时丢了，`cudnn.benchmark` 是进程级设置，在 worker 里也是关的**。
   - 修复：新增 `ensure_accel()`，在 `setup_model` 里执行（worker 也会执行），重新打开这些加速；加速表改为读模型的实际状态，任何一项没开就报错。
2. **第二次**（12:00）：刚开始训练就崩了，`'DistributedDataParallel' object has no attribute 'init_criterion'`。
   - 原因：`build_optimizer` 调用时，模型已经被 DDP 包了一层。
   - 修复：`ensure_accel` 先解开包装。

**以后怎么在一开始就发现**：
- **本地**：`tests/test_s3t_e2e.py` 的 `ddp_worker_emulation` 按 ultralytics 的真实做法，在 CPU 上走一遍 worker 路径：父进程建模 → cloudpickle → worker 调用 `setup_model` → 单进程 gloo 的真 DDP → `build_optimizer` → 一步前向和反向。它会读 worker 端的加速表。
  - 上面两个问题，它在本地都能复现：去掉修复就失败，报错和 GPU 上一模一样。
- **GPU**：正式训练前先跑**冒烟**（`run_smoke`）：同一个 trainer、DDP、编译、AMP、D-FINE，用 40 帧训练 1 个 epoch，检查三件事：
  - worker 的加速表全部 ON；
  - 预热后的 s/it 不超过 0.82（探针 0.51 的 1.6 倍）；
  - 保存和验证正常。
  - 通过打印 `SMOKE ok: x s/it ...`，失败打印 `SMOKE FAILED: ...` 并停止。**约 3 分钟就能知道**，不用等渲染完、跑完一个 epoch。
- **渲染单独成 notebook**：`zetaoxia/hod26-s3t-render`，只用 CPU（`python3 tools/s3t_round.py --render-only --slug <账号>/hod26-s3t-render`）。
  - 输出 `ds_<key>/` 和 `render_manifest.json`，manifest 里的指纹覆盖通道、增强、重复采样和 train/val 划分。
  - 训练 kernel 挂上它（`--render-kernel <账号>/hod26-s3t-render`），直接用渲染好的数据，省掉约 6.5 分钟；配置对不上就在 preflight 报错。
- **数据加载提速**：mosaic 画布改为每个加载进程复用一块缓冲区，不再每个样本新分配 67 MB。单核每个样本从 173 ms 降到 106 ms，输出逐字节一致。加速表里有这一行。
- 每个 epoch 的日志多了两张卡的 GPU 利用率，用来判断瓶颈在 GPU 还是数据加载。

**额度**：`kaggle quota` 显示每个账号每周 30 小时。不要用 SDK 的 `to_json()` 去读额度：timedelta 序列化时会丢掉"天"，30 小时会显示成 21600s。

## 你要做的（等 MAE v3 跑完）

1. **检查额度**：约 11–12 小时 GPU。
2. **确认能打开数据集** `xishengfeng/hod26-planar`。
3. 生成并上传检测 kernel：
   ```bash
   python3 tools/s3t_round.py --render-only --slug <你的账号>/hod26-s3t-render   # 先渲染（CPU）
   python3 tools/s3t_round.py --slug <你的账号>/hod26-s3t-detr --arch xca \
       --mae-kernel zetaoxia/hod26-s3t-mae-pretrain3 --render-kernel <你的账号>/hod26-s3t-render
   kaggle kernels push -p kernels/s3t_detr/build
   ```
   或者在网页上 Import `handoff/s3t/hod26_round.py`，Input 里挂两项：`hod26-planar` 和 notebook `zetaoxia/hod26-s3t-mae-pretrain3`（公开）。
4. **Accelerator → GPU T4 x2**，**Internet → On**，Save & Run All。

### 前两分钟应该看到

```
preflight ok: S3T MAE encoder /kaggle/input/.../pretrain3_mae.pt (arch xca)
preflight passed
  box loss: GIoU, MAL (target IoU^1.5, positives at full weight), D-FINE FDR (+FGL 0.15, GO-LSD/DDF 1.5) on the pretrained decoder, log-space wh
  S3T encoder: MAE weights from ... (step N)
  S3T-X: cross-covariance attention, windows [16, 16, None, None], channels_last; no checkpoints; encoder trained
  S3T-X front: 0.21M-param encoder at 0.5x input scale, stride-4 output joined to HGStem's 48-ch output (zero-init 1x1); ...
  S3T blocks compiled in place: 4
  S3T-X input: loader at 1/2 of the detector's resolution; the encoder reads it directly (scale 1), ...
  BatchNorm: 122 layers frozen to COCO's running statistics (eval mode throughout; 2 images/card, no SyncBN)
  staged unfreezing (...): head (0, 0, 1.0) 0.97M; new 1.24M; ...; backbone (5, 3, 0.1) 13.48M; stem frozen ...
  acceleration table:
    ON   AMP fp16 (+GradScaler)
    ON   loss + Hungarian matching in fp32
    ON   RT-DETR attention via SDPA  (7 nn.MultiheadAttention swapped)
    ON   fused AdamW
    ON   cudnn.benchmark
    ON   mosaic canvas reuse (data loader)
    off  FlashAttention / TF32 / bf16  (not supported on T4 (sm75))
  2 GPU(s) visible; DDP across [0, 1], batch 4 (2/card)
```

- `PREFLIGHT FAILED: ... arch='tokens'; the run asks for s3t_arch='xca'`：挂成了旧的 MAE notebook，换成 `pretrain3`。
- `PREFLIGHT FAILED: ... no MAE checkpoint`：notebook 没挂上，**还没花额度**。
- `S3T encoder: NO pretrained weights`：**立刻停掉**并告诉我们。

## 数据增强（正式训练里都开着）

| 类型 | 做法 | 为什么 |
| --- | --- | --- |
| 离线·光谱 | **Savitzky-Golay 沿波长顺序**平滑（窗口 7，2 阶） | mosaic 编号 ≠ 波长顺序，`sg_chain` 按波长顺序平滑 |
| 离线·光谱 | 同类 SMOTE（α=0.3） | 同类目标之间插值，不改标签 |
| 离线·空间 | 超像素 CutMix（p=0.4） | 按超像素整块移植，不切断目标 |
| 离线 | 每帧额外生成 1 份增强副本，训练集变成 2 倍 | |
| 在线（ultralytics） | mosaic、水平翻转、缩放/平移 | 最后 3 个 epoch 关闭 mosaic |
| 不做 | HSV、mixup | 16 通道上 HSV 无意义；mixup 跨类插值 |

MAE 预训练**不做**任何增强：只用官方的 4000 帧原图裁块。验证集（600 张）永远不增强。

## 怎么判断它有没有用

和 README 那条线用同一把尺子：held-out 600 张上的 pycocotools mAP50-95。

| 参照 | held-out | LB |
| --- | --- | --- |
| 当前队伍最佳（epoch-52 微调） | 0.69528 | 0.62994 |

要比参照高出 **0.01 以上**才算有效（run-to-run 噪声约 0.0017）。日志末尾会打印每一类的 AP，重点看弱类（stone_block / people / e-bike / car）。

## 时间安排（截止 2026-09-24 16:00 UTC）

| 时间（UTC） | 内容 |
| --- | --- |
| 09:54 – 11:26 | MAE v3 预训练（90.6 分钟，公开 notebook）✅ |
| 15:56 – 约 02:30 | 检测训练 44 epoch（smoke_only 已经单独跑过，所以用 `--no-smoke`；时钟保护会在超时前停下，并保存 best/last） |
| 之后 | 同一个 session 里预测测试集，写 `submission.csv` |

还有余量：如果第一轮结束后时间和额度都够，可以挂上 `final_last.pt` 再续一轮。

## 历史：band-token 版本（MAE v1/v2）

- **v1**（`qwyi123/hod26-s3t-mae-pretrain`）和 **v2**：逐像素 16 个波段 token 的光谱自注意力。
- 检测时 OOM：一次 checkpoint 包住整个编码器，每张图约 13 GB。改成分块逐层 checkpoint 后能放下，但仍要 3.4 s/step，11 小时训不完。
- 这部分代码还留在 `src/hod26/s3t/spectral.py`、`mae.py`、`mae2.py`，`--arch tokens` 仍可生成。
- 细节见 git 历史里这份文档的旧版本。

## 相关文件

- `src/hod26/s3t/xca.py`：S3T-X 编码器
- `src/hod26/s3t/front.py`：`observed_features`（无泄漏特征）、`S3TXFront`（stem 融合、光谱金字塔）、注入与 AIFI 上下文
- `src/hod26/s3t/mae3.py`：MAE v3
- `kernels/hod26_round/driver.py`：`FDRDecoder`、`fdr_losses`（FGL/DDF）、MAL、best/last 保存
- `tools/train_s3t_mae.py`（`--mae-version 3`）、`tools/build_s3t_smoke.py`（`--mode pretrain`）：MAE 预训练
- `tools/s3t_round.py`：检测 kernel 生成器；`tools/build_s3t_detr_probe.py`：速度探针
- 测试：`tests/test_s3t_xca.py`（编码器、MAE v3、检测前端）、`tests/test_dfine.py`（FDR/FGL/DDF/MAL）、`tests/test_s3t_e2e.py`（端到端）、`tests/test_s3t_detr.py`、`tests/test_s3t.py`
