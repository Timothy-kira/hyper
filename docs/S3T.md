# S3T：面向 HOD26 高光谱目标检测的光谱 Transformer

> 参考 S3M 技术报告（ICPR 2026 "Beyond Visible Spectrum" 分类方案）的设计，按本比赛重新设计。
> S3M 的光谱 Mamba 流在这里**全部换成 Transformer**；空间部分不再自己搭，直接使用 COCO 预训练的 RT-DETR-L。

## 1 为什么用高光谱

本比赛的 18 类中有多组"外形相同、材质不同"的配对（apple / apple_plastic、egg / egg_plastic / egg_wood、car / car_toy），它们只能靠光谱区分。另一方面，我们的误差分解显示：模型已经找到了 98.3% 的目标，其中 99.9% 类别判对；剩下的差距全部在**框不够紧**，集中在 stone_block、people、e-bike、car 这四类。CPU 扫描发现，这四类的光谱几乎就是背景光谱乘上 0.68–0.90 的亮度系数（"灰对灰"）。所以模型既要能读懂光谱，又必须**保留绝对亮度和局部对比**，这是下文几处关键改动的出发点。

## 2 数据处理

- **传感器**：XIMEA MQ022HG-IM-SM4X4-VIS3，4×4 光谱滤光片阵列（SFA），460–600 nm，16 个窄带。
- **去马赛克**：使用官方 `X2Cube`，得到 (H, W, 16)。我们的实现 `hod26.cube.x2cube` 在测试中与官方函数逐位一致。
- **逐波段亚像素对齐（S3M 没有这一步）**：X2Cube 把原图位置 (4i + k//4, 4j + k%4) 的像素放到 cube[i, j, k]，所以第 k 个波段的实际采样点相对 cube 像素偏移了 (k//4, k%4)/4 个像素，波段之间最多相差 3/4 个像素。在目标边缘，这会让"一个像素的光谱"其实混进了相邻像素。我们按这个已知偏移做线性重采样，把 16 个波段对齐到 4×4 块的中心。测试中，对齐前同一线性梯度场上 16 个波段读数相差超过 1，对齐后相差不到 0.001。
- **波长顺序**：mosaic 编号并不是波长顺序。由相邻波段相关性恢复出的顺序是 `[15,13,14,12,10,8,9,7,6,1,0,2,3,5,4,11]`，band 4 和 band 11 错位。凡是需要"波长相邻"的地方（连续光谱掩码、光谱梯度损失、S-G 平滑）都按这个顺序来。
- **归一化**：先取 log1p 辐亮度，再把每帧的 P2–P98 线性映射到 0–1。只做缩放、**不截断**，这样帧内所有亮度比例都保留下来。数据是辐亮度而不是反射率，每帧缩放可以消除曝光差异。
- **只用官方图片**：MAE 用的是官方全部 4000 帧（训练 3000 + 测试 1000），不读标签，也不含任何增强生成的图片。

## 3 相关工作（与 S3M 报告相同，略）

3D CNN → Transformer（SST、HyperSIGMA）→ Mamba（S²Mamba）。**对本比赛而言**，光谱序列只有 16 个 token，Mamba 的核心卖点（线性复杂度）用不上；自注意力的开销可以忽略，天然是双向的，而且不需要额外编译 `mamba-ssm` 的 CUDA 扩展。这就是 S3T 用 Transformer 替换 Mamba 的理由。

## 4 模型架构

### 4.1 先验特征（替代 S3M 的植被指数）

460–600 nm 里既没有红边也没有 NIR，NDVI 这一类指数根本算不出来。我们改为给每个波段附上 3 个特征，这些特征都经过 CPU 扫描实测：

| 特征 | 定义 | 作用 |
| --- | --- | --- |
| level | 归一化后的 log 辐亮度 | 保留绝对亮度，这是灰色类别唯一剩下的线索 |
| shape | level 减去该像素 16 个波段的均值 | 去掉亮度后的光谱形状 |
| contrast | level 减去 63px 环形窗口（中心 31px 挖空）的背景均值 | 局部对比。窗口要挖空中心，否则背景均值里会混进目标本身 |

第二轮扫描中，`shape + lr_ann63` 组合把弱类的 AUC 从 0.711 提到 0.728、边缘 d′ 从 0.26 提到 0.28，其余类别的 AUC 从 0.895 提到 0.922、边缘 d′ 从 0.93 提到 1.02，是所有测过的变换里最均衡的。

### 4.2 删掉 IWS（S3M 在这里有问题）

S3M 的 IWS 模块给每个通道乘一个与输入内容无关的系数 α ∈ (0, 1)，紧接着是逐波段的线性卷积 PatchEmbed 和 LayerNorm。线性卷积会把 α 原样带过去，LayerNorm 再把尺度归一化掉，α 的作用几乎被完全抵消。另外，传感器固定时波段也固定，"Fourier(λ) → MLP"实际上就是学 16 个常数。S3T 不用 IWS，波段身份交给一个可学习的光谱位置编码（S3M 4.4 节）。

### 4.3 Stem（替代 S3M 的 PatchEmbed）

- 所有波段共享一个重叠卷积：Conv 3×3，stride 2 → BatchNorm → GELU → Conv 1×1。输出为 (B, S, C=16, D=64)。
- **不在每个 token 上做 LayerNorm**。如果一个 patch 亮度均匀、值为 b，它的 embedding 是 b·v，LayerNorm 之后 b 就被消掉了，而亮度正是灰色类别的关键线索。测试中专门检查了"整体亮度翻倍时输出会改变"这一点。
- 用 stride 2 而不是 stride 4 的 patchify：cube 本身已经是原图缩小 4 倍的结果，再用 stride 4，11×20 px 的 stone_block 只剩 3×5 个 token。

### 4.4 光谱 Transformer 块（替代 SpectralMambaBlock）

```
x: (B, S, C, D) → (B·S, C, D)
x ← x + γ1 · MHSA(LN(x))      # 在 16 个波段 token 上做自注意力，4 头，SDPA 实现
x ← x + γ2 · MLP(LN(x))       # 4D 中间维度
```

- γ 是 LayerScale（初值 0.1）。LayerNorm 只放在残差分支里，残差主干保留绝对亮度。
- 不同空间位置之间互不依赖，所以 MAE 预训练时**只需要计算没被遮住的位置**。
- 每两个光谱块后面接一个 **SpatialMix**：各波段共享的 7×7 depthwise 卷积，再接一个 MLP。被遮住的位置按 0 输入（FCMAE 的稀疏卷积做法），可见 token 读不到被遮住的内容。这一点有专门的测试。

### 4.5 光谱聚合与检测接口

- **SpectralPool**：按内容对 16 个波段 token 做注意力加权求和，得到 (B, D, H/2, W/2)。
- **主通路**：一个固定形式的 16→3 投影（用 LDA 初始化），加上零初始化的 1×1 卷积把 S3T 特征投成 3 通道并上采样，两者相加后送入 COCO 预训练的 RT-DETR stem。
- **侧注入**：S3T 特征池化到 P3、P4、P5 三个尺度，经零初始化的 1×1 卷积，加到 hybrid encoder 的三个输入投影上（第 19、14、10 层）。
- S3M 的融合门写的是"初始化为 0，所以训练初期影响为零"，但它用的是 sigmoid(0)，实际等于 0.5。S3T 的零初始化是真正的恒等：训练第 0 步，检测器看到的只是那个 16→3 投影。
- ultralytics 会把 493×241 的 cube 放大到 imgsz=1024 左右，所以编码器先把输入缩到 0.5 倍，回到 MAE 预训练时的原生像素尺度，显存也降到四分之一。编码器用梯度检查点，内部用 fp16。

## 5 训练

### 5.1 MAE 预训练（只训练光谱编码器，DETR 不参与）

- **Tube 空间掩码**：以 4×4 个 token（原生 8×8 px）为单位，遮掉 75%，16 个波段同时遮住。
- **连续光谱掩码**：按波长顺序遮掉连续 2 个波段（15%）。
- **损失**：逐 token 归一化 MSE（除以 target 的标准差再加 0.05，防止平坦区域数值爆炸）+ 0.5 × 未归一化 L1 + 0.5 × 光谱梯度损失（沿波长顺序做一阶差分）。S3M 报告自己也指出，只用归一化 MSE 时模型可以靠输出平坦光谱来刷低 loss；加上 L1 就堵住了这个漏洞。
- **解码器**：1 层 SpatialMix + 1 层光谱块，宽度 32，MLP 扩展比 2。预训练结束后丢弃。
- **双卡 T4 加速**：
  - DDP（NCCL）。
  - fp16 autocast + GradScaler（T4 不支持 bf16）。
  - SDPA 实际走 mem-efficient 后端。FlashAttention 需要 sm80 以上，T4 是 sm75，用不了。
  - 编码器只计算可见位置。
  - fused AdamW、channels_last、pin_memory 并常驻 DataLoader worker。
  - `torch.compile`：第一轮对 DDP 包装后的整个模型编译，两次都触发了 inductor 的 stride 断言，退回普通模式（退回后吞吐约 51 crops/s，冒烟测试中编译模式约 54 crops/s）。现已改为逐个 block 原地编译，并关掉 dynamo 的 `optimize_ddp`，可选再加 CUDA Graphs（`--compile-mode reduce-overhead`）来合并小 kernel、省掉 kernel 启动开销；冒烟测试 kernel 会在双卡上对比不编译、编译融合、编译融合加 CUDA Graphs 这三种方案的吞吐和 GPU 利用率。

### 5.2 检测微调

- COCO 预训练的 RT-DETR-L，fp32（ultralytics 的建议：RT-DETR 的匈牙利匹配在 AMP 下可能出现 NaN），imgsz 1024，每卡 batch 2，双卡 DDP，nbs=64。
- 数据增强见 `handoff/S3T.md`：离线的 S-G 平滑（沿波长顺序）、同类 SMOTE、超像素 CutMix，每帧额外生成 1 份副本；在线的 mosaic、翻转、缩放。
- 600 张验证集不参与训练，用 pycocotools 计算 mAP50-95 作为唯一的比较标准。

## 6 不足

- 检测微调代码只在 CPU 上测过（安装、前向、反向、深拷贝、保存与加载、融合都覆盖了），还没有在 GPU 上跑过。
- 计划中"对齐阶段冻结 DETR"这一步没有实现。现在靠零初始化和从头开始的完整微调来代替。
- 计划中给检测头加 P2 这一步没有实现，这需要改 RT-DETR 的解码器结构，风险太大。
- 逐 block 编译加 CUDA Graphs 只在 CPU 上验证过能正常训练，在 GPU 上的提速效果要等下一个账号跑冒烟测试确认。

## 引用

S3M 报告；He et al., *Masked Autoencoders*（2022）；Tong et al., *VideoMAE*（2022）；Woo et al., *ConvNeXt V2 / FCMAE*（2023）；Xiao et al., *Early Convolutions Help Transformers See Better*（2021）；Zhao et al., *RT-DETR*（2024）；Touvron et al., *CaiT / LayerScale*（2021）。
