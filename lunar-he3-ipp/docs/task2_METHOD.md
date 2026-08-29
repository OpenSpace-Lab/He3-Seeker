# Task 2 方法文档：He-3 真值场 + 地形数据

## 1. 区域选择理由

### Region T1：Taurus-Littrow (Apollo 17 着陆区)
- 中心坐标：(20.190°N, 30.772°E)，范围 500×500 m
- 位于月海玄武岩谷地，两侧为 North/South Massif 高地山体。500 m 范围内 TiO₂ 跨越 <1%（高地）到 >6%（谷地玄武岩），形成强烈的成分梯度。地形有显著陡坡（山体斜面可达 25–35°），构成天然不可通行区。
- 文献：Schmitt et al. (2017); Robinson & Jolliff (2002); Sun et al. (2021)

### Region T2：Hadley Rille (Apollo 15 着陆区)
- 中心坐标：(26.132°N, 3.633°E)，范围 500×500 m
- 月溪（蜿蜒沟谷，深约 300 m）与亚平宁山前月海交汇处。溪谷壁面陡峭，是天然地形障碍；同时月海玄武岩与溪谷暴露的分层提供了有趣的 He-3 成分对比。
- 文献：Spudis et al. (2011); Staid et al. (2011); Bell & Hawke (1995)

## 2. He-3 真值场处理

完全复用 Task 1 的五步流程（详见 lunar-he3-groundtruth/docs/METHOD.md）：
1. 远程裁切 FeO + OMAT（Kaguya MI 矿物反演图，~59 m/pixel）
2. FeO → TiO₂（Lucey 1998: TiO₂ = 10^(0.06×FeO - 0.54)）
3. 双三次插值至 500×500 网格
4. 联合多变量 GRF 扰动（ℓ=25 m, σ=5%）
5. He-3 计算（Fa & Jin 2007: C₀ = 0.56 × S_Ti × F / OMAT + 1.62 ppb）
   - T1 (20.19°N): F ≈ 0.50
   - T2 (26.13°N): F ≈ 0.51

## 3. DEM 数据源与处理

### 数据源
- SLDEM2015: LRO LOLA + Kaguya TC 融合 DEM
- 分辨率：512 ppd (~59 m/pixel)
- 来源：MIT LOLA Science Team
- URL: `https://imbrium.mit.edu/DATA/SLDEM2015/TILES/FLOAT_IMG/`
- 瓦片：`SLDEM2015_512_00N_30N_000_045_FLOAT` (PDS3 格式)
- 原始数据单位：km（相对 1737.4 km 参考球），转换为 m
- 覆盖范围：0°N–30°N, 0°E–45°E（涵盖 T1 和 T2）
- 投影：Equirectangular，中央经线 180°E

### 引用
> Barker, M.K., et al. (2016). A new lunar digital elevation model from the Lunar Orbiter Laser Altimeter and SELENE Terrain Camera. *Icarus*, 273, 346-355.

### 处理流程
1. 通过 GDAL `/vsicurl/` 远程读取 PDS 格式 DEM（HTTP range request）
2. 坐标转换：经纬度(°) → Equirectangular 投影坐标(m)
3. 按像素窗口裁切目标区域
4. km → m 单位转换
5. 双三次插值到 500×500 网格（与 He-3 场对齐）

## 4. 坡度计算方法

使用 `numpy.gradient` 计算 DEM 的空间梯度：

$$\text{slope} = \arctan\sqrt{\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2}$$

- 网格间距 Δx = Δy = 1 m（500 m / 500 pixels）
- 输出单位：度

## 5. 不可通行区判据

### 5.1 坡度判据（初版，实测不足）
- **主用阈值 20°**：月球巡视器（如玉兔号）的典型最大爬坡能力上限。NASA 和 ESA 的巡视器设计指标通常在 15°–25° 之间。
- **备用阈值 15°/25°**：保守/激进设定供论文对比。
- **实测问题**：源 DEM 仅 59 m/pixel，500 m 窗口内只有约 9×9 个真实采样点，双三次插值后的曲面过于平滑，T1/T2 最大坡度仅 ~10.4°/10.7°，三个坡度阈值下不可通行区均为 **0%**，坡度判据在此分辨率/尺度下失效。

### 5.2 局部相对高差判据（最终采用）
单纯坡度无法反映"凹坑陷穴"和"孤立小丘"这类局部障碍，因此叠加局部相对高差判据，与坡度判据取**逻辑或**：

**计算方法：**
1. 用 100×100 px（~100 m）滑动窗口均值滤波得到局部背景高程 $\bar{z}_{local}$
2. 局部相对高差：$\Delta z = z - \bar{z}_{local}$
3. 统计 $\Delta z$ 的标准差 $\sigma$，阈值取 $K\sigma$，$K=1.5$（数据驱动，非任意指定绝对值）

**判据：**
- **坑洞/凹地**（主要判据）：$\Delta z < -K\sigma$ → 不可通行，**不设面积上限**（凹坑无论大小都视为通行障碍，符合巡视器避坑逻辑）
- **孤立小丘**（次要判据）：$\Delta z > +K\sigma$ **且**连通区域面积 $< 5\%$ 总面积 → 不可通行（面积约束排除大范围平缓抬升地形，那只是区域趋势，不是障碍；只有面积小、局部突起的"小丘/陡包"才视为障碍，对应真实地貌中可能伴随岩块堆积的孤立高地）

**最终掩码** = 坡度掩码 OR 坑洞掩码 OR 小丘掩码，再做形态学闭操作（`scipy.ndimage.binary_closing`，3×3 结构元素）填小洞、连通边界。

**实测结果：**

| 区域 | relief σ | depth/height 阈值 | 坑洞占比 | 小丘占比(过滤后) | 最终不可通行占比 |
|------|----------|-------------------|----------|-------------------|------------------|
| T1 | 0.530 m | 0.795 m | 7.17% | 2.88% (2 个连通域，均保留) | 9.91% |
| T2 | 0.762 m | 1.143 m | 7.24% | 5.98% (6 个连通域，均保留) | 13.01% |

由于坡度分量贡献为 0%，三个坡度阈值版本（15°/20°/25°）的最终掩码完全相同（均由 relief 判据主导），这是当前 DEM 分辨率下的真实结果，如实记录，留待未来更高分辨率 DEM（如 NAC DTM，~1–5 m/px）或扩大区域范围时重新评估坡度判据的贡献。

## 6. 输出文件清单

| 路径 | 说明 |
|------|------|
| `data_raw/T{1,2}_feo.tif` | 原始 FeO 裁切 |
| `data_raw/T{1,2}_omat.tif` | 原始 OMAT 裁切 |
| `data_raw/T{1,2}_dem.tif` | 原始 DEM 裁切 (m) |
| `outputs/T{1,2}_he3.npy` | He-3 真值场 (500×500, ppb) |
| `outputs/T{1,2}_dem.npy` | DEM 高程场 (500×500, m) |
| `outputs/T{1,2}_slope.npy` | 坡度场 (500×500, degrees) |
| `outputs/T{1,2}_nontraversable.npy` | 不可通行掩码 (20°, uint8) |
| `outputs/T{1,2}_nontraversable_15deg.npy` | 不可通行掩码 (15°) |
| `outputs/T{1,2}_nontraversable_25deg.npy` | 不可通行掩码 (25°) |
| `figures/T{1,2}_he3.png` | He-3 热力图 |
| `figures/T{1,2}_dem.png` | DEM 高程图 (带等高线) |
| `figures/T{1,2}_slope.png` | 坡度图 |
| `figures/T{1,2}_nontraversable.png` | 不可通行区 B&W 图 |
| `figures/T{1,2}_overlay.png` | He-3 + 障碍边界叠加图 |
| `docs/task2_METHOD.md` | 本文档 |
