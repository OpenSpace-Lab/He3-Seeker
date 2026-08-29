# He-3 真值场生成方法

## 数据源

Kaguya (SELENE) Multiband Imager 矿物反演图，来源 Lemelin et al. (2015/2016)，
USGS Astrogeology 发布：

- FeO (wt%): Lunar_Kaguya_MIMap_MineralDeconv_FeOWeightPercent_50N50S.tif
- OMAT (光学成熟度指数): Lunar_Kaguya_MIMap_MineralDeconv_OpticalMaturityIndex_50N50S.tif
- 覆盖 50N-50S，分辨率约 59 m/pixel，Simple Cylindrical 投影，月球半径 1737.4 km
- https://planetarymaps.usgs.gov/mosaic/Lunar_MI_mineral_maps/

引用：Lemelin, M., Lucey, P.G., Song, E., Taylor, G.J. (2015). Lunar central peak
mineralogy and iron content using the Kaguya Multiband Imager: Reassessment of
the compositional structure of the lunar crust. J. Geophys. Res. Planets, 120, 869-887.

## 区域

Region A，高钛月海（Mare Tranquillitatis / Apollo 11 附近），中心 (0.674N, 23.473E)，
500x500 m。玄武岩月海，富含钛铁矿，Apollo 11 样品 TiO2 约 7-12 wt%。

Region B，高地（Descartes Highlands / Apollo 16 附近），中心 (8.973S, 15.500E)，
500x500 m。斜长岩高地，FeO 和 TiO2 含量都低。

## 处理流程

Step 1：通过 GDAL /vsicurl/ 远程读取 GeoTIFF（HTTP range request），按像素窗口
裁切目标区域，不下载整个数据集。

Step 2：FeO 转 TiO2，用 Lucey (1998) 公式

    TiO2 = 10^(0.06 * FeO - 0.54)   (wt%)

FeO <= 0 的像元设为 NaN。引用：Lucey, P.G., Blewett, D.T., Hawke, B.R. (1998).
Mapping the FeO and TiO2 content of the lunar surface with multispectral imagery.
J. Geophys. Res., 103(E2), 3679-3699.

Step 3：原始窗口约 9x9 像元，用 scipy.ndimage.zoom(order=3) 双三次插值到
500x500 网格（约 1 m/pixel）。NaN 像元先用均值填充再插值，再恢复掩膜。

Step 3b：给插值后的 FeO、OMAT 叠加一对相关高斯随机场，模拟 59 m 分辨率以下的
空间变异。TiO2 不直接扰动，而是从扰动后的 FeO 重新代入 Lucey 公式算出来，
所以 FeO 和 TiO2 的关系始终成立，不需要额外的物理约束步骤。

做法：
- 用 FFT 生成两个独立的零均值、单位方差高斯场 GRF1、GRF2（指数协方差，相关长度 25 m）
- 算出插值后 FeO 和 OMAT 的 Pearson 相关系数 r
- 2x2 Cholesky 变换：GRF_FeO = GRF1，GRF_OMAT = r*GRF1 + sqrt(1-r^2)*GRF2
- FeO' = FeO + 0.05*mean(FeO)*GRF_FeO，OMAT' 同理（扰动幅度取局部均值的 5%）
- TiO2' 从扰动后的 FeO' 重新代入 Lucey 公式
- FeO'、TiO2' clip >= 0，OMAT' clip >= 0.05（防止除零）
- 随机种子 42，两个区域共用同一个 RNG 依次往下取，所以各自拿到不同的随机场实现

TiO2 和 He-3 的正相关、OMAT 和 He-3 的负相关在每个像元上都是公式自带的，
因为只有 FeO 和 OMAT 是真正被扰动的量，TiO2 和 He-3 全是重新算出来的。

Step 4：He-3 浓度，用 Fa & Jin (2007) Eq.5，输入是上一步算出的 TiO2'、OMAT'

    C0 = 0.56 * (S_Ti * F / OMAT) + 1.62   (ppb)

F 按纬度查表（Fa & Jin 2007 Fig.1）：Region A F=0.50，Region B F=0.48。
结果 clip 到 [0, 15] ppb。引用：Fa, W., Jin, Y.-Q. (2007). Quantitative
estimation of helium-3 in lunar regolith. Icarus, 190(1), 15-23.

Step 5：每个区域出 5 张图（FeO / TiO2 / OMAT / He-3，外加一张 colorbar
自适应区域自身范围的 He-3 图，方便范围窄的区域看清纹理），并打印
corr(TiO2, He-3) 和 corr(OMAT, He-3) 作为一致性检查（前者应为正，后者应为负）。

## 输出文件

- data_raw/region{A,B}_feo.tif, region{A,B}_omat.tif：原始裁切
- data_processed/region{A,B}_tio2.tif：TiO2，原始分辨率
- data_processed/region{A,B}_{feo,omat,tio2}_interp.npy：插值后 500x500
- data_processed/region{A,B}_{feo,tio2,omat}_grf.npy：GRF 扰动后 500x500
- outputs/region{A,B}_he3_groundtruth.npy / .tif：He-3 真值场
- figures/region{A,B}_{feo,tio2,omat,he3,he3_adaptive}.png：可视化，共 10 张
- docs/METHOD.md：本文档
