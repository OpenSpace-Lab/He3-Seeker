"""
Generate He-3 ground truth fields for two lunar regions at 59m resolution.
Phase 1: Mare Tranquillitatis (Region A) and Descartes Highlands (Region B).
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom as scipy_zoom

# GDAL HTTP tuning for remote /vsicurl/ reads (legitimate, portable settings)
os.environ.setdefault('GDAL_HTTP_TIMEOUT', '120')
os.environ.setdefault('GDAL_HTTP_MAX_RETRY', '5')
os.environ.setdefault('GDAL_HTTP_RETRY_DELAY', '5')
os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif')
os.environ.setdefault('VSI_CACHE', 'TRUE')
os.environ.setdefault('VSI_CACHE_SIZE', '67108864')

from osgeo import gdal
gdal.UseExceptions()

import rasterio
from rasterio.transform import from_bounds as transform_from_bounds
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DIR_RAW = ROOT / "data_raw"
DIR_PROC = ROOT / "data_processed"
DIR_OUT = ROOT / "outputs"
DIR_FIG = ROOT / "figures"
DIR_DOC = ROOT / "docs"
for d in [DIR_RAW, DIR_PROC, DIR_OUT, DIR_FIG, DIR_DOC]:
    d.mkdir(parents=True, exist_ok=True)

# ── Remote URLs ────────────────────────────────────────────────────────
URL_FEO  = "/vsicurl/https://planetarymaps.usgs.gov/mosaic/Lunar_MI_mineral_maps/Lunar_Kaguya_MIMap_MineralDeconv_FeOWeightPercent_50N50S.tif"
URL_OMAT = "/vsicurl/https://planetarymaps.usgs.gov/mosaic/Lunar_MI_mineral_maps/Lunar_Kaguya_MIMap_MineralDeconv_OpticalMaturityIndex_50N50S.tif"

# ── Region definitions ─────────────────────────────────────────────────
HALF_SIZE_DEG = 0.0165 / 2

REGIONS = {
    "regionA": {
        "label": "Region A: Mare Tranquillitatis (Apollo 11)",
        "center_lat": 0.674,
        "center_lon": 23.473,
        "solar_flux_F": 0.50,
    },
    "regionB": {
        "label": "Region B: Descartes Highlands (Apollo 16)",
        "center_lat": -8.973,
        "center_lon": 15.500,
        "solar_flux_F": 0.48,
    },
}

TARGET_GRID = 500
MOON_CRS_WKT = None


def print_stats(name, arr):
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        sys.stdout.write("  " + name + ": ALL NaN (" + str(arr.size) + " pixels)\n")
        sys.stdout.flush()
        return
    sys.stdout.write("  " + name + ": min=" + str(round(float(valid.min()), 4))
                     + ", max=" + str(round(float(valid.max()), 4))
                     + ", mean=" + str(round(float(valid.mean()), 4))
                     + ", valid_pixels=" + str(valid.size) + "/" + str(arr.size) + "\n")
    sys.stdout.flush()


def crop_remote_gdal(url, lon_min, lat_min, lon_max, lat_max, out_path):
    global MOON_CRS_WKT
    t0 = time.time()
    fname = url.split('/')[-1]
    sys.stdout.write("    Opening: ..." + fname + "\n"); sys.stdout.flush()
    ds = gdal.Open(url)
    gt = ds.GetGeoTransform()
    if MOON_CRS_WKT is None:
        MOON_CRS_WKT = ds.GetProjection()
        px_deg = gt[1]
        px_m = px_deg * (np.pi / 180) * 1737400
        sys.stdout.write("    CRS loaded. Pixel: " + str(round(px_deg, 6))
                         + " deg = " + str(round(px_m, 1)) + " m\n")
        sys.stdout.flush()

    col_off = int((lon_min - gt[0]) / gt[1])
    row_off = int((lat_max - gt[3]) / gt[5])
    ncols = int(np.ceil((lon_max - lon_min) / gt[1]))
    nrows = int(np.ceil((lat_max - lat_min) / abs(gt[5])))
    sys.stdout.write("    Window: col=" + str(col_off) + ", row=" + str(row_off)
                     + ", " + str(ncols) + "x" + str(nrows) + " px\n")
    sys.stdout.flush()

    band = ds.GetRasterBand(1)
    sys.stdout.write("    Reading...\n"); sys.stdout.flush()
    data = band.ReadAsArray(col_off, row_off, ncols, nrows).astype(np.float64)
    elapsed = time.time() - t0
    sys.stdout.write("    Read OK in " + str(round(elapsed, 1)) + "s, shape="
                     + str(data.shape) + "\n")
    sys.stdout.flush()

    new_gt = (
        gt[0] + col_off * gt[1],
        gt[1], 0,
        gt[3] + row_off * gt[5],
        0, gt[5]
    )
    drv = gdal.GetDriverByName('GTiff')
    out_ds = drv.Create(str(out_path), ncols, nrows, 1, gdal.GDT_Float64)
    out_ds.SetGeoTransform(new_gt)
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(data)
    out_ds.FlushCache()
    out_ds = None
    ds = None
    return data


def save_geotiff(arr, bounds, out_path, crs_wkt=None):
    h, w = arr.shape
    transform = transform_from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], w, h)
    profile = {
        'driver': 'GTiff',
        'height': h,
        'width': w,
        'count': 1,
        'dtype': 'float64',
        'transform': transform,
    }
    if crs_wkt:
        profile['crs'] = rasterio.crs.CRS.from_wkt(crs_wkt)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(arr, 1)


def make_figure(arr, title, cbar_label, out_path, cmap_name='viridis', vmin=None, vmax=None):
    """Generate a heatmap PNG using PIL + matplotlib colormaps (bypasses Agg backend)."""
    h, w = arr.shape
    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))

    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mapped = cmap(norm(arr))
    nan_mask = np.isnan(arr)
    mapped[nan_mask] = [0.2, 0.2, 0.2, 1.0]
    img_arr = (mapped[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(img_arr)

    cbar_w = 25
    cbar_h = h
    cbar_data = np.linspace(1, 0, cbar_h).reshape(-1, 1).repeat(cbar_w, axis=1)
    cbar_mapped = cmap(cbar_data)[:, :, :3]
    cbar_arr = (cbar_mapped * 255).astype(np.uint8)
    cbar_img = Image.fromarray(cbar_arr)

    top_margin = 55
    left_margin = 10
    gap = 15
    label_w = 80
    total_w = left_margin + w + gap + cbar_w + label_w
    total_h = top_margin + h + 30
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    canvas.paste(img, (left_margin, top_margin))
    canvas.paste(cbar_img, (left_margin + w + gap, top_margin))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font

    lines = title.split('\n')
    for i, line in enumerate(lines):
        draw.text((left_margin, 5 + i * 18), line, fill=(0, 0, 0), font=font)

    cx = left_margin + w + gap + cbar_w + 5
    draw.text((cx, top_margin - 2), str(round(vmax, 2)), fill=(0, 0, 0), font=font_sm)
    draw.text((cx, top_margin + h - 14), str(round(vmin, 2)), fill=(0, 0, 0), font=font_sm)
    mid = (vmin + vmax) / 2
    draw.text((cx, top_margin + h // 2 - 7), str(round(mid, 2)), fill=(0, 0, 0), font=font_sm)
    draw.text((left_margin + w + gap, top_margin + h + 5), cbar_label, fill=(0, 0, 0), font=font_sm)

    draw.text((left_margin, top_margin + h + 5), "500x500 m, ~1 m/px", fill=(100, 100, 100), font=font_sm)

    canvas.save(str(out_path), dpi=(150, 150))
    sys.stdout.write("  Saved figure: " + out_path.name + "\n"); sys.stdout.flush()


# ── GRF helpers (defined once, outside region loop) ────────────────

def generate_grf_fft(shape, corr_length, rng):
    """Zero-mean, unit-variance GRF via FFT (exponential covariance). No LAPACK."""
    ny, nx = shape
    fy = np.fft.fftfreq(ny).reshape(-1, 1)
    fx = np.fft.fftfreq(nx).reshape(1, -1)
    freq_r = np.sqrt(fy**2 + fx**2)
    freq_r[0, 0] = 1e-10
    k_scaled = 2 * np.pi * freq_r * corr_length
    psd = (corr_length**2) / np.power(1 + k_scaled**2, 1.5)
    psd[0, 0] = 0
    amplitude = np.sqrt(psd)
    noise = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    field_fft = amplitude * np.fft.fft2(noise)
    field = np.real(np.fft.ifft2(field_fft))
    field = field / (field.std() + 1e-12)
    return field


def manual_pearson(a, b):
    """Pearson correlation without np.corrcoef (LAPACK-free)."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    return float((a_c * b_c).sum() / (np.sqrt((a_c**2).sum() * (b_c**2).sum()) + 1e-30))


# Single RNG — advances across regions so each gets a different GRF realization
grf_rng = np.random.default_rng(42)

# ══════════════════════════════════════════════════════════════════════
sys.stdout.write("=" * 60 + "\n"); sys.stdout.flush()
sys.stdout.write("He-3 Ground Truth Generation - Phase 1\n"); sys.stdout.flush()
sys.stdout.write("=" * 60 + "\n"); sys.stdout.flush()

for rname, rinfo in REGIONS.items():
    sys.stdout.write("\n" + "=" * 60 + "\n"); sys.stdout.flush()
    sys.stdout.write("Processing " + rinfo['label'] + "\n"); sys.stdout.flush()
    sys.stdout.write("=" * 60 + "\n"); sys.stdout.flush()

    clat = rinfo["center_lat"]
    clon = rinfo["center_lon"]
    lon_min = clon - HALF_SIZE_DEG
    lon_max = clon + HALF_SIZE_DEG
    lat_min = clat - HALF_SIZE_DEG
    lat_max = clat + HALF_SIZE_DEG
    bounds = (lon_min, lat_min, lon_max, lat_max)

    sys.stdout.write("  Center: (" + str(clat) + ", " + str(clon) + ")\n")
    sys.stdout.write("  Bounds: lon [" + str(round(lon_min, 5)) + ", "
                     + str(round(lon_max, 5)) + "], lat ["
                     + str(round(lat_min, 5)) + ", " + str(round(lat_max, 5)) + "]\n")
    sys.stdout.flush()

    # ── Step 1 ─────────────────────────────────────────────────────
    sys.stdout.write("\n[Step 1] Remote crop FeO and OMAT...\n"); sys.stdout.flush()

    feo_path = DIR_RAW / (rname + "_feo.tif")
    omat_path = DIR_RAW / (rname + "_omat.tif")

    sys.stdout.write("  Cropping FeO...\n"); sys.stdout.flush()
    feo_raw = crop_remote_gdal(URL_FEO, lon_min, lat_min, lon_max, lat_max, feo_path)
    print_stats("FeO raw (wt%)", feo_raw)

    sys.stdout.write("  Cropping OMAT...\n"); sys.stdout.flush()
    omat_raw = crop_remote_gdal(URL_OMAT, lon_min, lat_min, lon_max, lat_max, omat_path)
    print_stats("OMAT raw", omat_raw)

    # ── Step 2 ─────────────────────────────────────────────────────
    sys.stdout.write("\n[Step 2] FeO -> TiO2 (Lucey 1998)...\n"); sys.stdout.flush()
    feo = feo_raw.copy()
    feo[feo <= 0] = np.nan
    tio2 = np.power(10, 0.06 * feo - 0.54)
    print_stats("TiO2 (wt%)", tio2)

    tio2_path = DIR_PROC / (rname + "_tio2.tif")
    save_geotiff(tio2, bounds, tio2_path, MOON_CRS_WKT)
    sys.stdout.write("  Saved: " + tio2_path.name + "\n"); sys.stdout.flush()

    # ── Step 3 ─────────────────────────────────────────────────────
    sys.stdout.write("\n[Step 3] Bicubic interpolation " + str(feo_raw.shape)
                     + " -> " + str(TARGET_GRID) + "x" + str(TARGET_GRID) + "...\n")
    sys.stdout.flush()

    def interp_layer(arr, target=TARGET_GRID):
        h, w = arr.shape
        zf_y = target / h
        zf_x = target / w
        mask = np.isnan(arr)
        arr_filled = arr.copy()
        if mask.any():
            arr_filled[mask] = np.nanmean(arr)
        result = scipy_zoom(arr_filled, (zf_y, zf_x), order=3)
        if mask.any():
            mask_up = scipy_zoom(mask.astype(float), (zf_y, zf_x), order=0) > 0.5
            result[mask_up] = np.nan
        return result

    feo_interp = interp_layer(feo)
    omat_interp = interp_layer(omat_raw)
    tio2_interp = interp_layer(tio2)

    print_stats("FeO interp", feo_interp)
    print_stats("OMAT interp", omat_interp)
    print_stats("TiO2 interp", tio2_interp)

    np.save(str(DIR_PROC / (rname + "_feo_interp.npy")), feo_interp)
    np.save(str(DIR_PROC / (rname + "_omat_interp.npy")), omat_interp)
    np.save(str(DIR_PROC / (rname + "_tio2_interp.npy")), tio2_interp)
    sys.stdout.write("  Saved interpolated .npy files\n"); sys.stdout.flush()

    # ── Step 3b: Joint GRF on FeO & OMAT, then re-derive TiO2 ─────
    sys.stdout.write("\n[Step 3b] Correlated GRF on FeO & OMAT (L=25m, sigma=5%)...\n")
    sys.stdout.flush()

    CORR_LENGTH = 25.0   # pixels (~25 m at 1 m/px)
    SIGMA_FRAC  = 0.05   # 5% of local mean
    grid_shape = (TARGET_GRID, TARGET_GRID)

    # FeO-OMAT correlation from interpolated fields
    flat_f = feo_interp.ravel()
    flat_o = omat_interp.ravel()
    valid_mask = ~(np.isnan(flat_f) | np.isnan(flat_o))
    r_feo_omat = manual_pearson(flat_f[valid_mask], flat_o[valid_mask])
    sys.stdout.write("  corr(FeO, OMAT) from interp = " + str(round(r_feo_omat, 4)) + "\n")
    sys.stdout.flush()

    # Generate 2 independent GRFs
    sys.stdout.write("  Generating GRFs...\n"); sys.stdout.flush()
    grf1 = generate_grf_fft(grid_shape, CORR_LENGTH, grf_rng)
    grf2 = generate_grf_fft(grid_shape, CORR_LENGTH, grf_rng)
    sys.stdout.write("    GRF1 std=" + str(round(grf1.std(), 4))
                     + ", GRF2 std=" + str(round(grf2.std(), 4)) + "\n")
    sys.stdout.flush()

    # Manual 2x2 Cholesky: [[1, r], [r, 1]] -> L = [[1, 0], [r, sqrt(1-r^2)]]
    r = r_feo_omat
    grf_feo  = grf1
    grf_omat = r * grf1 + np.sqrt(max(1.0 - r * r, 0.0)) * grf2

    sys.stdout.write("  Correlated GRFs: corr(grf_feo, grf_omat) = "
                     + str(round(manual_pearson(grf_feo.ravel(), grf_omat.ravel()), 4))
                     + " (target " + str(round(r, 4)) + ")\n")
    sys.stdout.flush()

    # Perturb FeO and OMAT
    feo_mean  = float(np.nanmean(feo_interp))
    omat_mean = float(np.nanmean(omat_interp))

    feo_perturbed  = feo_interp  + SIGMA_FRAC * feo_mean  * grf_feo
    omat_perturbed = omat_interp + SIGMA_FRAC * omat_mean * grf_omat

    # Physical bounds
    feo_perturbed  = np.clip(feo_perturbed, 0, None)
    omat_perturbed = np.clip(omat_perturbed, 0.05, None)

    # Re-derive TiO2 from perturbed FeO (Lucey 1998) — physics consistency by construction
    feo_for_tio2 = feo_perturbed.copy()
    feo_for_tio2[feo_for_tio2 <= 0] = np.nan
    tio2_perturbed = np.power(10, 0.06 * feo_for_tio2 - 0.54)
    tio2_perturbed = np.clip(tio2_perturbed, 0, None)

    # Propagate NaN
    nan_mask = np.isnan(feo_interp) | np.isnan(omat_interp)
    feo_perturbed[nan_mask]  = np.nan
    tio2_perturbed[nan_mask] = np.nan
    omat_perturbed[nan_mask] = np.nan

    sys.stdout.write("  After GRF perturbation:\n"); sys.stdout.flush()
    print_stats("FeO perturbed", feo_perturbed)
    print_stats("TiO2 re-derived", tio2_perturbed)
    print_stats("OMAT perturbed", omat_perturbed)

    np.save(str(DIR_PROC / (rname + "_feo_grf.npy")), feo_perturbed)
    np.save(str(DIR_PROC / (rname + "_tio2_grf.npy")), tio2_perturbed)
    np.save(str(DIR_PROC / (rname + "_omat_grf.npy")), omat_perturbed)
    sys.stdout.write("  Saved perturbed .npy files\n"); sys.stdout.flush()

    # ── Step 4: He-3 from perturbed inputs ─────────────────────────
    sys.stdout.write("\n[Step 4] He-3 from GRF-perturbed inputs (Fa & Jin 2007, F="
                     + str(rinfo['solar_flux_F']) + ")...\n")
    sys.stdout.flush()
    F = rinfo["solar_flux_F"]

    he3 = 0.56 * (tio2_perturbed * F / omat_perturbed) + 1.62
    he3 = np.clip(he3, 0, 15)
    print_stats("He-3 (ppb)", he3)

    # Physical consistency check (LAPACK-free)
    valid_h = ~(np.isnan(tio2_perturbed.ravel()) | np.isnan(he3.ravel()))
    r_ti_he3 = manual_pearson(tio2_perturbed.ravel()[valid_h], he3.ravel()[valid_h])
    r_omat_he3 = manual_pearson(omat_perturbed.ravel()[valid_h], he3.ravel()[valid_h])
    sys.stdout.write("  Physical consistency check:\n")
    sys.stdout.write("    corr(TiO2, He-3)  = " + str(round(r_ti_he3, 4))
                     + " (expect > 0)\n")
    sys.stdout.write("    corr(OMAT, He-3)  = " + str(round(r_omat_he3, 4))
                     + " (expect < 0)\n")
    sys.stdout.flush()

    np.save(str(DIR_OUT / (rname + "_he3_groundtruth.npy")), he3)
    save_geotiff(he3, bounds, DIR_OUT / (rname + "_he3_groundtruth.tif"), MOON_CRS_WKT)
    sys.stdout.write("  Saved: " + rname + "_he3_groundtruth.npy + .tif\n")
    sys.stdout.flush()

    # ── Step 5 ─────────────────────────────────────────────────────
    sys.stdout.write("\n[Step 5] Generating figures...\n"); sys.stdout.flush()
    coord_str = "Center: (" + str(clat) + "N, " + str(clon) + "E)"

    make_figure(
        feo_perturbed,
        rinfo['label'] + "\nFeO (GRF) | " + coord_str,
        "FeO (wt%)",
        DIR_FIG / (rname + "_feo.png"),
        cmap_name='hot',
    )
    make_figure(
        tio2_perturbed,
        rinfo['label'] + "\nTiO2 (GRF) | " + coord_str,
        "TiO2 (wt%)",
        DIR_FIG / (rname + "_tio2.png"),
        cmap_name='inferno',
    )
    make_figure(
        omat_perturbed,
        rinfo['label'] + "\nOMAT (GRF) | " + coord_str,
        "OMAT (index)",
        DIR_FIG / (rname + "_omat.png"),
        cmap_name='cividis',
    )
    make_figure(
        he3,
        rinfo['label'] + "\nHe-3 Ground Truth (GRF) | " + coord_str,
        "He-3 (ppb)",
        DIR_FIG / (rname + "_he3.png"),
        cmap_name='plasma',
        vmin=0, vmax=15,
    )
    # Same data as above, colorbar auto-scaled to this region's own min/max
    # instead of the fixed [0, 15] used for cross-region comparison. Useful
    # when a region's range is narrow (e.g. highlands) and gets compressed
    # into a sliver of the shared 0-15 scale.
    make_figure(
        he3,
        rinfo['label'] + "\nHe-3 Ground Truth (GRF) | " + coord_str + " [adaptive scale]",
        "He-3 (ppb)",
        DIR_FIG / (rname + "_he3_adaptive.png"),
        cmap_name='plasma',
    )
    sys.stdout.write("  " + rname + " complete!\n"); sys.stdout.flush()


# ── METHOD.md ──────────────────────────────────────────────────────
sys.stdout.write("\n" + "=" * 60 + "\n"); sys.stdout.flush()
sys.stdout.write("Generating METHOD.md...\n"); sys.stdout.flush()

method_md = """\
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
"""

doc_path = DIR_DOC / "METHOD.md"
doc_path.write_text(method_md, encoding="utf-8", newline="\n")
sys.stdout.write("Saved: docs/METHOD.md\n"); sys.stdout.flush()
sys.stdout.write("\nPipeline complete!\n"); sys.stdout.flush()
