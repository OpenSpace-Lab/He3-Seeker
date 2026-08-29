"""
Task 2: He-3 ground truth + DEM terrain analysis
Regions: T1 (Taurus-Littrow / Apollo 17), T2 (Hadley Rille / Apollo 15)
Reuses Task 1 He-3 pipeline; adds DEM, slope, and non-traversable masks.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom as scipy_zoom
from scipy.ndimage import binary_closing

# If GDAL can't locate its data dir automatically, set GDAL_DATA yourself before
# running this script (conda: <env>/Library/share/gdal on Windows, <env>/share/gdal
# on Linux/Mac).
os.environ['GDAL_HTTP_TIMEOUT'] = '120'
os.environ['GDAL_HTTP_MAX_RETRY'] = '5'
os.environ['GDAL_HTTP_RETRY_DELAY'] = '5'
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tif,.LBL,.IMG'
os.environ['VSI_CACHE'] = 'TRUE'
os.environ['VSI_CACHE_SIZE'] = '67108864'
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent.parent / '.mpl_cache'))

from osgeo import gdal, osr
gdal.UseExceptions()

import rasterio
from rasterio.transform import from_bounds as transform_from_bounds
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DIR_RAW = ROOT / "data_raw"
DIR_OUT = ROOT / "outputs"
DIR_FIG = ROOT / "figures"
DIR_DOC = ROOT / "docs"
for d in [DIR_RAW, DIR_OUT, DIR_FIG, DIR_DOC]:
    d.mkdir(parents=True, exist_ok=True)

# ── Remote URLs ────────────────────────────────────────────────────────
URL_FEO  = "/vsicurl/https://planetarymaps.usgs.gov/mosaic/Lunar_MI_mineral_maps/Lunar_Kaguya_MIMap_MineralDeconv_FeOWeightPercent_50N50S.tif"
URL_OMAT = "/vsicurl/https://planetarymaps.usgs.gov/mosaic/Lunar_MI_mineral_maps/Lunar_Kaguya_MIMap_MineralDeconv_OpticalMaturityIndex_50N50S.tif"
URL_DEM  = "/vsicurl/https://imbrium.mit.edu/DATA/SLDEM2015/TILES/FLOAT_IMG/SLDEM2015_512_00N_30N_000_045_FLOAT.LBL"

# ── Region definitions ─────────────────────────────────────────────────
HALF_SIZE_DEG = 0.0165 / 2  # same as Task 1

REGIONS = {
    "T1": {
        "label": "Region T1: Taurus-Littrow (Apollo 17)",
        "center_lat": 20.190,
        "center_lon": 30.772,
        "solar_flux_F": 0.50,
    },
    "T2": {
        "label": "Region T2: Hadley Rille (Apollo 15)",
        "center_lat": 26.132,
        "center_lon": 3.633,
        "solar_flux_F": 0.51,
    },
}

TARGET_GRID = 500
MOON_RADIUS_M = 1737400.0
MOON_CRS_WKT = None
SLOPE_THRESHOLDS = [15, 20, 25]


# ══════════════════════════════════════════════════════════════════════
# Helper functions — He-3 pipeline (unchanged from Task 1)
# ══════════════════════════════════════════════════════════════════════

def print_stats(name, arr):
    if np.issubdtype(arr.dtype, np.floating):
        valid = arr[~np.isnan(arr)]
    else:
        valid = arr.ravel()
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
    """Crop from geographic CRS raster (FeO / OMAT). Identical to Task 1."""
    global MOON_CRS_WKT
    t0 = time.time()
    fname = url.split('/')[-1]
    sys.stdout.write("    Opening: ..." + fname + "\n"); sys.stdout.flush()
    ds = gdal.Open(url)
    gt = ds.GetGeoTransform()
    if MOON_CRS_WKT is None:
        MOON_CRS_WKT = ds.GetProjection()
        px_deg = gt[1]
        px_m = px_deg * (np.pi / 180) * MOON_RADIUS_M
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


def crop_dem_remote(url, lon_min, lat_min, lon_max, lat_max, out_path):
    """Crop from projected CRS raster (SLDEM2015 Equirectangular, km)."""
    t0 = time.time()
    fname = url.split('/')[-1]
    sys.stdout.write("    Opening: ..." + fname + "\n"); sys.stdout.flush()
    ds = gdal.Open(url)
    gt = ds.GetGeoTransform()

    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    R = srs.GetSemiMajor()
    center_lon = srs.GetProjParm("central_meridian", 0.0)
    deg2m = np.pi / 180.0 * R
    sys.stdout.write("    Projected CRS: R=" + str(R) + " m, center_lon="
                     + str(center_lon) + " deg\n")
    sys.stdout.write("    Pixel size: " + str(round(gt[1], 2)) + " m\n")
    sys.stdout.flush()

    x_min = (lon_min - center_lon) * deg2m
    x_max = (lon_max - center_lon) * deg2m
    y_min = lat_min * deg2m
    y_max = lat_max * deg2m

    col_off = int((x_min - gt[0]) / gt[1])
    row_off = int((y_max - gt[3]) / gt[5])
    ncols = int(np.ceil((x_max - x_min) / gt[1]))
    nrows = int(np.ceil((y_max - y_min) / abs(gt[5])))
    sys.stdout.write("    Window: col=" + str(col_off) + ", row=" + str(row_off)
                     + ", " + str(ncols) + "x" + str(nrows) + " px\n")
    sys.stdout.flush()

    band = ds.GetRasterBand(1)
    sys.stdout.write("    Reading...\n"); sys.stdout.flush()
    data = band.ReadAsArray(col_off, row_off, ncols, nrows).astype(np.float64)

    nodata = band.GetNoDataValue()
    if nodata is not None:
        data[data < -1e30] = np.nan

    data = data * 1000.0  # km -> m

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


def generate_grf_fft(shape, corr_length, rng):
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
    a_c = a - a.mean()
    b_c = b - b.mean()
    return float((a_c * b_c).sum() / (np.sqrt((a_c**2).sum() * (b_c**2).sum()) + 1e-30))


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


# ══════════════════════════════════════════════════════════════════════
# Visualization functions (PIL-based, avoiding Agg savefig crash)
# ══════════════════════════════════════════════════════════════════════

def _get_fonts():
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font
    return font, font_sm


def _make_colorbar(cmap, h, w=25):
    cbar_data = np.linspace(1, 0, h).reshape(-1, 1).repeat(w, axis=1)
    cbar_mapped = cmap(cbar_data)[:, :, :3]
    return (cbar_mapped * 255).astype(np.uint8)


def _arr_to_rgb(arr, cmap_name, vmin=None, vmax=None):
    if vmin is None:
        vmin = float(np.nanmin(arr))
    if vmax is None:
        vmax = float(np.nanmax(arr))
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mapped = cmap(norm(arr))
    nan_mask = np.isnan(arr)
    mapped[nan_mask] = [0.2, 0.2, 0.2, 1.0]
    return (mapped[:, :, :3] * 255).astype(np.uint8), cmap, vmin, vmax


def _build_canvas(img_arr, title, cbar_label, cmap, vmin, vmax):
    h, w = img_arr.shape[:2]
    font, font_sm = _get_fonts()
    cbar_w = 25
    cbar_arr = _make_colorbar(cmap, h, cbar_w)
    top_margin = 55
    left_margin = 10
    gap = 15
    label_w = 80
    total_w = left_margin + w + gap + cbar_w + label_w
    total_h = top_margin + h + 30
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    canvas.paste(Image.fromarray(img_arr), (left_margin, top_margin))
    canvas.paste(Image.fromarray(cbar_arr), (left_margin + w + gap, top_margin))
    draw = ImageDraw.Draw(canvas)
    lines = title.split('\n')
    for i, line in enumerate(lines):
        draw.text((left_margin, 5 + i * 18), line, fill=(0, 0, 0), font=font)
    cx = left_margin + w + gap + cbar_w + 5
    draw.text((cx, top_margin - 2), str(round(vmax, 2)), fill=(0, 0, 0), font=font_sm)
    draw.text((cx, top_margin + h - 14), str(round(vmin, 2)), fill=(0, 0, 0), font=font_sm)
    mid = (vmin + vmax) / 2
    draw.text((cx, top_margin + h // 2 - 7), str(round(mid, 2)), fill=(0, 0, 0), font=font_sm)
    draw.text((left_margin + w + gap, top_margin + h + 5), cbar_label,
              fill=(0, 0, 0), font=font_sm)
    draw.text((left_margin, top_margin + h + 5), "500x500 m, ~1 m/px",
              fill=(100, 100, 100), font=font_sm)
    return canvas


def make_he3_figure(arr, title, out_path):
    img_arr, cmap, vmin, vmax = _arr_to_rgb(arr, 'plasma')
    canvas = _build_canvas(img_arr, title, 'He-3 (ppb)', cmap, vmin, vmax)
    canvas.save(str(out_path), dpi=(150, 150))
    sys.stdout.write("  Saved: " + out_path.name + "\n"); sys.stdout.flush()


def make_dem_figure(arr, title, out_path):
    img_arr, cmap, vmin, vmax = _arr_to_rgb(arr, 'terrain')
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    h, w = arr.shape
    elev_range = vmax - vmin
    if elev_range > 200:
        interval = 20
    elif elev_range > 50:
        interval = 10
    elif elev_range > 10:
        interval = 5
    elif elev_range > 2:
        interval = 1
    else:
        interval = 0.5
    levels = np.arange(np.floor(vmin / interval) * interval,
                       np.ceil(vmax / interval) * interval + interval, interval)
    for lev in levels:
        binary = (arr >= lev).astype(np.uint8)
        edges_y = np.abs(np.diff(binary, axis=0))
        edges_x = np.abs(np.diff(binary, axis=1))
        for r in range(edges_y.shape[0]):
            for c in range(edges_y.shape[1]):
                if edges_y[r, c]:
                    draw.point((c, r), fill=(0, 0, 0))
        for r in range(edges_x.shape[0]):
            for c in range(edges_x.shape[1]):
                if edges_x[r, c]:
                    draw.point((c, r), fill=(0, 0, 0))
    canvas = _build_canvas(np.array(img), title, 'Elevation (m)', cmap, vmin, vmax)
    font_sm = _get_fonts()[1]
    draw2 = ImageDraw.Draw(canvas)
    draw2.text((10, 55 + h + 15), "Contour interval: " + str(interval) + " m",
               fill=(80, 80, 80), font=font_sm)
    canvas.save(str(out_path), dpi=(150, 150))
    sys.stdout.write("  Saved: " + out_path.name + "\n"); sys.stdout.flush()


def make_slope_figure(arr, title, out_path):
    vmax_val = max(float(np.nanmax(arr)), 1.0)
    img_arr, cmap, vmin, vmax = _arr_to_rgb(arr, 'YlOrRd', vmin=0, vmax=vmax_val)
    canvas = _build_canvas(img_arr, title, 'Slope (deg)', cmap, 0, vmax_val)
    canvas.save(str(out_path), dpi=(150, 150))
    sys.stdout.write("  Saved: " + out_path.name + "\n"); sys.stdout.flush()


def make_nontraversable_figure(mask, title, out_path):
    h, w = mask.shape
    display = np.where(mask, 0, 255).astype(np.uint8)
    img = Image.fromarray(display, mode='L').convert('RGB')
    font, font_sm = _get_fonts()
    pct = float(mask.sum()) / mask.size * 100
    top_margin = 55
    left_margin = 10
    total_w = left_margin + w + 10
    total_h = top_margin + h + 30
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    canvas.paste(img, (left_margin, top_margin))
    draw = ImageDraw.Draw(canvas)
    lines = title.split('\n')
    for i, line in enumerate(lines):
        draw.text((left_margin, 5 + i * 18), line, fill=(0, 0, 0), font=font)
    draw.text((left_margin, top_margin + h + 5),
              "Black=non-traversable (" + str(round(pct, 1)) + "%), White=traversable",
              fill=(100, 100, 100), font=font_sm)
    canvas.save(str(out_path), dpi=(150, 150))
    sys.stdout.write("  Saved: " + out_path.name + "\n"); sys.stdout.flush()


def make_overlay_figure(he3, mask, title, out_path):
    img_arr, cmap, vmin, vmax = _arr_to_rgb(he3, 'plasma')
    img = Image.fromarray(img_arr)
    if mask.any():
        draw = ImageDraw.Draw(img)
        binary = mask.astype(np.uint8)
        edges = np.zeros_like(binary)
        edges[:-1, :] |= np.abs(np.diff(binary, axis=0)).astype(np.uint8)
        edges[:, :-1] |= np.abs(np.diff(binary, axis=1)).astype(np.uint8)
        ys, xs = np.where(edges > 0)
        for y, x in zip(ys, xs):
            draw.rectangle([x-1, y-1, x+1, y+1], fill=(255, 0, 0))
    canvas = _build_canvas(np.array(img), title, 'He-3 (ppb)', cmap, vmin, vmax)
    pct = float(mask.sum()) / mask.size * 100
    font_sm = _get_fonts()[1]
    draw2 = ImageDraw.Draw(canvas)
    draw2.text((10, 55 + he3.shape[0] + 15),
               "Red boundary = non-traversable (" + str(round(pct, 1)) + "%)",
               fill=(200, 0, 0), font=font_sm)
    canvas.save(str(out_path), dpi=(150, 150))
    sys.stdout.write("  Saved: " + out_path.name + "\n"); sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════
# Pre-check: verify DEM URL
# ══════════════════════════════════════════════════════════════════════

sys.stdout.write("=" * 60 + "\n")
sys.stdout.write("Task 2: He-3 + DEM Ground Truth Generation\n")
sys.stdout.write("=" * 60 + "\n\n")

sys.stdout.write("[Pre-check] Verifying DEM URL...\n"); sys.stdout.flush()
try:
    dem_test = gdal.Open(URL_DEM)
    dem_gt = dem_test.GetGeoTransform()
    sys.stdout.write("  DEM OK: " + str(dem_test.RasterXSize) + "x"
                     + str(dem_test.RasterYSize) + " px, pixel="
                     + str(round(dem_gt[1], 2)) + " m\n")
    dem_test = None
except Exception as e:
    sys.stdout.write("  DEM URL FAILED: " + str(e) + "\n")
    sys.stdout.write("  Cannot proceed without DEM.\n")
    sys.stdout.flush()
    sys.exit(1)
sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════
# Main processing loop
# ══════════════════════════════════════════════════════════════════════

grf_rng = np.random.default_rng(42)

for rname, rinfo in REGIONS.items():
    sys.stdout.write("\n" + "=" * 60 + "\n")
    sys.stdout.write("Processing " + rinfo['label'] + "\n")
    sys.stdout.write("=" * 60 + "\n"); sys.stdout.flush()

    clat = rinfo["center_lat"]
    clon = rinfo["center_lon"]
    lon_min = clon - HALF_SIZE_DEG
    lon_max = clon + HALF_SIZE_DEG
    lat_min = clat - HALF_SIZE_DEG
    lat_max = clat + HALF_SIZE_DEG
    bounds = (lon_min, lat_min, lon_max, lat_max)

    sys.stdout.write("  Center: (" + str(clat) + "N, " + str(clon) + "E)\n")
    sys.stdout.write("  Bounds: lon [" + str(round(lon_min, 5)) + ", "
                     + str(round(lon_max, 5)) + "], lat ["
                     + str(round(lat_min, 5)) + ", " + str(round(lat_max, 5)) + "]\n")
    sys.stdout.flush()

    # ════════════════════════════════════════════════════════════
    # Part A: He-3 pipeline (identical to Task 1)
    # ════════════════════════════════════════════════════════════

    # Step 1: Remote crop FeO + OMAT
    sys.stdout.write("\n[Part A - Step 1] Remote crop FeO and OMAT...\n"); sys.stdout.flush()

    feo_path = DIR_RAW / (rname + "_feo.tif")
    omat_path = DIR_RAW / (rname + "_omat.tif")

    sys.stdout.write("  Cropping FeO...\n"); sys.stdout.flush()
    feo_raw = crop_remote_gdal(URL_FEO, lon_min, lat_min, lon_max, lat_max, feo_path)
    print_stats("FeO raw (wt%)", feo_raw)

    sys.stdout.write("  Cropping OMAT...\n"); sys.stdout.flush()
    omat_raw = crop_remote_gdal(URL_OMAT, lon_min, lat_min, lon_max, lat_max, omat_path)
    print_stats("OMAT raw", omat_raw)

    # Step 2: FeO -> TiO2
    sys.stdout.write("\n[Part A - Step 2] FeO -> TiO2 (Lucey 1998)...\n"); sys.stdout.flush()
    feo = feo_raw.copy()
    feo[feo <= 0] = np.nan
    tio2 = np.power(10, 0.06 * feo - 0.54)
    print_stats("TiO2 (wt%)", tio2)

    # Step 3: Bicubic interpolation
    sys.stdout.write("\n[Part A - Step 3] Bicubic interpolation " + str(feo_raw.shape)
                     + " -> " + str(TARGET_GRID) + "x" + str(TARGET_GRID) + "...\n")
    sys.stdout.flush()

    feo_interp = interp_layer(feo)
    omat_interp = interp_layer(omat_raw)
    tio2_interp = interp_layer(tio2)

    print_stats("FeO interp", feo_interp)
    print_stats("OMAT interp", omat_interp)
    print_stats("TiO2 interp", tio2_interp)

    # Step 3b: Joint GRF perturbation
    sys.stdout.write("\n[Part A - Step 3b] Correlated GRF on FeO & OMAT (L=25m, sigma=5%)...\n")
    sys.stdout.flush()

    CORR_LENGTH = 25.0
    SIGMA_FRAC = 0.05
    grid_shape = (TARGET_GRID, TARGET_GRID)

    flat_f = feo_interp.ravel()
    flat_o = omat_interp.ravel()
    valid_mask = ~(np.isnan(flat_f) | np.isnan(flat_o))
    r_feo_omat = manual_pearson(flat_f[valid_mask], flat_o[valid_mask])
    sys.stdout.write("  corr(FeO, OMAT) from interp = " + str(round(r_feo_omat, 4)) + "\n")
    sys.stdout.flush()

    grf1 = generate_grf_fft(grid_shape, CORR_LENGTH, grf_rng)
    grf2 = generate_grf_fft(grid_shape, CORR_LENGTH, grf_rng)

    r = r_feo_omat
    grf_feo = grf1
    grf_omat = r * grf1 + np.sqrt(max(1.0 - r * r, 0.0)) * grf2

    sys.stdout.write("  Correlated GRFs: corr(grf_feo, grf_omat) = "
                     + str(round(manual_pearson(grf_feo.ravel(), grf_omat.ravel()), 4))
                     + " (target " + str(round(r, 4)) + ")\n")
    sys.stdout.flush()

    feo_mean = float(np.nanmean(feo_interp))
    omat_mean = float(np.nanmean(omat_interp))

    feo_perturbed = feo_interp + SIGMA_FRAC * feo_mean * grf_feo
    omat_perturbed = omat_interp + SIGMA_FRAC * omat_mean * grf_omat

    feo_perturbed = np.clip(feo_perturbed, 0, None)
    omat_perturbed = np.clip(omat_perturbed, 0.05, None)

    feo_for_tio2 = feo_perturbed.copy()
    feo_for_tio2[feo_for_tio2 <= 0] = np.nan
    tio2_perturbed = np.power(10, 0.06 * feo_for_tio2 - 0.54)
    tio2_perturbed = np.clip(tio2_perturbed, 0, None)

    nan_mask = np.isnan(feo_interp) | np.isnan(omat_interp)
    feo_perturbed[nan_mask] = np.nan
    tio2_perturbed[nan_mask] = np.nan
    omat_perturbed[nan_mask] = np.nan

    sys.stdout.write("  After GRF perturbation:\n"); sys.stdout.flush()
    print_stats("FeO perturbed", feo_perturbed)
    print_stats("TiO2 re-derived", tio2_perturbed)
    print_stats("OMAT perturbed", omat_perturbed)

    # Step 4: He-3 calculation
    sys.stdout.write("\n[Part A - Step 4] He-3 (Fa & Jin 2007, F="
                     + str(rinfo['solar_flux_F']) + ")...\n")
    sys.stdout.flush()
    F = rinfo["solar_flux_F"]

    he3 = 0.56 * (tio2_perturbed * F / omat_perturbed) + 1.62
    he3 = np.clip(he3, 0, 15)
    print_stats("He-3 (ppb)", he3)

    valid_h = ~(np.isnan(tio2_perturbed.ravel()) | np.isnan(he3.ravel()))
    r_ti_he3 = manual_pearson(tio2_perturbed.ravel()[valid_h], he3.ravel()[valid_h])
    r_omat_he3 = manual_pearson(omat_perturbed.ravel()[valid_h], he3.ravel()[valid_h])
    sys.stdout.write("  Physical consistency:\n")
    sys.stdout.write("    corr(TiO2, He-3) = " + str(round(r_ti_he3, 4)) + " (expect > 0)\n")
    sys.stdout.write("    corr(OMAT, He-3) = " + str(round(r_omat_he3, 4)) + " (expect < 0)\n")
    sys.stdout.flush()

    np.save(str(DIR_OUT / (rname + "_he3.npy")), he3)
    sys.stdout.write("  Saved: " + rname + "_he3.npy\n"); sys.stdout.flush()

    # ════════════════════════════════════════════════════════════
    # Part B: DEM processing
    # ════════════════════════════════════════════════════════════

    sys.stdout.write("\n[Part B - Step 1] Remote crop DEM...\n"); sys.stdout.flush()
    dem_path = DIR_RAW / (rname + "_dem.tif")
    dem_raw = crop_dem_remote(URL_DEM, lon_min, lat_min, lon_max, lat_max, dem_path)
    print_stats("DEM raw (m)", dem_raw)

    sys.stdout.write("\n[Part B - Step 2] Bicubic interpolation DEM -> "
                     + str(TARGET_GRID) + "x" + str(TARGET_GRID) + "...\n")
    sys.stdout.flush()
    dem_interp = interp_layer(dem_raw)
    print_stats("DEM interp (m)", dem_interp)

    np.save(str(DIR_OUT / (rname + "_dem.npy")), dem_interp)
    sys.stdout.write("  Saved: " + rname + "_dem.npy\n"); sys.stdout.flush()

    sys.stdout.write("\n[Part B - Step 3] Computing slope...\n"); sys.stdout.flush()
    dx = 500.0 / TARGET_GRID  # 1.0 m
    grad_y, grad_x = np.gradient(dem_interp, dx)
    slope_deg = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
    print_stats("Slope (degrees)", slope_deg)

    np.save(str(DIR_OUT / (rname + "_slope.npy")), slope_deg)
    sys.stdout.write("  Saved: " + rname + "_slope.npy\n"); sys.stdout.flush()

    # ════════════════════════════════════════════════════════════
    # Part C: Non-traversable masks
    # ════════════════════════════════════════════════════════════

    sys.stdout.write("\n[Part C] Non-traversable masks...\n"); sys.stdout.flush()
    struct = np.ones((3, 3), dtype=bool)

    nontraversable_20 = None
    for thresh in SLOPE_THRESHOLDS:
        raw_mask = (slope_deg > thresh).astype(np.uint8)
        closed_mask = binary_closing(raw_mask, structure=struct).astype(np.uint8)
        pct = float(closed_mask.sum()) / closed_mask.size * 100

        if thresh == 20:
            out_name = rname + "_nontraversable.npy"
            nontraversable_20 = closed_mask
        else:
            out_name = rname + "_nontraversable_" + str(thresh) + "deg.npy"

        np.save(str(DIR_OUT / out_name), closed_mask)
        sys.stdout.write("  Slope > " + str(thresh) + " deg: "
                         + str(int(closed_mask.sum())) + " / " + str(closed_mask.size)
                         + " px (" + str(round(pct, 1)) + "%) -> " + out_name + "\n")
        sys.stdout.flush()

    # ════════════════════════════════════════════════════════════
    # Part D: Figures
    # ════════════════════════════════════════════════════════════

    sys.stdout.write("\n[Part D] Generating figures...\n"); sys.stdout.flush()
    coord_str = "(" + str(clat) + "N, " + str(clon) + "E)"

    make_he3_figure(
        he3,
        rinfo['label'] + "\nHe-3 Ground Truth | " + coord_str,
        DIR_FIG / (rname + "_he3.png"),
    )
    make_dem_figure(
        dem_interp,
        rinfo['label'] + "\nDEM Elevation | " + coord_str,
        DIR_FIG / (rname + "_dem.png"),
    )
    make_slope_figure(
        slope_deg,
        rinfo['label'] + "\nSlope | " + coord_str,
        DIR_FIG / (rname + "_slope.png"),
    )
    make_nontraversable_figure(
        nontraversable_20,
        rinfo['label'] + "\nNon-traversable (slope > 20 deg) | " + coord_str,
        DIR_FIG / (rname + "_nontraversable.png"),
    )
    make_overlay_figure(
        he3, nontraversable_20,
        rinfo['label'] + "\nHe-3 + Obstacles | " + coord_str,
        DIR_FIG / (rname + "_overlay.png"),
    )

    sys.stdout.write("  " + rname + " complete!\n"); sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════
# Part E: Documentation
# ══════════════════════════════════════════════════════════════════════

sys.stdout.write("\n" + "=" * 60 + "\n")
sys.stdout.write("Generating task2_METHOD.md...\n"); sys.stdout.flush()

method_md = """\
# Task 2 方法文档：He-3 真值场 + 地形数据

## 1. 区域选择理由

### Region T1：Taurus-Littrow (Apollo 17 着陆区)
- 中心坐标：(20.190°N, 30.772°E)，范围 500×500 m
- 位于月海玄武岩谷地，两侧为 North/South Massif 高地山体。
  500 m 范围内 TiO₂ 跨越 <1% (高地) 到 >6% (谷地玄武岩)，形成强烈的
  成分梯度。地形有显著陡坡（山体斜面可达 25–35°），构成天然不可通行区。
- 文献：Schmitt et al. (2017); Robinson & Jolliff (2002); Sun et al. (2021)

### Region T2：Hadley Rille (Apollo 15 着陆区)
- 中心坐标：(26.132°N, 3.633°E)，范围 500×500 m
- 月溪（蜿蜒沟谷，深约 300 m）与亚平宁山前月海交汇处。
  溪谷壁面陡峭，是天然地形障碍；同时月海玄武岩与溪谷暴露的分层
  提供了有趣的 He-3 成分对比。
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
> Barker, M.K., et al. (2016). A new lunar digital elevation model from the
> Lunar Orbiter Laser Altimeter and SELENE Terrain Camera.
> *Icarus*, 273, 346-355.

### 处理流程
1. 通过 GDAL `/vsicurl/` 远程读取 PDS 格式 DEM（HTTP range request）
2. 坐标转换：经纬度(°) → Equirectangular 投影坐标(m)
3. 按像素窗口裁切目标区域
4. km → m 单位转换
5. 双三次插值到 500×500 网格（与 He-3 场对齐）

## 4. 坡度计算方法

使用 `numpy.gradient` 计算 DEM 的空间梯度：

$$\\text{slope} = \\arctan\\sqrt{\\left(\\frac{\\partial z}{\\partial x}\\right)^2 + \\left(\\frac{\\partial z}{\\partial y}\\right)^2}$$

- 网格间距 Δx = Δy = 1 m（500 m / 500 pixels）
- 输出单位：度

## 5. 不可通行区阈值选择依据

- **主用阈值 20°**：月球巡视器（如玉兔号）的典型最大爬坡能力上限。
  NASA 和 ESA 的巡视器设计指标通常在 15°–25° 之间。
- **备用阈值 15°**：保守设定，适用于负载较重或地面松软的场景
- **备用阈值 25°**：激进设定，适用于高性能巡视器
- 形态学闭操作（`scipy.ndimage.binary_closing`，3×3 结构元素）用于
  填充小洞，使不可通行区域边界更连通、更符合实际通行约束。

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
"""

doc_path = DIR_DOC / "task2_METHOD.md"
doc_path.write_text(method_md, encoding="utf-8")
sys.stdout.write("Saved: docs/task2_METHOD.md\n")
sys.stdout.write("\nTask 2 pipeline complete!\n")
sys.stdout.flush()
