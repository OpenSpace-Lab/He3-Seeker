"""
Task 2 addendum: relief-based non-traversable obstacles.

Pure slope thresholding (15/20/25 deg) produced 0% non-traversable area for
both T1 and T2, because the 59 m/px source DEM only has 9x9 real samples
across the 500 m window and the bicubic-interpolated surface is too smooth
(max slope ~10.4-10.7 deg). This script adds a local-relief criterion on top
of the existing slope mask:

  - Depression (crater/pit): local relief below -K*sigma  -> non-traversable
    (no area constraint - any deep low spot is an obstacle)
  - Peak (small hill): local relief above +K*sigma AND connected-component
    area < AREA_FRAC_MAX of the domain -> non-traversable
    (broad, gentle highs are NOT obstacles - only sharp, small-footprint
    peaks are, since those indicate rugged/blocky terrain)

Final mask = slope_mask OR crater_mask OR peak_mask, then morphological
closing (3x3) as before.
"""

import sys
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter, binary_closing, label
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DIR_OUT = ROOT / "outputs"
DIR_FIG = ROOT / "figures"

REGIONS = ["T1", "T2"]
SLOPE_THRESHOLDS = [15, 20, 25]

RELIEF_WINDOW = 100      # px (~100 m) - local-neighborhood scale for relief
RELIEF_K = 1.5           # obstacle threshold = K * std(relief)
PEAK_AREA_FRAC_MAX = 0.05  # peaks must cover < 5% of domain area to count

struct3 = np.ones((3, 3), dtype=bool)


def print_stats(name, arr):
    sys.stdout.write("  " + name + ": min=" + str(round(float(np.nanmin(arr)), 4))
                     + ", max=" + str(round(float(np.nanmax(arr)), 4))
                     + ", mean=" + str(round(float(np.nanmean(arr)), 4)) + "\n")
    sys.stdout.flush()


def _get_fonts():
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_sm = ImageFont.truetype("arial.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        font_sm = font
    return font, font_sm


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
    vmin = float(np.nanmin(he3))
    vmax = float(np.nanmax(he3))
    cmap = plt.get_cmap('plasma')
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mapped = cmap(norm(he3))
    img_arr = (mapped[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(img_arr)
    if mask.any():
        draw = ImageDraw.Draw(img)
        binary = mask.astype(np.uint8)
        edges = np.zeros_like(binary)
        edges[:-1, :] |= np.abs(np.diff(binary, axis=0)).astype(np.uint8)
        edges[:, :-1] |= np.abs(np.diff(binary, axis=1)).astype(np.uint8)
        ys, xs = np.where(edges > 0)
        for y, x in zip(ys, xs):
            draw.rectangle([x - 1, y - 1, x + 1, y + 1], fill=(255, 0, 0))

    h, w = he3.shape
    font, font_sm = _get_fonts()
    cbar_w = 25
    cbar_data = np.linspace(1, 0, h).reshape(-1, 1).repeat(cbar_w, axis=1)
    cbar_mapped = (cmap(cbar_data)[:, :, :3] * 255).astype(np.uint8)
    top_margin = 55
    left_margin = 10
    gap = 15
    label_w = 80
    total_w = left_margin + w + gap + cbar_w + label_w
    total_h = top_margin + h + 30
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    canvas.paste(img, (left_margin, top_margin))
    canvas.paste(Image.fromarray(cbar_mapped), (left_margin + w + gap, top_margin))
    draw2 = ImageDraw.Draw(canvas)
    lines = title.split('\n')
    for i, line in enumerate(lines):
        draw2.text((left_margin, 5 + i * 18), line, fill=(0, 0, 0), font=font)
    cx = left_margin + w + gap + cbar_w + 5
    draw2.text((cx, top_margin - 2), str(round(vmax, 2)), fill=(0, 0, 0), font=font_sm)
    draw2.text((cx, top_margin + h - 14), str(round(vmin, 2)), fill=(0, 0, 0), font=font_sm)
    mid = (vmin + vmax) / 2
    draw2.text((cx, top_margin + h // 2 - 7), str(round(mid, 2)), fill=(0, 0, 0), font=font_sm)
    draw2.text((left_margin + w + gap, top_margin + h + 5), 'He-3 (ppb)',
               fill=(0, 0, 0), font=font_sm)
    pct = float(mask.sum()) / mask.size * 100
    draw2.text((left_margin, top_margin + h + 5),
               "Red = non-traversable (" + str(round(pct, 1)) + "%)",
               fill=(200, 0, 0), font=font_sm)
    canvas.save(str(out_path), dpi=(150, 150))
    sys.stdout.write("  Saved: " + out_path.name + "\n"); sys.stdout.flush()


sys.stdout.write("=" * 60 + "\n")
sys.stdout.write("Adding relief-based obstacles (craters + small peaks)\n")
sys.stdout.write("Window=" + str(RELIEF_WINDOW) + "m, K=" + str(RELIEF_K)
                 + " sigma, peak_area_max=" + str(PEAK_AREA_FRAC_MAX * 100) + "%\n")
sys.stdout.write("=" * 60 + "\n"); sys.stdout.flush()

for rname in REGIONS:
    sys.stdout.write("\n--- " + rname + " ---\n"); sys.stdout.flush()

    dem = np.load(str(DIR_OUT / (rname + "_dem.npy")))
    slope = np.load(str(DIR_OUT / (rname + "_slope.npy")))
    he3 = np.load(str(DIR_OUT / (rname + "_he3.npy")))

    local_mean = uniform_filter(dem, size=RELIEF_WINDOW)
    relief = dem - local_mean
    relief_std = float(relief.std())
    depth_thresh = RELIEF_K * relief_std
    height_thresh = RELIEF_K * relief_std

    print_stats("Local relief (m)", relief)
    sys.stdout.write("  relief_std=" + str(round(relief_std, 4))
                     + " m -> depth_thresh=" + str(round(depth_thresh, 4))
                     + " m, height_thresh=" + str(round(height_thresh, 4)) + " m\n")
    sys.stdout.flush()

    crater_mask = relief < -depth_thresh
    crater_pct = float(crater_mask.sum()) / crater_mask.size * 100
    sys.stdout.write("  Crater/pit mask (no area limit): " + str(int(crater_mask.sum()))
                     + " px (" + str(round(crater_pct, 2)) + "%)\n"); sys.stdout.flush()

    peak_raw = relief > height_thresh
    labeled, num = label(peak_raw)
    peak_mask = np.zeros_like(peak_raw)
    area_max_px = PEAK_AREA_FRAC_MAX * dem.size
    kept, rejected = 0, 0
    for i in range(1, num + 1):
        comp = (labeled == i)
        comp_area = comp.sum()
        if comp_area < area_max_px:
            peak_mask |= comp
            kept += 1
        else:
            rejected += 1
    peak_pct = float(peak_mask.sum()) / peak_mask.size * 100
    sys.stdout.write("  Peak mask: " + str(num) + " raw components, " + str(kept)
                     + " kept (small, <" + str(PEAK_AREA_FRAC_MAX * 100) + "% area), "
                     + str(rejected) + " rejected (too broad) -> "
                     + str(int(peak_mask.sum())) + " px (" + str(round(peak_pct, 2)) + "%)\n")
    sys.stdout.flush()

    relief_obstacle = crater_mask | peak_mask
    relief_pct = float(relief_obstacle.sum()) / relief_obstacle.size * 100
    sys.stdout.write("  Relief obstacle (crater OR peak): "
                     + str(round(relief_pct, 2)) + "%\n"); sys.stdout.flush()

    for thresh in SLOPE_THRESHOLDS:
        slope_mask = slope > thresh
        combined = slope_mask | relief_obstacle
        combined_closed = binary_closing(combined, structure=struct3).astype(np.uint8)
        pct = float(combined_closed.sum()) / combined_closed.size * 100

        if thresh == 20:
            out_name = rname + "_nontraversable.npy"
        else:
            out_name = rname + "_nontraversable_" + str(thresh) + "deg.npy"

        np.save(str(DIR_OUT / out_name), combined_closed)
        sys.stdout.write("  Combined (slope>" + str(thresh) + " OR relief), closed: "
                         + str(int(combined_closed.sum())) + " / " + str(combined_closed.size)
                         + " px (" + str(round(pct, 2)) + "%) -> " + out_name + "\n")
        sys.stdout.flush()

        if thresh == 20:
            mask_20 = combined_closed

    # Figures (primary 20-deg variant)
    region_label = "Region " + rname + (
        ": Taurus-Littrow (Apollo 17)" if rname == "T1" else ": Hadley Rille (Apollo 15)")
    coord_str = "(T1: 20.19N,30.772E)" if rname == "T1" else "(T2: 26.132N,3.633E)"

    make_nontraversable_figure(
        mask_20,
        region_label + "\nNon-traversable (slope>20 OR relief-based)",
        DIR_FIG / (rname + "_nontraversable.png"),
    )
    make_overlay_figure(
        he3, mask_20,
        region_label + "\nHe-3 + Obstacles (slope + relief)",
        DIR_FIG / (rname + "_overlay.png"),
    )

sys.stdout.write("\nDone.\n"); sys.stdout.flush()
