import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import style as S  # auto-applies paper style
# data.py not needed: this figure reads the raw yearly climate workbook.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(REPO_ROOT, "Yearly_Results", "locations_result.xlsx")

# quantity key -> (workbook column, panel title, y-axis label)
QUANTS = [
    ("pv",      "Total pv (kWh/kW Capacity)",  "(a) Annual PV yield",
     "PV yield [kWh/kW$\\cdot$yr]"),
    ("heating", "Total Heating_Load (kWh T)",  "(b) Annual heating demand",
     "Heating demand [kWh$_\\mathrm{th}$/yr]"),
    ("cooling", "Total Cooling_Load (kWh T)",  "(c) Annual cooling demand",
     "Cooling demand [kWh$_\\mathrm{th}$/yr]"),
]

# ---- load: one sheet per climate, 25 years (1998-2022) each ----------------- #
raw = {loc: pd.read_excel(XLSX, sheet_name=loc) for loc in S.LOCATION_ORDER}

# per-climate value arrays for each quantity
values = {q: [raw[loc][col].to_numpy(dtype=float) for loc in S.LOCATION_ORDER]
          for q, col, _, _ in QUANTS}

# ---- printed stats: median + CV (std/mean) per climate & quantity ----------- #
rows = []
for q, col, title, _ in QUANTS:
    for loc in S.LOCATION_ORDER:
        v = raw[loc][col].to_numpy(dtype=float)
        mean = float(np.mean(v)); std = float(np.std(v, ddof=1))
        rows.append(dict(
            quantity=q, climate=S.CLIMATE_LABEL[loc], location=loc,
            n_years=int(v.size), median=float(np.median(v)),
            mean=mean, std=std, cv=std / mean if mean else np.nan,
            min=float(np.min(v)), max=float(np.max(v))))
stats = pd.DataFrame(rows)

BOX = "#BFBFBF"      # single neutral fill for all boxes
EDGE = "#333333"
MED = "#000000"

fig, axes = plt.subplots(1, 3, figsize=S.figsize_double(height=3.2))
xpos = np.arange(1, len(S.LOCATION_ORDER) + 1)
xlabels = [S.CLIMATE_LABEL[l] for l in S.LOCATION_ORDER]

for ax, (q, col, title, ylab) in zip(axes, QUANTS):
    bp = ax.boxplot(values[q], positions=xpos, widths=0.6,
                    patch_artist=True, showfliers=True,
                    medianprops=dict(color=MED, linewidth=1.3),
                    whiskerprops=dict(color=EDGE, linewidth=0.8),
                    capprops=dict(color=EDGE, linewidth=0.8),
                    boxprops=dict(edgecolor=EDGE, linewidth=0.8),
                    flierprops=dict(marker="o", markersize=2.5,
                                    markerfacecolor=EDGE, markeredgecolor=EDGE,
                                    alpha=0.7))
    for patch in bp["boxes"]:
        patch.set_facecolor(BOX)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.06)
    S.despine(ax)
    S.ygrid(ax)

fig.tight_layout(w_pad=1.4)

caption = (
    "Interannual variability of climate drivers by site, over 25 weather years "
    "(1998-2022; one boxplot per climate). Panels show the annual (a) PV yield "
    "[kWh per kW installed], (b) heating thermal demand and (c) cooling thermal "
    "demand [kWh_th/yr]; boxes span the interquartile range with the median line, "
    "whiskers to 1.5xIQR, and outlier years as points. Climates are ordered "
    "cold-to-hot. Note the large across-climate magnitude differences (e.g. "
    "polar heating vs near-zero tropical heating). Annual PEAK ELECTRICAL LOAD, a "
    "fourth requested panel, is omitted because current exports contain only "
    "annual energy totals, not peak power. Source: Yearly_Results/"
    "locations_result.xlsx.")

S.save_fig(fig, "si_fig_variability", section="si", data=stats, caption=caption)

# console: median + CV per climate/quantity
pd.set_option("display.width", 160)
for q, _, title, _ in QUANTS:
    sub = stats[stats.quantity == q].set_index("climate")
    print(f"\n{title}")
    print(sub[["median", "cv"]].to_string(
        float_format=lambda x: f"{x:,.3f}"))
