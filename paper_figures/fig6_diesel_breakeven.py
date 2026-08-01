import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np, pandas as pd
import style as S; import data as D

# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
VOLL = "Med"                 # representative operating point (HVAC $3, Critical $100 /kWh)
BAND = (100.0, 600.0)        # contested/hostile delivered fully-burdened fuel cost ($/gal)
LOWBAND = (13.0, 45.0)       # routine forward delivery -> protected convoy ($/gal)
REP  = "California"          # representative climate for panel (a)
PMAX = 800.0                 # max delivered price shown in panel (a) ($/gal)

# --------------------------------------------------------------------------- #
#  Diesel (VoLL-independent). fixed_cost = total - fuel; both fixed_cost and
#  gallons are delivered-price INDEPENDENT, so total(p) = fixed_cost + gallons*p.
# --------------------------------------------------------------------------- #
dfold = D.load_folds(architectures=("Diesel",))
dfold = dfold[(dfold.split == "test") & (dfold.voll == VOLL)].copy()
# The swept diesel workbook has a fuel-price dimension (Low/Med/High); fixed_cost and
# gallons are price-INDEPENDENT, so collapse to one level per (location, fold).
if "fuel_price_level" in dfold.columns and dfold["fuel_price_level"].notna().any():
    lvl = "Med" if (dfold["fuel_price_level"] == "Med").any() else dfold["fuel_price_level"].dropna().iloc[0]
    dfold = dfold[dfold["fuel_price_level"] == lvl].copy()
dfold["fixed_cost"] = dfold["total_cost"] - dfold["fuel_cost"]

dies = (dfold.groupby("location", observed=True)
        .agg(fixed_cost=("fixed_cost", "mean"),
             gallons=("fuel_gallons", "mean")).reset_index())
dies_by_loc = dies.set_index("location")
dmap = dfold.set_index(["location", "fold"])[["fixed_cost", "fuel_gallons"]].sort_index()

# --------------------------------------------------------------------------- #
#  Renewables: best design per (architecture, location) = min mean test total
#  cost across methods at Med VoLL.
# --------------------------------------------------------------------------- #
rsum = D.load_summary(architectures=("PCM", "PVB"))
rsum = rsum[(rsum.split == "test") & (rsum.voll == VOLL)].copy()
rfold = D.load_folds(architectures=("PCM", "PVB"))
rfold = rfold[(rfold.split == "test") & (rfold.voll == VOLL)].copy()

DESIGN_METHOD = "SO_CVaR"   # paper's recommended design (not the cheapest) — breakeven is method-insensitive
rows = []
for (arch, loc), g in rsum.groupby(["architecture", "location"], observed=True):
    gm = g[g["method"] == DESIGN_METHOD]
    gm = gm if not gm.empty else g
    i = gm["total_cost"].idxmin()   # single SO_CVaR row; idxmin just selects it robustly
    best_method = str(gm.loc[i, "method"])
    best_total = float(gm.loc[i, "total_cost"])
    fc = float(dies_by_loc.loc[loc, "fixed_cost"])
    gal = float(dies_by_loc.loc[loc, "gallons"])
    be_mean = (best_total - fc) / gal
    # across-fold dispersion of the breakeven price
    sub = rfold[(rfold.architecture == arch) & (rfold.location == loc)
                & (rfold.method == best_method)]
    be_folds = []
    for _, rr in sub.iterrows():
        key = (loc, rr["fold"])
        if key in dmap.index:
            fcf, galf = dmap.loc[key, ["fixed_cost", "fuel_gallons"]]
            be_folds.append((rr["total_cost"] - fcf) / galf)
    be_folds = np.asarray(be_folds, dtype=float)
    be_std = float(np.std(be_folds, ddof=1)) if be_folds.size > 1 else np.nan
    rows.append(dict(architecture=str(arch), location=str(loc),
                     climate=S.CLIMATE_LABEL[str(loc)], best_method=best_method,
                     best_total=best_total, diesel_fixed_cost=fc,
                     diesel_gallons=gal, breakeven_usd_per_gal=be_mean,
                     breakeven_std=be_std, n_folds=int(be_folds.size)))
be = pd.DataFrame(rows)
be["architecture"] = pd.Categorical(be["architecture"], S.ARCH_ORDER, ordered=True)
be["location"] = pd.Categorical(be["location"], S.LOCATION_ORDER, ordered=True)
be = be.sort_values(["location", "architecture"]).reset_index(drop=True)

# --------------------------------------------------------------------------- #
#  Figure
# --------------------------------------------------------------------------- #
fig, (axa, axb) = plt.subplots(1, 2, figsize=S.figsize_double(3.3))

# ============================ Panel (a) ==================================== #
p = np.logspace(np.log10(0.8), np.log10(PMAX), 400)

# thin diesel lines for the non-representative climates (context)
for loc in S.LOCATION_ORDER:
    if loc == REP:
        continue
    fc = dies_by_loc.loc[loc, "fixed_cost"]
    gal = dies_by_loc.loc[loc, "gallons"]
    axa.plot(p, fc + gal * p, color="0.7", lw=0.6, alpha=0.8, zorder=1)

# representative-climate diesel line (bold, grey dashed = Diesel style)
fc_r = float(dies_by_loc.loc[REP, "fixed_cost"])
gal_r = float(dies_by_loc.loc[REP, "gallons"])
axa.plot(p, fc_r + gal_r * p, zorder=4,
         label=f"Diesel ({S.CLIMATE_LABEL[REP]})", **S.DIESEL_LINE)

# fuel-cost bands: routine forward delivery (light) and contested/hostile (darker)
axa.axvspan(LOWBAND[0], LOWBAND[1], color="0.92", alpha=0.8, zorder=0)
axa.text(np.sqrt(LOWBAND[0] * LOWBAND[1]), 1.3e6,
         f"routine delivery\n${LOWBAND[0]:.0f}-{LOWBAND[1]:.0f}/gal",
         ha="center", va="top", fontsize=6.0, color="0.40")
axa.axvspan(BAND[0], BAND[1], color="0.82", alpha=0.6, zorder=0)
axa.text(np.sqrt(BAND[0] * BAND[1]), 1.3e6,
         f"contested delivery\n${BAND[0]:.0f}-{BAND[1]:.0f}/gal",
         ha="center", va="top", fontsize=6.0, color="0.30")

# best-renewable horizontal reference lines + breakeven markers (representative)
# price labels are offset per-architecture (up-right vs down-left) because the
# two CA reference levels almost coincide.
be_rep = be[be.location == REP]
price_off = {"PVB": (8, 9, "left", "bottom"),
             "PCM": (8, -11, "left", "top")}
ren_handles = []
for _, r in be_rep.iterrows():
    arch = r["architecture"]
    c = S.method_color(r["best_method"])
    axa.axhline(r["best_total"], color=c, lw=1.2, ls="-", alpha=0.9, zorder=3)
    ms = S.marker_style(r["best_method"], arch)
    axa.errorbar(r["breakeven_usd_per_gal"], r["best_total"],
                 xerr=r["breakeven_std"], zorder=6, markersize=7,
                 ecolor="black", elinewidth=0.8, capsize=2.5,
                 linestyle="none", **ms)
    dx, dy, ha, va = price_off[arch]
    axa.annotate(f"${r['breakeven_usd_per_gal']:.1f}/gal",
                 (r["breakeven_usd_per_gal"], r["best_total"]),
                 xytext=(dx, dy), textcoords="offset points",
                 ha=ha, va=va, fontsize=6.6, color=c, weight="bold")
    ren_handles.append(Line2D([0], [0], color=c, lw=1.2, linestyle="-",
                              label=f"{S.ARCH_LABEL[arch]}, {S.METHOD_LABEL[r['best_method']]}",
                              **{k: v for k, v in ms.items() if k != "color"}))

axa.set_xscale("log")
axa.set_yscale("log")
axa.set_xlim(0.8, PMAX)
axa.set_ylim(3e2, 2e6)
axa.set_xlabel("Delivered diesel price ($/gal, log)")
axa.set_ylabel("Total annualized cost ($/yr, log)")
axa.set_title(f"(a) Cost vs fuel price — {S.CLIMATE_LABEL[REP]}", loc="left")
S.despine(axa)
S.ygrid(axa)

# panel-(a) legend
handles_a = [
    Line2D([0], [0], **S.DIESEL_LINE, label=f"Diesel ({S.CLIMATE_LABEL[REP]})"),
    Line2D([0], [0], color="0.7", lw=0.8, label="Diesel (other climates)"),
] + ren_handles
axa.legend(handles=handles_a, loc="upper left", fontsize=6.0,
           handlelength=2.0, borderaxespad=0.5, labelspacing=0.35)

# ============================ Panel (b) ==================================== #
x = np.arange(len(S.LOCATION_ORDER), dtype=float)
w = 0.38
offsets = {"PVB": -w / 2, "PCM": +w / 2}

for arch in S.ARCH_ORDER:
    sub = be[be.architecture == arch].set_index("location")
    xs, hs, errs, cols, hatches = [], [], [], [], []
    for j, loc in enumerate(S.LOCATION_ORDER):
        if loc in sub.index:
            r = sub.loc[loc]
            xs.append(x[j] + offsets[arch])
            hs.append(r["breakeven_usd_per_gal"])
            errs.append(r["breakeven_std"])
            cols.append(S.method_color(r["best_method"]))
            hatches.append(S.ARCH_HATCH[arch])
    xs = np.array(xs); hs = np.array(hs); errs = np.array(errs)
    for xi, hi, ei, ci, ha in zip(xs, hs, errs, cols, hatches):
        axb.bar(xi, hi, width=w, color=ci, edgecolor="black",
                linewidth=0.6, hatch=ha, zorder=3)
        # error bar (across folds); clip lower whisker to stay on the log axis
        lo = min(ei, hi - 1.05) if np.isfinite(ei) else 0.0
        axb.errorbar(xi, hi, yerr=[[max(lo, 0.0)], [ei if np.isfinite(ei) else 0.0]],
                     ecolor="black", elinewidth=0.7, capsize=2, zorder=4,
                     linestyle="none")
        axb.annotate(f"{hi:.1f}", (xi, hi), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=6.2)

# fuel-cost bands (both far above every breakeven)
axb.axhspan(LOWBAND[0], LOWBAND[1], color="0.92", alpha=0.85, zorder=0)
axb.text(len(x) - 0.5, np.sqrt(LOWBAND[0] * LOWBAND[1]),
         f"routine delivery ${LOWBAND[0]:.0f}-{LOWBAND[1]:.0f}/gal",
         ha="right", va="center", fontsize=6.0, color="0.40")
axb.axhspan(BAND[0], BAND[1], color="0.82", alpha=0.6, zorder=0)
axb.text(len(x) - 0.5, np.sqrt(BAND[0] * BAND[1]),
         f"contested delivery\n${BAND[0]:.0f}-{BAND[1]:.0f}/gal",
         ha="right", va="center", fontsize=6.0, color="0.30")

axb.set_yscale("log")
axb.set_ylim(0.5, 1000.0)
axb.set_xticks(x)
axb.set_xticklabels([S.CLIMATE_LABEL[l] for l in S.LOCATION_ORDER],
                    rotation=28, ha="right")
axb.set_ylabel("Breakeven diesel price ($/gal, log)")
axb.set_title("(b) Breakeven price by climate", loc="left")
S.despine(axb)
S.ygrid(axb)

# panel-(b) legend: architecture (hatch) + best method (colour)
arch_h = [Patch(facecolor="white", edgecolor="black", hatch=S.ARCH_HATCH[a],
                label=S.ARCH_LABEL[a]) for a in S.ARCH_ORDER]
methods_used = [m for m in S.METHOD_ORDER if m in be["best_method"].unique()]
meth_h = [Patch(facecolor=S.method_color(m), edgecolor="black",
                label=f"{S.METHOD_LABEL[m]} design") for m in methods_used]
axb.legend(handles=arch_h + meth_h, loc="upper left", fontsize=6.0,
           handlelength=1.4, ncol=1, borderaxespad=0.3, labelspacing=0.3)

fig.tight_layout(w_pad=1.6)

# --------------------------------------------------------------------------- #
#  Save + printed stats
# --------------------------------------------------------------------------- #
out = be[["architecture", "location", "climate", "best_method", "best_total",
          "diesel_fixed_cost", "diesel_gallons", "breakeven_usd_per_gal",
          "breakeven_std", "n_folds"]].copy()

caption = (
    f"Diesel break-even analysis at the representative Med VoLL operating point "
    f"(HVAC $3/kWh, critical $100/kWh; diesel is VoLL-independent); annualized USD/yr "
    f"on test-split years. Diesel total cost is recomputed at any delivered price p as "
    f"fixed_cost + gallons*p (gallons price-independent). (a) For {S.CLIMATE_LABEL[REP]}, "
    f"the diesel cost line (grey dashed; thin grey = other climates) crosses each "
    f"architecture's recommended SO-CVaR design at "
    f"{be_rep['breakeven_usd_per_gal'].min():.1f}-{be_rep['breakeven_usd_per_gal'].max():.1f} "
    f"$/gal, far left of the routine forward-delivery "
    f"(${LOWBAND[0]:.0f}-{LOWBAND[1]:.0f}/gal) and contested-delivery "
    f"(${BAND[0]:.0f}-{BAND[1]:.0f}/gal) bands. (b) Break-even delivered diesel price "
    f"for the recommended SO-CVaR design in every climate (bars: hatch = architecture; "
    f"error bars = +/-1 SD across folds); it stays below ~9 $/gal everywhere -- below "
    f"even the routine forward-delivery band."
)
S.save_fig(fig, "fig6_diesel_breakeven", section="main", data=out, caption=caption)

# printed stats
print("\n=== BREAKEVEN DELIVERED DIESEL PRICE ($/gal), Med VoLL, test split ===")
pv = be.pivot_table(index="location", columns="architecture",
                    values="breakeven_usd_per_gal", observed=True)
print(pv.round(2).to_string())
print("\nPer-row detail:")
for _, r in be.iterrows():
    print(f"  {r['climate']:<18s} {r['architecture']:<4s} best={r['best_method']:<8s}"
          f" breakeven={r['breakeven_usd_per_gal']:6.2f} $/gal (+/-{r['breakeven_std']:.2f})"
          f"  [ren best=${r['best_total']:.0f}, fixed=${r['diesel_fixed_cost']:.0f},"
          f" gal={r['diesel_gallons']:.0f}]")
allbe = be["breakeven_usd_per_gal"]
print(f"\nRange across all climates/architectures: {allbe.min():.2f} - {allbe.max():.2f} $/gal")
print(f"All values are {BAND[0]/allbe.max():.0f}x-{BAND[1]/allbe.min():.0f}x below the "
      f"${BAND[0]:.0f}-{BAND[1]:.0f}/gal plausible resupply band.")
