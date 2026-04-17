"""
SPR LOD Analyzer
================
Calculation logic:
  1. User selects baseline1/sig1, baseline2/sig2, ... intervals on the sensorgram.
  2. For each channel:
       ΔIntensity_i = mean(sig_i) - mean(baseline_i)
       ΔRIU_i       = RIU_sig_i   - RIU_baseline1      (baseline RIU is always the first one)
  3. Fit  ΔIntensity = slope * ΔRIU + intercept  (regression line)
  4. 3σ = 3 * std(baseline1_intensity)
  5. LOD_ΔRIU  = (3σ - intercept) / slope
  6. From calibration arrays, build delta-based calibration:
       ΔRIU_cal_i  = r[i] - r[0]      (i = 1, 2, ...)
       Δ%_cal_i    = pct[i] - pct[0]
       ΔM_cal_i    = m[i]  - m[0]
       Fit  Δ% = f(ΔRIU)  →  LOD_%  = f(LOD_ΔRIU)
       Fit  ΔM = f(ΔRIU)  →  LOD_M  = f(LOD_ΔRIU)
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import SpanSelector
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Default calibration constants
# ---------------------------------------------------------------------------
DEFAULT_R_VALUES   = np.array([1.3280, 1.3287, 1.3290, 1.3294, 1.3298, 1.3305], dtype=float)
DEFAULT_PERCENTAGES = np.array([0,      0.5,    1.0,    1.5,    2.0,   2.5],    dtype=float)
DEFAULT_M_VALUES   = np.array([0, 0.089258, 0.178652, 0.267977, 0.357303, 0.44629], dtype=float)

SELECTION_TYPES = ("baseline", "sig")
INTENSITY_LABEL  = "Intensity (TM/TE)"
DELTA_INT_LABEL  = "ΔIntensity (TM/TE)"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Selection(NamedTuple):
    kind:  str    # "baseline" | "sig"
    label: str
    start: float
    end:   float


class Calibration(NamedTuple):
    r_values:    np.ndarray
    percentages: np.ndarray
    m_values:    np.ndarray


class ChannelResult(NamedTuple):
    y_values:    np.ndarray   # ΔIntensity for each pair
    x_riu:       np.ndarray   # ΔRIU for each pair
    slope:       float
    intercept:   float
    r_squared:   float
    sigma3:      float        # 3 * std(baseline1)
    lod_delta_riu: float      # LOD expressed as ΔRIU
    lod_percent:   float
    lod_m:         float


class AnalysisResult(NamedTuple):
    pair_count:        int
    channel_results:   dict[str, ChannelResult]
    delta_riu_to_pct_slope:     float
    delta_riu_to_pct_intercept: float
    delta_riu_to_m_slope:       float
    delta_riu_to_m_intercept:   float


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def fmt_array(arr: np.ndarray) -> str:
    return ", ".join(f"{v:g}" for v in arr)


def parse_array(text: str, name: str) -> np.ndarray:
    try:
        values = [float(p.strip()) for p in text.split(",") if p.strip()]
    except ValueError as exc:
        raise ValueError(f"{name}: comma-separated numbers required.") from exc
    if len(values) < 2:
        raise ValueError(f"{name}: need at least 2 values.")
    return np.array(values, dtype=float)


def load_data(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        for enc in ("utf-8-sig", "utf-8", "cp950", "big5", "latin1"):
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                pass
        raise ValueError(f"Cannot decode CSV: {path.name}")
    raise ValueError(f"Unsupported format: {path.suffix}")


def find_time_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower().startswith("sampletime"):
            return col
    raise ValueError("No column starting with 'sampletime' found.")


def find_intensity_columns(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame | None]:
    """
    Returns (column_names, extra_df_or_None).
    If TM/TE pairs are detected, extra_df contains the ratio columns.
    """
    intensity_cols = [c for c in df.columns if c.lower().startswith("intensity")]
    if intensity_cols:
        return intensity_cols, None

    tm_cols = [c for c in df.columns if c.lower().startswith("tm_intensity")]
    te_cols = [c for c in df.columns if c.lower().startswith("te_intensity")]
    if tm_cols and te_cols and len(tm_cols) == len(te_cols):
        ratio_df = pd.DataFrame(
            {f"Intensity{i}": df[tm] / df[te]
             for i, (tm, te) in enumerate(zip(tm_cols, te_cols), 1)}
        )
        return list(ratio_df.columns), ratio_df

    raise ValueError("No Intensity or TM/TE intensity columns found.")


def build_display_df(
    df: pd.DataFrame,
    time_col: str,
    intensity_cols: list[str],
    extra_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if extra_df is not None:
        out = pd.concat([df[[time_col]].copy(), extra_df], axis=1)
    else:
        out = df[[time_col] + intensity_cols].copy()
    return out.dropna().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def interval_mask(time_arr: np.ndarray, start: float, end: float) -> np.ndarray:
    return (time_arr >= start) & (time_arr <= end)


def detrend(
    display_df: pd.DataFrame,
    time_col: str,
    intensity_cols: list[str],
    selections: list[Selection],
) -> pd.DataFrame:
    """
    Remove baseline drift by linearly interpolating between the mean of each
    baseline region and subtracting that trend from the whole time series.
    """
    corrected = display_df.copy()
    time_arr = corrected[time_col].to_numpy(dtype=float)
    baselines = [s for s in selections if s.kind == "baseline"]

    for col in intensity_cols:
        centers, means = [], []
        for sel in baselines:
            mask = interval_mask(time_arr, sel.start, sel.end)
            if mask.sum() < 2:
                raise ValueError(f"{sel.label}: fewer than 2 points in interval.")
            centers.append(time_arr[mask].mean())
            means.append(corrected.loc[mask, col].to_numpy(dtype=float).mean())

        centers = np.array(centers, dtype=float)
        means   = np.array(means,   dtype=float)
        idx     = np.argsort(centers)
        centers, means = centers[idx], means[idx]

        trend = (
            np.full(len(time_arr), means[0])
            if len(centers) == 1
            else np.interp(time_arr, centers, means, left=means[0], right=means[-1])
        )
        corrected[col] = corrected[col].to_numpy(dtype=float) - trend

    return corrected


def analyze(
    corrected_df: pd.DataFrame,
    time_col: str,
    intensity_cols: list[str],
    selections: list[Selection],
    calibration: Calibration,
) -> AnalysisResult:
    """
    Main LOD analysis.

    Pairing rule: baseline → sig → baseline → sig → ...
    ΔRIU_i = RIU(sig_i) - RIU(baseline1)   (baseline RIU is always index 0)
    """
    ordered = sorted(selections, key=lambda s: s.start)

    # Build (baseline, sig) pairs in order
    pairs: list[tuple[Selection, Selection]] = []
    pending_baseline: Selection | None = None
    for sel in ordered:
        if sel.kind == "baseline":
            pending_baseline = sel
        elif sel.kind == "sig" and pending_baseline is not None:
            pairs.append((pending_baseline, sel))
            pending_baseline = None

    if not pairs:
        raise ValueError("Need at least one baseline → sig pair.")
    if len(pairs) < 2:
        raise ValueError("Need at least two pairs to fit a regression line.")

    n = len(pairs)
    r  = calibration.r_values
    pct = calibration.percentages
    m   = calibration.m_values

    if n > len(r) - 1:
        raise ValueError(
            f"Only {len(r) - 1} pairs supported by the current calibration arrays."
        )

    # RIU values: index 0 = baseline1, indices 1..n = sig1..sigN
    riu_baseline1 = r[0]
    riu_sigs      = r[1 : 1 + n]          # absolute RIU for each sig
    x_delta_riu   = riu_sigs - riu_baseline1  # ΔRIU for regression x-axis

    # Calibration regression: ΔRIU → Δ% and ΔRIU → ΔM
    # Force through origin (ΔRIU=0 ↔ Δ%=0 by definition).
    # Slope = Σ(x·y) / Σ(x²)  (ordinary least squares, no intercept)
    cal_delta_riu = r[1:]  - r[0]    # ΔRIU for every calibration point after baseline
    cal_delta_pct = pct[1:] - pct[0] # Δ% for every calibration point after baseline
    cal_delta_m   = m[1:]  - m[0]    # ΔM  for every calibration point after baseline
    denom = float(np.dot(cal_delta_riu, cal_delta_riu))
    if np.isclose(denom, 0.0):
        raise ValueError("Calibration ΔRIU values are all zero.")
    pct_cal_slope = float(np.dot(cal_delta_riu, cal_delta_pct) / denom)
    m_cal_slope   = float(np.dot(cal_delta_riu, cal_delta_m)   / denom)

    time_arr = corrected_df[time_col].to_numpy(dtype=float)
    baseline1_sel = pairs[0][0]
    b1_mask = interval_mask(time_arr, baseline1_sel.start, baseline1_sel.end)

    channel_results: dict[str, ChannelResult] = {}

    for col in intensity_cols:
        # ΔIntensity for each pair
        y_delta = np.empty(n, dtype=float)
        for i, (b_sel, s_sel) in enumerate(pairs):
            b_mask = interval_mask(time_arr, b_sel.start, b_sel.end)
            s_mask = interval_mask(time_arr, s_sel.start, s_sel.end)

            b_vals = corrected_df.loc[b_mask, col].to_numpy(dtype=float)
            s_vals = corrected_df.loc[s_mask, col].to_numpy(dtype=float)

            if len(b_vals) < 2 or len(s_vals) < 2:
                raise ValueError(
                    f"{b_sel.label}/{s_sel.label}: too few points in interval."
                )
            y_delta[i] = s_vals.mean() - b_vals.mean()

        # Regression: ΔIntensity = slope * ΔRIU + intercept
        fit = linregress(x_delta_riu, y_delta)
        slope, intercept, r_sq = float(fit.slope), float(fit.intercept), float(fit.rvalue ** 2)

        if np.isclose(slope, 0.0, atol=1e-12):
            raise ValueError(f"{col}: regression slope is effectively zero.")

        # 3σ from baseline1 only
        b1_vals = corrected_df.loc[b1_mask, col].to_numpy(dtype=float)
        if len(b1_vals) < 2:
            raise ValueError(f"{baseline1_sel.label}: too few points to compute σ.")
        sigma3 = 3.0 * float(np.std(b1_vals, ddof=1))

        # LOD
        lod_delta_riu = (sigma3 - intercept) / slope
        lod_pct = pct_cal_slope * lod_delta_riu
        lod_m   = m_cal_slope   * lod_delta_riu

        channel_results[col] = ChannelResult(
            y_values      = y_delta,
            x_riu         = x_delta_riu,
            slope         = slope,
            intercept     = intercept,
            r_squared     = r_sq,
            sigma3        = sigma3,
            lod_delta_riu = lod_delta_riu,
            lod_percent   = lod_pct,
            lod_m         = lod_m,
        )

    return AnalysisResult(
        pair_count                  = n,
        channel_results             = channel_results,
        delta_riu_to_pct_slope      = pct_cal_slope,
        delta_riu_to_pct_intercept  = 0.0,
        delta_riu_to_m_slope        = m_cal_slope,
        delta_riu_to_m_intercept    = 0.0,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class LODAnalyzerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SPR LOD Analyzer")
        self.root.geometry("1500x940")

        # State
        self.file_path:        Path | None        = None
        self.display_df:       pd.DataFrame | None = None
        self.corrected_df:     pd.DataFrame | None = None
        self.time_col:         str | None          = None
        self.intensity_cols:   list[str]           = []
        self.selections:       list[Selection]     = []
        self.result:           AnalysisResult | None = None

        # Tk variables
        self.sv_file       = tk.StringVar(value="No file loaded")
        self.sv_status     = tk.StringVar(value="Load a sensorgram file, then drag on the top plot to mark intervals.")
        self.sv_sel_kind   = tk.StringVar(value="baseline")
        self.sv_r_values   = tk.StringVar(value=fmt_array(DEFAULT_R_VALUES))
        self.sv_percentages = tk.StringVar(value=fmt_array(DEFAULT_PERCENTAGES))
        self.sv_m_values   = tk.StringVar(value=fmt_array(DEFAULT_M_VALUES))

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = self.root
        container = ttk.Frame(root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        # ---- Controls ----
        ctrl = ttk.LabelFrame(container, text="Controls", padding=10)
        ctrl.pack(fill=tk.X, pady=(0, 8))

        # File row
        fr = ttk.Frame(ctrl)
        fr.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(fr, text="Load File", command=self._choose_file).pack(side=tk.LEFT)
        ttk.Label(fr, textvariable=self.sv_file).pack(side=tk.LEFT, padx=(10, 0))

        # Selection mode row
        mr = ttk.Frame(ctrl)
        mr.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(mr, text="Selection type:").pack(side=tk.LEFT)
        ttk.Combobox(
            mr, textvariable=self.sv_sel_kind,
            values=SELECTION_TYPES, state="readonly", width=10,
        ).pack(side=tk.LEFT, padx=(6, 14))
        for text, cmd in (
            ("Undo", self._undo),
            ("Clear", self._clear),
            ("Analyze", self._run_analysis),
        ):
            ttk.Button(mr, text=text, command=cmd).pack(side=tk.LEFT, padx=(0, 6))

        # Calibration inputs
        cal = ttk.LabelFrame(ctrl, text="Calibration Arrays (comma-separated, index 0 = baseline)", padding=8)
        cal.pack(fill=tk.X, pady=(0, 6))
        cal.columnconfigure(1, weight=1)
        for row_idx, (label, sv) in enumerate([
            ("R_VALUES (RIU)",    self.sv_r_values),
            ("PERCENTAGES (%)",   self.sv_percentages),
            ("M_VALUES (mol/L)",  self.sv_m_values),
        ]):
            ttk.Label(cal, text=label).grid(row=row_idx, column=0, sticky="w", padx=(0, 8), pady=2)
            ttk.Entry(cal, textvariable=sv).grid(row=row_idx, column=1, sticky="ew", pady=2)

        ttk.Label(ctrl, textvariable=self.sv_status, foreground="#444").pack(fill=tk.X, pady=(4, 0))

        # ---- Main content ----
        content = ttk.Frame(container)
        content.pack(fill=tk.BOTH, expand=True)

        left  = ttk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        right = ttk.Frame(content)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # Matplotlib figure
        self.fig = plt.Figure(figsize=(11, 8))
        gs = GridSpec(2, 2, figure=self.fig, height_ratios=[1.1, 1], width_ratios=[1.15, 1])
        self.ax_raw        = self.fig.add_subplot(gs[0, :])
        self.ax_corrected  = self.fig.add_subplot(gs[1, 0])
        self.ax_regression = self.fig.add_subplot(gs[1, 1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, left).update()

        self.span_selector = SpanSelector(
            self.ax_raw, self._on_span,
            "horizontal", useblit=True,
            props=dict(alpha=0.18, facecolor="tab:blue"),
            interactive=False, drag_from_anywhere=True,
        )

        # Selection list
        sf = ttk.LabelFrame(right, text="Selections", padding=6)
        sf.pack(fill=tk.BOTH, expand=True)
        self.lb_selections = tk.Listbox(sf, width=36, height=18)
        self.lb_selections.pack(fill=tk.BOTH, expand=True)

        # Result panel
        rf = ttk.LabelFrame(right, text="Results", padding=6)
        rf.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.txt_result = tk.Text(
            rf, width=44, height=26, state=tk.DISABLED,
            font=("Consolas", 9), wrap=tk.NONE,
        )
        self.txt_result.pack(fill=tk.BOTH, expand=True)

        self._refresh_plots()

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open sensorgram file",
            filetypes=[
                ("Excel / CSV", "*.xlsx *.xls *.csv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._load_file(Path(path))

    def _load_file(self, path: Path) -> None:
        try:
            df = load_data(path)
            time_col = find_time_column(df)
            intensity_cols, extra_df = find_intensity_columns(df)
            display_df = build_display_df(df, time_col, intensity_cols, extra_df)

            self.file_path      = path
            self.display_df     = display_df
            self.time_col       = time_col
            self.intensity_cols = intensity_cols
            self.corrected_df   = None
            self.result         = None
            self._clear(update_status=False)

            self.sv_file.set(str(path))
            self._set_status("File loaded. Drag on the raw plot to add baseline / sig intervals.")
            self._set_result_text("Select intervals and click Analyze.")
            self._refresh_plots()
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    # ------------------------------------------------------------------
    # Span selection
    # ------------------------------------------------------------------

    def _on_span(self, xmin: float, xmax: float) -> None:
        if self.display_df is None or xmax <= xmin:
            return
        kind  = self.sv_sel_kind.get()
        count = sum(1 for s in self.selections if s.kind == kind) + 1
        label = f"{kind}{count}"
        sel   = Selection(kind=kind, label=label, start=float(xmin), end=float(xmax))
        self.selections.append(sel)
        self._set_status(f"Added {label}: {xmin:.2f} → {xmax:.2f}")
        self._refresh_selection_list()
        self._refresh_plots()

    def _undo(self) -> None:
        if not self.selections:
            return
        removed = self.selections.pop()
        self.corrected_df = None
        self.result       = None
        self._set_status(f"Removed {removed.label}.")
        self._refresh_selection_list()
        self._refresh_plots()

    def _clear(self, update_status: bool = True) -> None:
        self.selections   = []
        self.corrected_df = None
        self.result       = None
        self._refresh_selection_list()
        if update_status:
            self._set_status("Selections cleared.")
        if hasattr(self, "ax_raw"):
            self._refresh_plots()

    def _refresh_selection_list(self) -> None:
        self.lb_selections.delete(0, tk.END)
        for s in self.selections:
            self.lb_selections.insert(tk.END, f"{s.label}: {s.start:.2f} – {s.end:.2f}")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _parse_calibration(self) -> Calibration:
        r   = parse_array(self.sv_r_values.get(),    "R_VALUES")
        pct = parse_array(self.sv_percentages.get(), "PERCENTAGES")
        m   = parse_array(self.sv_m_values.get(),    "M_VALUES")

        if not (len(r) == len(pct) == len(m)):
            raise ValueError("R_VALUES, PERCENTAGES, and M_VALUES must all have the same length.")
        if len(r) < 3:
            raise ValueError("At least 3 calibration points are required.")
        return Calibration(r_values=r, percentages=pct, m_values=m)

    def _run_analysis(self) -> None:
        if self.display_df is None:
            messagebox.showwarning("No data", "Load a file first.")
            return

        ordered = sorted(self.selections, key=lambda s: s.start)
        baselines = [s for s in ordered if s.kind == "baseline"]
        sigs      = [s for s in ordered if s.kind == "sig"]

        if not baselines:
            messagebox.showwarning("Missing baseline", "Select at least one baseline interval.")
            return
        if not sigs:
            messagebox.showwarning("Missing signal", "Select at least one sig interval.")
            return

        try:
            cal           = self._parse_calibration()
            corrected_df  = detrend(self.display_df, self.time_col, self.intensity_cols, ordered)
            result        = analyze(corrected_df, self.time_col, self.intensity_cols, ordered, cal)

            self.corrected_df = corrected_df
            self.result       = result

            self._set_status("Analysis complete.")
            self._refresh_plots()
            self._render_results()
        except Exception as exc:
            messagebox.showerror("Analysis failed", str(exc))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _refresh_plots(self) -> None:
        for ax in (self.ax_raw, self.ax_corrected, self.ax_regression):
            ax.clear()

        if self.display_df is None:
            for ax in (self.ax_raw, self.ax_corrected, self.ax_regression):
                ax.text(0.5, 0.5, "No data loaded", ha="center", va="center",
                        transform=ax.transAxes, color="#999")
                ax.set_xticks([]); ax.set_yticks([])
            self.canvas.draw_idle()
            return

        time_arr = self.display_df[self.time_col].to_numpy(dtype=float)
        colors_sel = {"baseline": "tab:green", "sig": "tab:orange"}

        # ---- Raw signal ----
        for col in self.intensity_cols:
            self.ax_raw.plot(time_arr, self.display_df[col], lw=0.9, label=col)

        for sel in sorted(self.selections, key=lambda s: s.start):
            self.ax_raw.axvspan(sel.start, sel.end, color=colors_sel[sel.kind], alpha=0.18)
            mid = (sel.start + sel.end) / 2
            ylim = self.ax_raw.get_ylim()
            self.ax_raw.text(mid, ylim[1], sel.label,
                             rotation=90, va="top", ha="center", fontsize=7.5)

        self.ax_raw.set_title("Raw Signal")
        self.ax_raw.set_xlabel("Time (s)")
        self.ax_raw.set_ylabel(INTENSITY_LABEL)
        self.ax_raw.grid(alpha=0.2)
        self.ax_raw.legend(loc="upper left", ncol=2, fontsize=8)

        # ---- Detrended signal ----
        src = self.corrected_df if self.corrected_df is not None else self.display_df
        for col in self.intensity_cols:
            self.ax_corrected.plot(time_arr, src[col], lw=0.9, label=col)
        self.ax_corrected.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        self.ax_corrected.set_title("Detrended Signal")
        self.ax_corrected.set_xlabel("Time (s)")
        self.ax_corrected.set_ylabel(INTENSITY_LABEL)
        self.ax_corrected.grid(alpha=0.2)
        self.ax_corrected.legend(loc="upper right", ncol=2, fontsize=8)

        # ---- Regression ----
        if self.result is None:
            self.ax_regression.text(0.5, 0.5, "Run Analyze to see regression",
                                    ha="center", va="center",
                                    transform=self.ax_regression.transAxes, color="#999")
            self.ax_regression.set_xticks([]); self.ax_regression.set_yticks([])
        else:
            lod_riu_values = []
            for col, ch in self.result.channel_results.items():
                lod_riu_values.append(ch.lod_delta_riu)
                x_line = np.linspace(ch.x_riu.min(), ch.x_riu.max(), 200)
                sc = self.ax_regression.scatter(ch.x_riu, ch.y_values, s=36, label=f"{col} (R²={ch.r_squared:.4f})", zorder=3)
                color = sc.get_facecolor()[0]
                self.ax_regression.plot(
                    x_line, ch.slope * x_line + ch.intercept,
                    lw=1.2, color=color,
                    label=f"{col}: y={ch.slope:.4f}x+{ch.intercept:.4f}",
                )
                # 3σ horizontal line
                self.ax_regression.axhline(ch.sigma3, lw=0.9, ls="--", color=color, alpha=0.6)
                # LOD point
                self.ax_regression.scatter(
                    [ch.lod_delta_riu], [ch.sigma3],
                    marker="x", s=72, linewidths=2, color=color, zorder=4,
                )

            # 顯示平均 LOD (RIU) 到圖表右下角
            avg_lod_riu = np.mean(lod_riu_values)
            self.ax_regression.text(
                0.98, 0.02, f"LOD (RIU): {avg_lod_riu:.5f}",
                transform=self.ax_regression.transAxes,
                fontsize=9, color="black", weight="bold",
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8, edgecolor="gray", linewidth=0.8),
            )

            self.ax_regression.set_title("ΔIntensity vs ΔRIU Regression")
            self.ax_regression.set_xlabel("ΔRIU  (RIU_sig − RIU_baseline1)")
            self.ax_regression.set_ylabel(DELTA_INT_LABEL)
            self.ax_regression.grid(alpha=0.2)
            self.ax_regression.legend(fontsize=7.5)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Result text
    # ------------------------------------------------------------------

    def _render_results(self) -> None:
        if self.result is None:
            self._set_result_text("Select intervals and click Analyze.")
            return

        r = self.result
        lines = [
            "=" * 40,
            "SPR LOD Analysis",
            "=" * 40,
            f"Pairs used : {r.pair_count}",
            "",
            "Calibration regressions (delta, forced through origin):",
            f"  Δ% = {r.delta_riu_to_pct_slope:.4f} × ΔRIU",
            f"  ΔM = {r.delta_riu_to_m_slope:.4f} × ΔRIU",
            "",
            "Regression model per channel:",
            "  ΔIntensity = slope * ΔRIU + intercept",
            "  ΔRIU = RIU(sig_i) - RIU(baseline1)",
            "  LOD criterion : ΔIntensity = 3σ(baseline1)",
            "",
        ]

        for col, ch in r.channel_results.items():
            lines += [
                f"── {col} " + "─" * (34 - len(col)),
                f"  slope         = {ch.slope:.6f}",
                f"  intercept     = {ch.intercept:.6f}",
                f"  R²            = {ch.r_squared:.6f}",
                f"  3σ(baseline1) = {ch.sigma3:.6f}",
                "",
                f"  LOD (ΔRIU)    = {ch.lod_delta_riu:.6f}",
                f"  LOD (%)       = {ch.lod_percent:.6f}",
                f"  LOD (mol/L)   = {ch.lod_m:.6f}",
                "",
            ]

        self._set_result_text("\n".join(lines))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _set_status(self, msg: str) -> None:
        self.sv_status.set(msg)

    def _set_result_text(self, text: str) -> None:
        self.txt_result.configure(state=tk.NORMAL)
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, text)
        self.txt_result.configure(state=tk.DISABLED)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    LODAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
