
from __future__ import annotations

from pathlib import Path
import math
import re
from typing import Dict, List, Tuple, Optional

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium
from plotly.subplots import make_subplots

st.set_page_config(page_title="Hamra Regional Exploration Dashboard", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
NULLS = {-999.25, -999.2500, -9999.0, -9999.25}

WELL_FILES = {
    "Haraz-1": DATA_DIR / "Haraz-1_SLAM.csv",
    "Haraz-2": DATA_DIR / "HARAZ-2_SLAM.las",
    "Haraz-3": DATA_DIR / "Haraz-3_SLAM.csv",
    "Haraz-4": DATA_DIR / "Haraz-4_SLAM.csv",
    "Haraz-5": DATA_DIR / "Haraz-5_SLAM.csv",
}

LITH_COLORS = {
    "Clean Sand": "#E9C46A",
    "Shaly Sand": "#F4A261",
    "Shale": "#4F5D75",
    "Dense / Carbonate": "#8D99AE",
    "Unknown": "#CED4DA",
}

FORMATION_COLORS = {
    "Bentiu": "#E9C46A",
    "Aradeiba": "#F4A261",
    "Abu Gabra": "#6C757D",
}

THEME_CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1rem;}
div[data-testid="stMetricValue"] {color: #0f4c5c;}
div[data-testid="stMetricLabel"] {color: #5c6770;}
.main-title {
    padding: 1rem 1.25rem;
    border-radius: 18px;
    background: linear-gradient(90deg, rgba(15,76,92,0.12), rgba(201,162,39,0.20));
    border: 1px solid rgba(15,76,92,0.15);
    margin-bottom: 1rem;
}
.section-card {
    padding: 0.9rem 1rem;
    border-radius: 16px;
    background: #fbfcfd;
    border: 1px solid #e6eaee;
}
.small-note {
    color: #667085;
    font-size: 0.92rem;
}
</style>
"""


def dms_to_decimal(text: str) -> Optional[float]:
    m = re.search(r"(\d+)\s+(\d+)'\s+([\d.]+)\"\s*([NSEW])", str(text))
    if not m:
        return None
    deg, minute, second, hemi = m.groups()
    value = int(deg) + int(minute) / 60 + float(second) / 3600
    if hemi in {"S", "W"}:
        value *= -1
    return value


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore", encoding="utf-8")


def parse_las_like(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    lines = read_text(path).splitlines()
    curves: List[str] = []
    meta: Dict[str, str] = {}
    in_curve = False
    in_ascii = False
    data_rows: List[List[float]] = []

    for line in lines:
        raw = line.rstrip("\n")
        stripped = raw.strip()
        upper = stripped.upper()

        if upper.startswith("~CURVE"):
            in_curve = True
            in_ascii = False
            continue
        if upper.startswith("~A") or upper.startswith("~ASCII"):
            in_ascii = True
            in_curve = False
            continue
        if stripped.startswith("~"):
            in_curve = False

        if in_curve:
            if not stripped or stripped.startswith("#"):
                continue
            m = re.match(r"([A-Za-z0-9_]+)\s*\.", stripped)
            if m:
                curves.append(m.group(1).upper())
            continue

        if not in_ascii:
            mm = re.match(r"\s*([A-Za-z0-9_]+)\s*\.([^:]*)\s*:\s*(.*)", raw)
            if mm:
                key, value, _desc = mm.groups()
                meta[key.strip().upper()] = value.strip()
            continue

        if in_ascii:
            if not stripped or stripped.startswith("#"):
                continue
            try:
                data_rows.append([float(x) for x in stripped.split()])
            except ValueError:
                continue

    if not data_rows:
        raise ValueError(f"No ASCII data found in {path.name}")

    width = max(len(r) for r in data_rows)
    if not curves:
        curves = [f"CURVE_{i+1}" for i in range(width)]
    width = min(width, len(curves))
    clean_rows = [r[:width] + [np.nan] * max(0, width - len(r[:width])) for r in data_rows]
    return pd.DataFrame(clean_rows, columns=curves[:width]), meta


def parse_csv_log(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    text = read_text(path)
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    sep = "," if first_line.count(",") > 2 else r"\s+"
    engine = "python" if sep == r"\s+" else "c"
    df = pd.read_csv(path, sep=sep, engine=engine)
    meta = {"WELL": path.stem.replace("_SLAM", "")}
    return df, meta


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().upper() for c in out.columns]
    rename = {
        "DT24": "DT",
        "CNCF": "CNC",
        "NPHI": "CNC",
        "SPBD": "SPBR",
        "RT": "RD",
        "RHOB": "ZDEN",
        "DEPT": "DEPTH",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace(list(NULLS), np.nan)
    return out


def infer_lithology(row: pd.Series) -> str:
    gr = row.get("GR", np.nan)
    pe = row.get("PE", np.nan)
    zden = row.get("ZDEN", np.nan)
    cnc = row.get("CNC", np.nan)
    if pd.isna(gr):
        return "Unknown"
    if gr >= 100:
        return "Shale"
    if gr < 45 and ((pd.notna(pe) and pe > 3.8) or (pd.notna(zden) and zden > 2.58)):
        return "Dense / Carbonate"
    if gr < 75:
        if pd.notna(cnc) and cnc > 0.30:
            return "Shaly Sand"
        return "Clean Sand"
    return "Shaly Sand"


def assign_formations(depth_series: pd.Series) -> pd.Series:
    dmin, dmax = float(depth_series.min()), float(depth_series.max())
    span = max(dmax - dmin, 1.0)
    q1 = dmin + 0.33 * span
    q2 = dmin + 0.66 * span
    return pd.Series(
        np.where(depth_series <= q1, "Bentiu", np.where(depth_series <= q2, "Aradeiba", "Abu Gabra")),
        index=depth_series.index,
    )


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "GR" in out.columns and out["GR"].notna().any():
        gr = out["GR"]
        gr_min = float(np.nanpercentile(gr.dropna(), 5))
        gr_max = float(np.nanpercentile(gr.dropna(), 95))
        denom = max(gr_max - gr_min, 1e-6)
        igr = ((gr - gr_min) / denom).clip(0, 1)
        out["IGR"] = igr
        out["VSH"] = (0.083 * ((2 ** (3.7 * igr)) - 1)).clip(0, 1)
    else:
        out["IGR"] = np.nan
        out["VSH"] = np.nan

    if "ZDEN" in out.columns:
        out["PHID"] = ((2.65 - out["ZDEN"]) / (2.65 - 1.0)).clip(-0.15, 0.45)
    else:
        out["PHID"] = np.nan

    if "CNC" in out.columns:
        cnc = out["CNC"].copy()
        if cnc.dropna().median() > 1.5:
            cnc = cnc / 100.0
        out["NPHI"] = cnc.clip(-0.15, 0.60)
    else:
        out["NPHI"] = np.nan

    out["PHIT"] = out[["PHID", "NPHI"]].mean(axis=1, skipna=True).clip(0, 0.45)

    if "RD" in out.columns and out["RD"].notna().any():
        rt = out["RD"].clip(lower=0.01)
        phi = out["PHIT"].clip(lower=0.01)
        sw = ((1.0 * 0.08) / (rt * (phi ** 2.0))).pow(0.5)
        out["SW"] = sw.clip(0, 1)
    else:
        out["SW"] = np.nan

    out["LITHOLOGY"] = out.apply(infer_lithology, axis=1)
    out["PAY_FLAG"] = (
        (out.get("GR", pd.Series(index=out.index, dtype=float)).fillna(999) < 75)
        & (out.get("RD", pd.Series(index=out.index, dtype=float)).fillna(0) > 8)
        & (out.get("PHIT", pd.Series(index=out.index, dtype=float)).fillna(0) > 0.08)
        & (out.get("VSH", pd.Series(index=out.index, dtype=float)).fillna(1) < 0.45)
        & (out.get("SW", pd.Series(index=out.index, dtype=float)).fillna(1) < 0.65)
    )

    if "DEPTH" in out.columns:
        out["FORMATION"] = assign_formations(out["DEPTH"])
    else:
        out["FORMATION"] = "Unknown"

    out["RES_QUALITY_SCORE"] = (
        0.35 * out["PHIT"].fillna(0).clip(0, 0.35) / 0.35
        + 0.25 * out["RD"].fillna(0).clip(0, 40) / 40
        + 0.20 * (1 - out["VSH"].fillna(1))
        + 0.20 * (1 - out["SW"].fillna(1))
    ).clip(0, 1)

    out["FLUID_HINT"] = np.where(
        (out["RD"].fillna(0) > 15) & (out["SW"].fillna(1) < 0.45), "Possible HC",
        np.where((out["VSH"].fillna(1) > 0.5), "Shaly / Tight", "Water-prone"),
    )

    return out


def estimate_coords(well: str, meta: Dict[str, str]) -> Tuple[float, float]:
    lat = lon = None
    for key, value in meta.items():
        if key == "LATI":
            lat = dms_to_decimal(value)
        elif key == "LONG":
            lon = dms_to_decimal(value)
    if lat is not None and lon is not None:
        return lat, lon

    base_lat, base_lon = 10.5316, 28.0863
    offsets = {
        "Haraz-1": (-0.018, -0.015),
        "Haraz-2": (0.000, 0.000),
        "Haraz-3": (0.011, 0.017),
        "Haraz-4": (-0.012, 0.021),
        "Haraz-5": (0.019, -0.010),
    }
    dlat, dlon = offsets.get(well, (0.0, 0.0))
    return base_lat + dlat, base_lon + dlon


@st.cache_data(show_spinner=False)
def load_all_wells() -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    wells: Dict[str, pd.DataFrame] = {}
    rows = []

    for well_name, path in WELL_FILES.items():
        text_head = read_text(path)[:200].lstrip()
        if path.suffix.lower() == ".las" or text_head.startswith("~"):
            df, meta = parse_las_like(path)
        else:
            df, meta = parse_csv_log(path)

        df = standardize_columns(df)
        if "DEPTH" not in df.columns:
            continue

        df = add_derived_features(df)
        df = df.sort_values("DEPTH").reset_index(drop=True)

        lat, lon = estimate_coords(well_name, meta)
        df["WELL"] = well_name
        df["LAT"] = lat
        df["LON"] = lon

        depth_step = float(df["DEPTH"].diff().median()) if df["DEPTH"].diff().notna().any() else 0.1524
        gross = max(df["DEPTH"].max() - df["DEPTH"].min(), 0)
        net = float(df["PAY_FLAG"].fillna(False).sum() * abs(depth_step))
        wells[well_name] = df

        rows.append({
            "Well": well_name,
            "Latitude": lat,
            "Longitude": lon,
            "Top Depth": float(df["DEPTH"].min()),
            "Base Depth": float(df["DEPTH"].max()),
            "Gross Thickness": gross,
            "Net Pay": net,
            "Net/Gross": net / gross if gross else np.nan,
            "Samples": int(len(df)),
            "Mean GR": float(df["GR"].mean()) if "GR" in df else np.nan,
            "Mean RD": float(df["RD"].mean()) if "RD" in df else np.nan,
            "Mean PHIT": float(df["PHIT"].mean()),
            "Mean SW": float(df["SW"].mean()),
            "Pay Fraction": float(df["PAY_FLAG"].mean()),
            "Reservoir Score": float(df["RES_QUALITY_SCORE"].mean()),
        })

    return wells, pd.DataFrame(rows)


def build_multitrack(df: pd.DataFrame, well: str, depth_range: Tuple[float, float]) -> go.Figure:
    dff = df[(df["DEPTH"] >= depth_range[0]) & (df["DEPTH"] <= depth_range[1])].copy()
    fig = make_subplots(rows=1, cols=6, shared_yaxes=True, horizontal_spacing=0.03)

    tracks = [
        ("GR", "Gamma Ray", [0, 150], "line"),
        ("RD", "Resistivity", None, "line"),
        ("PHIT", "Porosity", [0, 0.35], "line"),
        ("SW", "Sw", [0, 1], "line"),
        ("VSH", "Vsh", [0, 1], "line"),
    ]

    for idx, (col, title, x_range, _) in enumerate(tracks, start=1):
        if col not in dff.columns:
            continue
        fig.add_trace(
            go.Scattergl(x=dff[col], y=dff["DEPTH"], mode="lines", name=title, showlegend=False),
            row=1, col=idx
        )
        fig.update_xaxes(title_text=title, row=1, col=idx)
        if x_range:
            fig.update_xaxes(range=x_range, row=1, col=idx)
        if col == "RD":
            vals = dff[col].dropna()
            vals = vals[vals > 0]
            if not vals.empty:
                fig.update_xaxes(type="log", row=1, col=idx)

    lith = dff[["DEPTH", "LITHOLOGY", "FORMATION"]].dropna()
    if not lith.empty:
        fig.add_trace(
            go.Scattergl(
                x=np.ones(len(lith)),
                y=lith["DEPTH"],
                mode="markers",
                marker=dict(
                    color=[LITH_COLORS.get(x, "#CED4DA") for x in lith["LITHOLOGY"]],
                    symbol="square",
                    size=7,
                ),
                text=lith["LITHOLOGY"] + " | " + lith["FORMATION"],
                hovertemplate="Depth: %{y:.2f} m<br>%{text}<extra></extra>",
                showlegend=False,
                name="Lithology"
            ),
            row=1, col=6
        )
        fig.update_xaxes(visible=False, row=1, col=6, title_text="Lithology")

    for form, color in FORMATION_COLORS.items():
        form_rows = dff[dff["FORMATION"] == form]
        if not form_rows.empty:
            fig.add_hrect(
                y0=float(form_rows["DEPTH"].min()),
                y1=float(form_rows["DEPTH"].max()),
                fillcolor=color,
                opacity=0.05,
                line_width=0,
            )

    fig.update_yaxes(autorange="reversed", title_text="Depth (m)", row=1, col=1)
    for col in range(2, 7):
        fig.update_yaxes(autorange="reversed", row=1, col=col, showticklabels=False)
    fig.update_layout(
        height=900,
        title=f"{well} | Multitrack Petrophysical Panel",
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return fig


def build_3d_plot(all_df: pd.DataFrame, sample_step: int = 10) -> go.Figure:
    dff = all_df.iloc[::sample_step].copy()
    dff["TVDSS_VIS"] = -dff["DEPTH"]
    fig = px.scatter_3d(
        dff,
        x="LON",
        y="LAT",
        z="TVDSS_VIS",
        color="LITHOLOGY",
        symbol="WELL",
        color_discrete_map=LITH_COLORS,
        hover_data=["WELL", "DEPTH", "FORMATION", "GR", "RD", "PHIT", "SW", "RES_QUALITY_SCORE"],
        opacity=0.78,
        title="3D Lithology & Wellbore Positioning",
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=760, scene=dict(zaxis_title="Depth (negative m)"))
    return fig


def build_map(summary_df: pd.DataFrame) -> folium.Map:
    center = [summary_df["Latitude"].mean(), summary_df["Longitude"].mean()]
    fmap = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
    for _, row in summary_df.iterrows():
        popup = folium.Popup(
            f"<b>{row['Well']}</b><br>"
            f"Gross: {row['Gross Thickness']:.1f} m<br>"
            f"Net Pay: {row['Net Pay']:.1f} m<br>"
            f"N/G: {100 * row['Net/Gross']:.1f}%<br>"
            f"Mean PHIT: {row['Mean PHIT']:.3f}<br>"
            f"Reservoir Score: {row['Reservoir Score']:.2f}",
            max_width=320,
        )
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=9,
            popup=popup,
            tooltip=row["Well"],
            fill=True,
            color="#0f4c5c",
            fill_color="#c9a227",
            fill_opacity=0.9,
            weight=2,
        ).add_to(fmap)
    return fmap


def crossplot(df: pd.DataFrame, x_col: str, y_col: str, color_col: str) -> go.Figure:
    cols = [x_col, y_col, color_col, "WELL", "DEPTH"]
    dff = df[cols].dropna().iloc[::8].copy()
    fig = px.scatter(
        dff,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=["WELL", "DEPTH"],
        color_discrete_map=LITH_COLORS if color_col == "LITHOLOGY" else None,
        opacity=0.75,
        title=f"Dynamic Cross-Plot | {x_col} vs {y_col}",
    )
    fig.update_layout(height=620)
    return fig


def corr_heatmap(df: pd.DataFrame, cols: List[str]) -> go.Figure:
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation Matrix")
    fig.update_layout(height=680)
    return fig


def tops_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well, dff in df.groupby("WELL"):
        for formation, fdf in dff.groupby("FORMATION"):
            rows.append({
                "Well": well,
                "Formation": formation,
                "Top": float(fdf["DEPTH"].min()),
                "Base": float(fdf["DEPTH"].max()),
                "Thickness": float(fdf["DEPTH"].max() - fdf["DEPTH"].min()),
                "Mean PHIT": float(fdf["PHIT"].mean()),
                "Mean SW": float(fdf["SW"].mean()),
                "Pay Fraction": float(fdf["PAY_FLAG"].mean()),
            })
    order = {"Bentiu": 0, "Aradeiba": 1, "Abu Gabra": 2}
    out = pd.DataFrame(rows)
    out["order"] = out["Formation"].map(order)
    return out.sort_values(["Well", "order"]).drop(columns="order")


def pay_summary_by_well_and_formation(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (well, formation), dff in all_df.groupby(["WELL", "FORMATION"]):
        depth_step = float(dff["DEPTH"].diff().median()) if dff["DEPTH"].diff().notna().any() else 0.1524
        gross = max(float(dff["DEPTH"].max() - dff["DEPTH"].min()), 0.0)
        net = float(dff["PAY_FLAG"].fillna(False).sum() * abs(depth_step))
        rows.append({
            "Well": well,
            "Formation": formation,
            "Gross Thickness": gross,
            "Net Pay": net,
            "Net/Gross": net / gross if gross else np.nan,
            "Mean PHIT": float(dff["PHIT"].mean()),
            "Mean SW": float(dff["SW"].mean()),
            "Mean VSH": float(dff["VSH"].mean()),
            "Reservoir Score": float(dff["RES_QUALITY_SCORE"].mean()),
            "Possible HC Fraction": float((dff["FLUID_HINT"] == "Possible HC").mean()),
        })
    return pd.DataFrame(rows).sort_values(["Reservoir Score", "Net Pay"], ascending=[False, False])


def ranking_table(pay_df: pd.DataFrame) -> pd.DataFrame:
    rank = pay_df.copy()
    rank["Ranking Index"] = (
        0.40 * rank["Reservoir Score"].fillna(0)
        + 0.30 * rank["Net/Gross"].fillna(0).clip(0, 1)
        + 0.20 * rank["Mean PHIT"].fillna(0).clip(0, 0.35) / 0.35
        + 0.10 * (1 - rank["Mean SW"].fillna(1))
    )
    return rank.sort_values("Ranking Index", ascending=False).reset_index(drop=True)


def bar_summary(summary_df: pd.DataFrame) -> go.Figure:
    plot_df = summary_df.sort_values("Reservoir Score", ascending=False)
    fig = px.bar(
        plot_df,
        x="Well",
        y=["Net Pay", "Gross Thickness"],
        barmode="group",
        title="Net Pay vs Gross Thickness by Well",
    )
    fig.update_layout(height=480)
    return fig


def formation_quality_chart(rank_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        rank_df.head(12),
        x="Well",
        y="Ranking Index",
        color="Formation",
        hover_data=["Net Pay", "Net/Gross", "Mean PHIT", "Mean SW", "Reservoir Score"],
        color_discrete_map=FORMATION_COLORS,
        title="Reservoir Quality Ranking",
    )
    fig.update_layout(height=540)
    return fig


def add_header():
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="main-title">
            <h2 style="margin:0;color:#0f4c5c;">Full-Scale Regional Exploration Dashboard</h2>
            <div class="small-note">Professional multi-well exploration dashboard for petrophysical screening, 3D well context, field mapping, formation summaries, and reservoir ranking.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    add_header()
    wells, summary_df = load_all_wells()
    all_df = pd.concat(wells.values(), ignore_index=True)
    pay_df = pay_summary_by_well_and_formation(all_df)
    rank_df = ranking_table(pay_df)
    top_df = tops_table(all_df)

    st.caption("Using the uploaded Haraz wells as the working regional dataset template for the requested Hamra-style dashboard. Coordinates for non-LAS wells are relative placeholders and can be replaced with surveyed field positions.")

    with st.sidebar:
        st.header("Dashboard Controls")
        selected_well = st.selectbox("Select well", list(wells.keys()), index=max(list(wells.keys()).index("Haraz-5"), 0) if "Haraz-5" in wells else 0)
        well_df = wells[selected_well]
        dmin = float(well_df["DEPTH"].min())
        dmax = float(well_df["DEPTH"].max())
        depth_range = st.slider("Depth window (m)", min_value=dmin, max_value=dmax, value=(dmin, dmax))
        sample_step = st.slider("3D downsample", 5, 50, 12, 1)
        map_metric = st.selectbox("Map pop-up emphasis", ["Net Pay", "Reservoir Score", "Mean PHIT"])
        st.markdown("---")
        st.markdown("**Screening cutoffs**")
        st.write("GR < 75 API")
        st.write("RD > 8 ohm·m")
        st.write("PHIT > 0.08")
        st.write("VSH < 0.45")
        st.write("SW < 0.65")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wells", f"{summary_df.shape[0]}")
    c2.metric("Total Samples", f"{len(all_df):,}")
    c3.metric("Avg Porosity", f"{all_df['PHIT'].mean():.3f}")
    c4.metric("Field Pay Fraction", f"{100 * all_df['PAY_FLAG'].mean():.1f}%")

    t1, t2, t3, t4, t5, t6, t7 = st.tabs([
        "Multitrack Logs",
        "3D Lithology",
        "Geospatial Map",
        "Exploratory Analytics",
        "Net Pay Summary",
        "Formation Tops",
        "Reservoir Ranking",
    ])

    with t1:
        st.plotly_chart(build_multitrack(well_df, selected_well, depth_range), use_container_width=True)
        a, b = st.columns([1.1, 1.2])
        with a:
            lith_counts = (
                well_df[(well_df["DEPTH"] >= depth_range[0]) & (well_df["DEPTH"] <= depth_range[1])]
                ["LITHOLOGY"].value_counts(dropna=False).rename_axis("Lithology").reset_index(name="Samples")
            )
            st.dataframe(lith_counts, use_container_width=True)
        with b:
            form_counts = (
                well_df[(well_df["DEPTH"] >= depth_range[0]) & (well_df["DEPTH"] <= depth_range[1])]
                .groupby("FORMATION")[["PHIT", "SW", "VSH"]].mean().round(3).reset_index()
            )
            st.dataframe(form_counts, use_container_width=True)

    with t2:
        st.plotly_chart(build_3d_plot(all_df, sample_step=sample_step), use_container_width=True)

    with t3:
        st_folium(build_map(summary_df), width=None, height=680)
        st.dataframe(summary_df.round(3), use_container_width=True)

    with t4:
        left, right, color_side = st.columns(3)
        numeric_cols = [c for c in ["GR", "RD", "PHIT", "SW", "VSH", "ZDEN", "CNC", "DT", "PE", "RES_QUALITY_SCORE"] if c in all_df.columns]
        with left:
            x_col = st.selectbox("X-axis", numeric_cols, index=0)
        with right:
            y_col = st.selectbox("Y-axis", numeric_cols, index=min(2, len(numeric_cols) - 1))
        with color_side:
            color_col = st.selectbox("Color by", ["LITHOLOGY", "WELL", "PAY_FLAG", "FORMATION", "FLUID_HINT"], index=0)
        st.plotly_chart(crossplot(all_df, x_col, y_col, color_col), use_container_width=True)
        st.plotly_chart(corr_heatmap(all_df, numeric_cols), use_container_width=True)

    with t5:
        st.plotly_chart(bar_summary(summary_df), use_container_width=True)
        st.dataframe(pay_df.round(3), use_container_width=True)

    with t6:
        st.dataframe(top_df.round(3), use_container_width=True)
        tops_fig = px.timeline(
            top_df,
            x_start="Top",
            x_end="Base",
            y="Well",
            color="Formation",
            color_discrete_map=FORMATION_COLORS,
            title="Formation Tops / Thickness Overview",
        )
        tops_fig.update_yaxes(autorange="reversed")
        tops_fig.update_layout(height=520)
        st.plotly_chart(tops_fig, use_container_width=True)
        st.markdown("<div class='small-note'>Formation names here are screening-style regional packages assigned from each well's depth distribution for dashboard demonstration. Replace with interpreted tops for final subsurface work.</div>", unsafe_allow_html=True)

    with t7:
        st.plotly_chart(formation_quality_chart(rank_df), use_container_width=True)
        st.dataframe(rank_df.round(3), use_container_width=True)
        st.subheader("Integrated dataset preview")
        preview_cols = [c for c in ["WELL", "DEPTH", "FORMATION", "GR", "RD", "PHIT", "SW", "VSH", "LITHOLOGY", "FLUID_HINT", "PAY_FLAG", "RES_QUALITY_SCORE", "LAT", "LON"] if c in all_df.columns]
        st.dataframe(all_df[preview_cols].head(600), use_container_width=True)

if __name__ == "__main__":
    main()
