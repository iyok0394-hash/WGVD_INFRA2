import streamlit as st
import rasterio
from rasterio.mask import mask
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import tempfile
import os
import sys
import gc
from datetime import datetime
from fpdf import FPDF

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="W-GVD Enterprise Pro",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
[data-testid="stSidebar"], header[data-testid="stHeader"] {display:none;}
.stApp {background:#0a1120;}
h1,h2,h3,h4,p,label,div,span {color:#e2e8f0;}

.section-card {
    background:#111d35;
    padding:25px;
    border-radius:18px;
    border:1px solid #1e40af;
    margin-top:20px;
}

.metric-card {
    background:#111d35;
    padding:25px;
    border-radius:18px;
    text-align:center;
}

/* ========================= */
/* BUTTON STYLE */
/* ========================= */
.stButton > button {
    background:#2563eb !important;
    color:white !important;
    height:58px !important;
    font-size:15px !important;
    border-radius:12px !important;
    font-weight:bold !important;
}

/* DOWNLOAD BUTTON */
.stDownloadButton > button {
    background:#22c55e !important;
    color:white !important;
    height:50px !important;
    border-radius:12px !important;
    font-weight:bold !important;
}

/* ========================= */
/* STA SELECTBOX FIX */
/* ========================= */

/* Container utama */
div[data-baseweb="select"] > div {
    background-color: #2563eb !important;
    border-radius: 12px !important;
    border: none !important;
    min-height: 45px !important;
}

/* Saat focus / klik */
div[data-baseweb="select"] > div:focus,
div[data-baseweb="select"] > div:active {
    background-color: #1d4ed8 !important;
}

/* Text di dalam */
div[data-baseweb="select"] span {
    color: white !important;
    font-weight: 600 !important;
}

/* Icon dropdown */
div[data-baseweb="select"] svg {
    fill: white !important;
}

/* Dropdown list menu */
ul[role="listbox"] {
    background-color: #111d35 !important;
}

/* Item di dropdown */
ul[role="listbox"] li {
    background-color: #111d35 !important;
    color: white !important;
}

/* Hover item */
ul[role="listbox"] li:hover {
    background-color: #2563eb !important;
    color: white !important;
}

.footer {
    position:fixed;
    bottom:0;
    width:100%;
    text-align:center;
    padding:8px;
    font-size:11px;
    color:#94a3b8;
    background:#111d35;
    border-top:1px solid #1e40af;
}
</style>
""", unsafe_allow_html=True)
# =========================================================
# PROFILE SAMPLING (CROSS SECTION)
# =========================================================
def get_profile_data_centered(raster_src, line_geom, nodata_val):

    total_len = line_geom.length
    distances = np.linspace(0, total_len, 300)
    rel_dist = distances - (total_len / 2)

    points = [line_geom.interpolate(d) for d in distances]
    coords = [(p.x, p.y) for p in points]

    vals = []
    for v in raster_src.sample(coords):
        if v[0] != nodata_val and v[0] > -9000:
            vals.append(v[0])
        else:
            vals.append(np.nan)

    return rel_dist, np.array(vals, dtype="float32")
# =========================================================
# SESSION
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "metode" not in st.session_state:
    st.session_state.metode = None

def go_to(p):
    st.session_state.page = p
    st.rerun()

# =========================================================
# LONG PROFILE PROFESSIONAL
# =========================================================
def get_long_profile_professional(raster_src, line_geom, nodata_val, interval=5):

    total_len = line_geom.length
    distances = np.arange(0, total_len, interval)

    points = [line_geom.interpolate(d) for d in distances]
    coords = [(p.x, p.y) for p in points]

    vals = []

    for v in raster_src.sample(coords):
        if v[0] != nodata_val and v[0] > -9000:
            vals.append(v[0])
        else:
            vals.append(np.nan)

    return distances, np.array(vals, dtype="float32")

# =========================================================
# VOLUME CALCULATION (FULL - FIXED)
# =========================================================
@st.cache_data(show_spinner=False)
def hitung_volume_universal(path_dem_akhir, path_aoi=None,
                            path_dem_awal=None, path_cl=None,
                            mode="Base Elevasi", manual_elev=0.0):

    with rasterio.open(path_dem_akhir) as src_akhir:

        nodata = src_akhir.nodata
        dem_crs = src_akhir.crs

        # ===============================
        # AOI MASKING
        # ===============================
        if path_aoi:
            if path_aoi.lower().endswith(".zip"):
                aoi = gpd.read_file(f"zip://{path_aoi}")
            else:
                aoi = gpd.read_file(path_aoi)

            if aoi.crs != dem_crs:
                aoi = aoi.to_crs(dem_crs)

            img_akhir, out_transform = mask(
                src_akhir, aoi.geometry, crop=True
            )

            data_akhir = img_akhir[0].astype("float32")

        else:
            data_akhir = src_akhir.read(1).astype("float32")
            out_transform = src_akhir.transform
            aoi = None

        h, w = data_akhir.shape
        area_px = abs(src_akhir.res[0] * src_akhir.res[1])
        valid = (data_akhir != nodata) & (data_akhir > -9000)

        # ===============================
        # BASE SURFACE
        # ===============================
        if mode == "Base Elevasi":
            data_awal = np.full((h, w),
                                manual_elev,
                                dtype="float32")

        elif mode == "Lowest Point":
            data_awal = np.full((h, w),
                                np.nanmin(data_akhir[valid]),
                                dtype="float32")

        else:  # Surface to Surface
            with rasterio.open(path_dem_awal) as src_awal:
                with WarpedVRT(
                    src_awal,
                    crs=dem_crs,
                    transform=out_transform,
                    width=w,
                    height=h,
                    resampling=Resampling.bilinear
                ) as vrt:

                    data_awal = vrt.read(1).astype("float32")

        # ===============================
        # VOLUME CALCULATION
        # ===============================
        diff = data_akhir[valid] - data_awal[valid]

        left, top = out_transform * (0, 0)
        right, bottom = out_transform * (w, h)

        res = {
            "fill": np.sum(diff[diff > 0]) * area_px,
            "cut": abs(np.sum(diff[diff < 0])) * area_px,
            "area": diff.size * area_px,
            "data_plot": data_akhir,
            "data_base_plot": data_awal,
            "extent": [left, right, bottom, top],
            "crs": dem_crs,
            "nodata": nodata
        }

        # ===============================
        # CENTERLINE PROCESSING
        # ===============================
        if path_cl and aoi is not None:

            # --- Read centerline universal ---
            if path_cl.lower().endswith(".zip"):
                cl = gpd.read_file(f"zip://{path_cl}")
            else:
                cl = gpd.read_file(path_cl)

            cl = cl.to_crs(dem_crs)

            # --- Clip to AOI ---
            cl_clip = gpd.clip(cl, aoi)

            if not cl_clip.empty:

                line = cl_clip.geometry.iloc[0]
                cross = []

                for d in np.arange(0, line.length, 25):

                    p = line.interpolate(d)
                    p_next = line.interpolate(
                        min(d + 0.1, line.length)
                    )

                    ang = np.arctan2(
                        p_next.y - p.y,
                        p_next.x - p.x
                    ) + np.pi / 2

                    p1 = Point(
                        p.x + 2000 * np.cos(ang),
                        p.y + 2000 * np.sin(ang)
                    )

                    p2 = Point(
                        p.x - 2000 * np.cos(ang),
                        p.y - 2000 * np.sin(ang)
                    )

                    clip_line = LineString(
                        [p1, p2]
                    ).intersection(
                        aoi.geometry.iloc[0]
                    )

                    if not clip_line.is_empty:
                        cross.append({
                            "sta": f"STA {int(d)}",
                            "geometry": clip_line
                        })

                res.update({
                    "cl_geom": cl_clip,
                    "cross_gdf": gpd.GeoDataFrame(
                        cross,
                        crs=dem_crs
                    )
                })

        return res
# =========================================================
# HEADER
# =========================================================
col_logo, col_title = st.columns([1,5])

with col_logo:
    if os.path.exists("logowaskita.png"):
        st.image("logowaskita.png", width=120)

with col_title:
    st.markdown(
        "<h1>W-GVD | Geospatial Volumetric Dashboard</h1>",
        unsafe_allow_html=True
    )

# =========================================================
# MAIN ROUTER
# =========================================================

page = st.session_state.page


# =========================================================
# HOME
# =========================================================
if page == "home":

    with st.container():
        st.markdown("### W-GVD | Geospatial Volumetric Dashboard")

        st.markdown("""
W-GVD merupakan sistem analisis volume pekerjaan tanah (**earthwork**) 
berbasis Digital Elevation Model (DEM) yang dirancang untuk mendukung 
perencanaan, monitoring, dan evaluasi proyek konstruksi secara presisi tinggi.

Sistem ini mendukung:

- Surface-to-Surface volumetric computation  
- Automatic cross-section generation  
- Centered station profiling  
- Masking berbasis AOI  
- Reprojection & resampling (WarpedVRT)  
        """)

    if st.button("🚀 Mulai Analisis"):
        st.session_state.step = 1
        go_to("select")


# =========================================================
# SELECT (WIZARD MODE)
# =========================================================
elif page == "select":

    # pastikan step ada
    if "step" not in st.session_state:
        st.session_state.step = 1

    current_step = st.session_state.step

    # ======================================================
    # STEP 1 — PILIH METODE
    # ======================================================
    if current_step == 1:

        st.markdown(
            "<h3 style='text-align:center;margin-bottom:40px;'>Pilih Metode Perhitungan</h3>",
            unsafe_allow_html=True
        )

        methods = [
            ("Base Elevasi", "Baseelevation.png"),
            ("Lowest Point", "Lowestpoint.png"),
            ("Surface to Surface", "surfacetosurface.png")
        ]

        descriptions = {
            "Base Elevasi": "Volume terhadap elevasi referensi manual.",
            "Lowest Point": "Volume terhadap elevasi DEM terendah.",
            "Surface to Surface": "Perbandingan dua permukaan DEM."
        }

        left, center, right = st.columns([1, 8, 1])

        with center:
            cols = st.columns(3, gap="large")

            for col, (label, img) in zip(cols, methods):
                with col:

                    if st.button(label, key=f"btn_{label}", use_container_width=True):
                        st.session_state.metode = label
                        st.session_state.step = 2
                        st.rerun()

                    st.markdown(
                        f"<p style='font-size:14px;color:#cbd5e1;margin:15px 0 20px 0;text-align:center;'>"
                        f"{descriptions[label]}</p>",
                        unsafe_allow_html=True
                    )

                    if os.path.exists(img):
                        st.image(img, use_container_width=True)

    # ======================================================
    # STEP 2 — UPLOAD DATA
    # ======================================================
    elif current_step == 2:

        st.markdown(f"### Upload & Konfigurasi — {st.session_state.metode}")

        f_aoi = st.file_uploader("AOI (ZIP/GeoJSON)", type=["zip","geojson","json"])
        f_cl = st.file_uploader("Centerline (SHP/ZIP/GeoJSON)", type=["geojson","json", "shp", "zip"])
        f_top = st.file_uploader("DEM Top (.tif)", type=["tif"])

        f_base = None
        manual = 0

        if st.session_state.metode == "Surface to Surface":
            f_base = st.file_uploader("DEM Base (.tif)", type=["tif"])

        elif st.session_state.metode == "Base Elevasi":
            manual = st.number_input("Elevasi Dasar (m)", value=0.0)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("⬅ Kembali"):
                st.session_state.step = 1
                st.rerun()

        with col2:
            if st.button("🚀 Proses", use_container_width=True):

                if not f_top or not f_aoi:
                    st.error("AOI dan DEM Top wajib diunggah.")
                else:

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # STEP 1 — Saving Files
                    status_text.text("📂 Menyimpan file input...")
                    progress_bar.progress(10)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as t1:
                        t1.write(f_top.getvalue())
                        p_top = t1.name

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".zip" if "zip" in f_aoi.name else ".json"
                    ) as t2:
                        t2.write(f_aoi.getvalue())
                        p_aoi = t2.name

                    progress_bar.progress(25)

                    # STEP 2 — Base Surface
                    p_base = None
                    if f_base:
                        status_text.text("📊 Memproses DEM Base...")
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as t3:
                            t3.write(f_base.getvalue())
                            p_base = t3.name

                    progress_bar.progress(40)

                    # STEP 3 — Centerline
                    p_cl = None
                    if f_cl:
                        status_text.text("📏 Memproses Centerline...")
                        ext = os.path.splitext(f_cl.name)[1].lower()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as t4:
                            t4.write(f_cl.getvalue())
                            p_cl = t4.name

                    progress_bar.progress(55)

                    # STEP 4 — Volume Calculation
                    status_text.text("🧮 Menghitung volume dan analisis...")
                    progress_bar.progress(70)

                    res = hitung_volume_universal(
                        p_top, p_aoi, p_base, p_cl,
                        st.session_state.metode, manual
                    )

                    progress_bar.progress(90)

                    # STEP 5 — Finalizing
                    status_text.text("📈 Menyusun visualisasi...")
                    progress_bar.progress(100)

                    if res:
                        st.session_state.result = res
                        st.session_state.p_top = p_top
                        st.session_state.p_base = p_base

                    status_text.empty()
                    progress_bar.empty()

                    go_to("result")
# =========================================================
# RESULT PAGE
# =========================================================
elif st.session_state.page == "result":

    r = st.session_state.result
    net = r["fill"] - r["cut"]

    # --- METRIC CARDS ---
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"<div class='metric-card'><h4>AREA</h4><h2>{r['area']:,.0f} m²</h2></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><h4 style='color:#22c55e'>FILL</h4><h2>{r['fill']:,.2f} m³</h2></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><h4 style='color:#ef4444'>CUT</h4><h2>{r['cut']:,.2f} m³</h2></div>", unsafe_allow_html=True)

    # --- NET STATUS ---
    if net > 0:
        st.success(f"Net Timbunan: {net:,.2f} m³")
    elif net < 0:
        st.warning(f"Net Galian: {abs(net):,.2f} m³")
    else:
        st.info("Cut & Fill Seimbang")

    st.divider()

    # --- TABS VISUALIZATION ---
    tab1, tab2 = st.tabs(["Plan View", "Cross Section"])

    # TAB 1: PLAN VIEW
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0a1120")
        ax.set_facecolor("#0a1120")

        v = r["data_plot"]
        valid = v[v > -9000]
        vmin, vmax = np.percentile(valid, [2, 98])

        im = ax.imshow(v, cmap="terrain", extent=r["extent"], vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Elevation (m)", color="white")
        cbar.ax.tick_params(colors="white")

        if "cl_geom" in r:
            r["cl_geom"].plot(ax=ax, color="#ef4444", lw=3)
        if "cross_gdf" in r:
            r["cross_gdf"].plot(ax=ax, color="white", lw=0.7, ls="--")

        ax.tick_params(colors="white")
        st.pyplot(fig)
        st.session_state.fig_plan = fig  # Store for PDF export
        # =====================================================
        # LONG PROFILE (PROFESSIONAL MODE)
        # =====================================================
        if "cl_geom" in r:

            st.divider()
            st.markdown("## Long Profile")

            line = r["cl_geom"].geometry.iloc[0]

            # Vertical exaggeration control
            col_a, col_b, col_c = st.columns([2,1,2])
            with col_b:
                v_exag = st.slider(
                    "Vertical Exaggeration",
                    min_value=1,
                    max_value=10,
                    value=2,
                    key="v_exag_slider"
                )

            # --- TOP PROFILE ---
            with rasterio.open(st.session_state.p_top) as s:
                dist_long, elev_top = get_long_profile_professional(
                    s, line, s.nodata, interval=5
                )

            # --- BASE PROFILE ---
            if st.session_state.metode == "Surface to Surface":
                with rasterio.open(st.session_state.p_base) as s2:
                    _, elev_base = get_long_profile_professional(
                        s2, line, s2.nodata, interval=5
                    )
            else:
                elev_base = np.full_like(
                    elev_top,
                    np.nanmin(r["data_base_plot"])
                )

            # --- SLOPE ---
            slope = np.gradient(elev_top, dist_long)
            slope_percent = slope * 100

            # --- CUMULATIVE VOLUME ---
            pixel_width = abs(
                (r["extent"][1] - r["extent"][0]) /
                r["data_plot"].shape[1]
            )

            diff_long = elev_top - elev_base
            cumulative_volume = np.nancumsum(diff_long * pixel_width * 5)

            # --- PLOT ---
            fig_long, ax_long = plt.subplots(
                figsize=(16,6),
                facecolor="#0a1120"
            )

            ax_long.set_facecolor("#0a1120")

            ax_long.plot(
                dist_long,
                elev_top,
                color="#3b82f6",
                lw=2.5,
                label="Top Surface"
            )

            ax_long.plot(
                dist_long,
                elev_base,
                color="#ef4444",
                lw=2,
                ls="--",
                label="Base Surface"
            )

            ax_long.fill_between(
                dist_long,
                elev_top,
                elev_base,
                where=(elev_top > elev_base),
                color="#3b82f6",
                alpha=0.3
            )

            ax_long.fill_between(
                dist_long,
                elev_top,
                elev_base,
                where=(elev_top < elev_base),
                color="#ef4444",
                alpha=0.3
            )

            # --- GRID ---
            major_ticks = np.arange(0, dist_long.max(), 100)
            minor_ticks = np.arange(0, dist_long.max(), 20)

            ax_long.set_xticks(major_ticks)
            ax_long.set_xticks(minor_ticks, minor=True)

            ax_long.grid(which="major", color="white", alpha=0.2)
            ax_long.grid(which="minor", color="white", alpha=0.05)

            ax_long.set_xticklabels(
                [f"STA {int(d/1000)}+{int(d%1000):03d}"
                 for d in major_ticks],
                rotation=45,
                color="white"
            )

            ax_long.set_xlabel("Station", color="white")
            ax_long.set_ylabel("Elevation (m)", color="white")

            ax_long.tick_params(colors="white")

            # Vertical exaggeration
            ax_long.set_aspect(v_exag)

            ax_long.legend(facecolor="#111d35", labelcolor="white")

            st.pyplot(fig_long)

            # --- METRICS ---
            st.divider()

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "Max Elevation",
                    f"{np.nanmax(elev_top):.2f} m"
                )

            with m2:
                st.metric(
                    "Max Slope",
                    f"{np.nanmax(np.abs(slope_percent)):.2f} %"
                )

            with m3:
                st.metric(
                    "Cumulative Volume",
                    f"{cumulative_volume[-1]:,.2f} m³"
                )
    # TAB 2: CROSS SECTION
    with tab2:
        if "cross_gdf" in r:
            # Selector STA
            left, center, right = st.columns([3, 1, 3])
            with center:
                sta = st.selectbox("Pilih STA:", r["cross_gdf"]["sta"].tolist(), key="sta_select")

            # Ambil Geometri & Data Profile
            gm = r["cross_gdf"][r["cross_gdf"]["sta"] == sta].geometry.iloc[0]
            
            with rasterio.open(st.session_state.p_top) as s:
                dist, vt = get_profile_data_centered(s, gm, s.nodata)

            if st.session_state.metode == "Surface to Surface":
                with rasterio.open(st.session_state.p_base) as s2:
                    _, vb = get_profile_data_centered(s2, gm, s2.nodata)
            else:
                vb = np.full_like(vt, np.nanmin(r["data_base_plot"]))

            # --- PLOTTING MAIN & INSET ---
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            
            fig2, ax2 = plt.subplots(figsize=(14, 6), facecolor="#0a1120")
            ax2.set_facecolor("#0a1120")

            # Main Cross Section Plot
            ax2.plot(dist, vt, color="#3b82f6", lw=3, label="Top")
            ax2.plot(dist, vb, color="#ef4444", lw=2, ls="--", label="Base")
            ax2.fill_between(dist, vt, vb, where=(vt > vb), color="#3b82f6", alpha=0.3)
            ax2.fill_between(dist, vt, vb, where=(vt < vb), color="#ef4444", alpha=0.3)
            ax2.axvline(0, color="white", lw=2)

            # X-Axis Formatting
            ticks = np.arange(np.floor(dist.min()/5)*5, np.ceil(dist.max()/5)*5+5, 5)
            ax2.set_xticks(ticks)
            ax2.set_xticklabels([f"+{int(t)}" if t > 0 else f"{int(t)}" for t in ticks], color="white")
            ax2.tick_params(colors="white")
            ax2.legend(facecolor="#111d35", labelcolor="white")

            # Inset Map (Lokasi Section)
            axins = inset_axes(ax2, width="25%", height="25%", loc="upper left", borderpad=3)
            axins.set_facecolor("#0a1120")
            axins.imshow(v, cmap="terrain", extent=r["extent"], vmin=vmin, vmax=vmax)
            
            if "cl_geom" in r:
                r["cl_geom"].plot(ax=axins, color="white", lw=1)
            
            x_line, y_line = gm.xy
            axins.plot(x_line, y_line, color="red", lw=2) # Highlight garis section aktif
            axins.set_xticks([]); axins.set_yticks([])
            axins.set_title("Lokasi Section", color="white", fontsize=9)

            st.pyplot(fig2)
            st.session_state.fig_cross = fig2  # Store for PDF export  
        else:
            st.info("Upload centerline untuk melihat cross section.")
# =====================================================
# EXPORT PDF PROFESSIONAL REPORT
# =====================================================
st.divider()

r = st.session_state.get("result", None)

if r:

    net = r["fill"] - r["cut"]

    fig_plan = st.session_state.get("fig_plan", None)
    fig_cross = st.session_state.get("fig_cross", None)

    fig_plan_path = None
    fig_cross_path = None

    # --- Save Plan View ---
    if fig_plan:
        fig_plan_path = "plan_view.png"
        fig_plan.savefig(fig_plan_path, dpi=300, bbox_inches="tight")

    # --- Save Cross Section ---
    if fig_cross:
        fig_cross_path = "cross_section.png"
        fig_cross.savefig(fig_cross_path, dpi=300, bbox_inches="tight")

    # --- Create PDF ---
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()

    # --- LOGO ---
    logo_path = "logowaskita.png"
    if os.path.exists(logo_path):
        pdf.image(logo_path, x=10, y=8, w=35)

    pdf.ln(20)

    # --- TITLE ---
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "W-GVD Engineering Volume Report", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.ln(5)

    pdf.cell(0, 8, f"Area: {r['area']:,.2f} m2", ln=True)
    pdf.cell(0, 8, f"Fill: {r['fill']:,.2f} m3", ln=True)
    pdf.cell(0, 8, f"Cut: {r['cut']:,.2f} m3", ln=True)
    pdf.cell(0, 8, f"Net Volume: {net:,.2f} m3", ln=True)

    pdf.ln(10)

    # --- Insert Plan View ---
    if fig_plan_path:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, "Plan View", ln=True)
        pdf.image(fig_plan_path, w=180)

    # --- Insert Cross Section ---
    if fig_cross_path:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, "Cross Section", ln=True)
        pdf.image(fig_cross_path, w=180)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")

    st.download_button(
        label="📄 Download Engineering PDF Report",
        data=pdf_bytes,
        file_name="W-GVD_Engineering_Report.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    # --- Cleanup ---
    if fig_plan_path and os.path.exists(fig_plan_path):
        os.remove(fig_plan_path)

    if fig_cross_path and os.path.exists(fig_cross_path):
        os.remove(fig_cross_path)


# =====================================================
# FOOTER ACTIONS
# =====================================================
st.divider()

if st.button("Analisis Baru", use_container_width=True):
    st.session_state.clear()
    st.rerun()

st.markdown(f"<div class='footer'>© {datetime.now().year} PT Waskita Karya</div>", unsafe_allow_html=True)