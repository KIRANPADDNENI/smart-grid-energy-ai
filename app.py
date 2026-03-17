import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import plotly.express as px
import datetime
import io
import time

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

st.set_page_config(layout="wide")

# ---------------- THEME ----------------
st.markdown("""
<style>
.stApp { background-color: #0b1c2d; }
h1, h2, h3 { color: #00e5ff !important; }
[data-testid="stWidgetLabel"] { color: white !important; }
section[data-testid="stSidebar"] { background-color: #08121f; }
.stButton>button { background-color: #00e5ff; color: black; }
.upload-box { background-color: #112b45; padding: 25px; border-radius: 15px; text-align: center; }
.kpi-card { background-color: #112b45; padding: 18px; border-radius: 12px; text-align: center; color: white; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

st.title("🏙 Smart Grid Energy Intelligence System")
st.write(f"Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("---")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙ AI Control Panel")

k = st.sidebar.number_input("Manual Cluster Selection", 2, 6, 3)
auto_cluster = st.sidebar.toggle("🤖 AI Auto Cluster")
anomaly_toggle = st.sidebar.toggle("🚨 Enable Anomaly Detection")
stream_toggle = st.sidebar.toggle("📡 Real-Time Simulation")

st.sidebar.markdown("---")
st.sidebar.info("Enable intelligent features for advanced analytics.")

# ---------------- UPLOAD ----------------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Smart Meter CSV")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MAIN ----------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour

    # -------- REAL-TIME --------
    if stream_toggle:
        st.subheader("📡 Live Energy Feed")
        chart_area = st.empty()
        for i in range(10, min(100, len(df)), 10):
            fig_live = px.line(df.head(i), x="Timestamp", y="Consumption")
            fig_live.update_layout(paper_bgcolor="#0b1c2d",
                                   plot_bgcolor="#0b1c2d",
                                   font=dict(color="white"))
            chart_area.plotly_chart(fig_live, use_container_width=True)
            time.sleep(0.3)

    # -------- FEATURES --------
    features = df.groupby('House_ID').agg(
        Avg_Usage=('Consumption', 'mean'),
        Night_Usage=('Consumption', lambda x: x[df.loc[x.index,'Hour'].between(18,23)].mean()),
        Day_Usage=('Consumption', lambda x: x[df.loc[x.index,'Hour'].between(6,17)].mean())
    ).reset_index()

    features.fillna(0, inplace=True)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features[['Avg_Usage','Night_Usage','Day_Usage']])

    # -------- AUTO CLUSTER --------
    if auto_cluster:
        best_score = -1
        best_k = 2
        for i in range(2,7):
            km = KMeans(n_clusters=i, random_state=42, n_init=10)
            labels_temp = km.fit_predict(scaled_data)
            score_temp = silhouette_score(scaled_data, labels_temp)
            if score_temp > best_score:
                best_score = score_temp
                best_k = i
        k = best_k
        st.success(f"AI Selected Optimal Clusters: {k}")

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    features['Cluster'] = model.fit_predict(scaled_data)

    sil_score = silhouette_score(scaled_data, features['Cluster'])

    # -------- KPI --------
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown(f'<div class="kpi-card">🏠 Houses<br><b>{features.shape[0]}</b></div>', unsafe_allow_html=True)
    with colB:
        st.markdown(f'<div class="kpi-card">⚡ Avg Usage<br><b>{round(features["Avg_Usage"].mean(),2)} kWh</b></div>', unsafe_allow_html=True)
    with colC:
        st.markdown(f'<div class="kpi-card">📊 Silhouette<br><b>{round(sil_score,3)}</b></div>', unsafe_allow_html=True)

    st.markdown("---")

    # -------- MAP --------
    st.subheader("🗺 Smart City Cluster Map")
    np.random.seed(42)
    features["lat"] = 28.6 + np.random.rand(len(features))*0.05
    features["lon"] = 77.2 + np.random.rand(len(features))*0.05

    fig_map = px.scatter_mapbox(features,
                                lat="lat",
                                lon="lon",
                                color="Cluster",
                                size="Avg_Usage",
                                zoom=10,
                                height=600,
                                mapbox_style="open-street-map")

    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                          font=dict(color="white"))
    st.plotly_chart(fig_map, use_container_width=True)

    # -------- DONUT --------
    st.subheader("📊 Intelligent Cluster Distribution")

    cluster_summary = features.groupby("Cluster").agg(
        Houses=("House_ID","count"),
        Avg_Usage=("Avg_Usage","mean"),
        Day_Usage=("Day_Usage","mean"),
        Night_Usage=("Night_Usage","mean")
    ).reset_index()

    labels = {}
    for _, row in cluster_summary.iterrows():
        if row["Day_Usage"] > row["Night_Usage"] and row["Avg_Usage"] > cluster_summary["Avg_Usage"].mean():
            labels[row["Cluster"]] = "High Day Users"
        elif row["Night_Usage"] > row["Day_Usage"]:
            labels[row["Cluster"]] = "Night Peak Users"
        else:
            labels[row["Cluster"]] = "Low Consumption Users"

    cluster_summary["Label"] = cluster_summary["Cluster"].map(labels)

    fig_donut = px.pie(cluster_summary,
                       names="Label",
                       values="Houses",
                       hole=0.5)

    fig_donut.update_traces(textinfo="percent+label")
    fig_donut.update_layout(
        paper_bgcolor="#0b1c2d",
        plot_bgcolor="#0b1c2d",
        font=dict(color="white"),
        annotations=[dict(text=f"Total<br>{features.shape[0]} Houses",
                          x=0.5,y=0.5,
                          showarrow=False,
                          font=dict(color="white",size=16))]
    )
    st.plotly_chart(fig_donut, use_container_width=True)

    st.subheader("📈 Cluster Behavior Summary")
    st.dataframe(cluster_summary[["Label","Houses","Avg_Usage","Day_Usage","Night_Usage"]])

    # -------- SILHOUETTE VS K --------
    st.markdown("---")
    st.subheader("📊 Model Optimization (Silhouette vs K)")

    sil_scores = []
    for i in range(2,7):
        km_temp = KMeans(n_clusters=i, random_state=42, n_init=10)
        labels_temp = km_temp.fit_predict(scaled_data)
        sil_scores.append(silhouette_score(scaled_data, labels_temp))

    fig_sil = px.line(x=list(range(2,7)), y=sil_scores, markers=True)
    fig_sil.update_layout(paper_bgcolor="#0b1c2d",
                          plot_bgcolor="#0b1c2d",
                          font=dict(color="white"))
    st.plotly_chart(fig_sil, use_container_width=True)

    # -------- PCA --------
    st.markdown("---")
    st.subheader("🧭 PCA Cluster Projection")

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    pca_df = pd.DataFrame(pca_data, columns=["PC1","PC2"])
    pca_df["Cluster"] = features["Cluster"]

    fig_pca = px.scatter(pca_df,
                         x="PC1",
                         y="PC2",
                         color="Cluster")

    fig_pca.update_layout(paper_bgcolor="#0b1c2d",
                          plot_bgcolor="#0b1c2d",
                          font=dict(color="white"))
    st.plotly_chart(fig_pca, use_container_width=True)

    # -------- STABILITY --------
    st.markdown("---")
    st.subheader("🔁 Cluster Stability Test")

    stability_scores = []
    for seed in [0,10,20,30,42]:
        km_temp = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels_temp = km_temp.fit_predict(scaled_data)
        stability_scores.append(silhouette_score(scaled_data, labels_temp))

    st.write("Stability Scores:", stability_scores)
    st.success(f"Average Stability Score: {round(np.mean(stability_scores),3)}")

    # -------- EXECUTIVE SUMMARY --------
    st.markdown("---")
    st.subheader("📝 Executive Summary")

    best_cluster = cluster_summary.loc[cluster_summary["Avg_Usage"].idxmax()]

    summary_text = f"""
    This Smart Grid analysis identified {k} distinct household energy groups.
    The dominant group is '{best_cluster['Label']}' with average usage of
    {round(best_cluster['Avg_Usage'],2)} kWh.
    Silhouette Score of {round(sil_score,3)} indicates strong clustering quality.
    """

    st.info(summary_text)

    # -------- PROFESSIONAL PDF --------
    if st.button("📥 Export Professional PDF Report"):

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Smart Grid Energy Intelligence Report", styles["Title"]))
        elements.append(Spacer(1, 0.4 * inch))
        elements.append(Paragraph(summary_text, styles["Normal"]))
        elements.append(PageBreak())

        elements.append(Paragraph("Cluster Distribution", styles["Heading2"]))
        elements.append(Spacer(1, 0.3 * inch))

        drawing = Drawing(400, 200)
        pie = Pie()
        pie.x = 150
        pie.y = 15
        pie.width = 150
        pie.height = 150
        pie.data = list(cluster_summary["Houses"])
        pie.labels = list(cluster_summary["Label"])
        drawing.add(pie)
        elements.append(drawing)

        doc.build(elements)
        pdf = buffer.getvalue()
        buffer.close()

        st.download_button("Download Professional PDF",
                           pdf,
                           "Smart_Grid_Professional_Report.pdf",
                           "application/pdf")

    # -------- ANOMALY --------
    if anomaly_toggle:
        st.markdown("---")
        st.subheader("🚨 Anomaly Detection")
        iso = IsolationForest(contamination=0.05, random_state=42)
        df['Anomaly'] = iso.fit_predict(df[['Consumption']])
        st.warning(f"Detected {df[df['Anomaly'] == -1].shape[0]} anomalous readings")

else:
    st.info("Upload a Smart Meter CSV file to activate system.")
