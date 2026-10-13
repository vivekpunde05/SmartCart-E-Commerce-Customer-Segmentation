import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pipeline
from typing import Dict, Any

# Page Configuration
st.set_page_config(
    page_title="SmartCart AI | Customer Segmentation & Behavioral Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark luxury analytics theme)
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Top Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        margin-bottom: 0px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.2;
    }
    .metric-subtext {
        font-size: 0.76rem;
        color: #38bdf8;
        margin-top: 4px;
    }
    
    /* Persona Card */
    .persona-card {
        background: rgba(15, 23, 42, 0.85);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-bottom: 10px;
    }
    
    /* Streamlit overrides */
    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.92rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.2) !important;
        border-bottom: 2px solid #818cf8 !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Data & Pipeline Caching
@st.cache_data(show_spinner="Loading and preprocessing customer dataset...")
def get_base_data():
    df_full, df_encoded, feature_cols, medians = pipeline.load_and_preprocess_data("smartcart_customers.csv")
    X_scaled, X_pca, scaler, pca = pipeline.compute_pca_pipeline(df_encoded)
    return df_full, df_encoded, feature_cols, medians, X_scaled, X_pca, scaler, pca


@st.cache_data(show_spinner="Computing clustering diagnostic metrics...")
def get_diagnostics(X_pca):
    return pipeline.compute_clustering_diagnostics(X_pca, max_k=8)


# Load dataset and PCA model
df_full_raw, df_encoded, feature_cols, medians, X_scaled, X_pca, scaler, pca = get_base_data()


# Sidebar Controls & Global Filters
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shopping-cart-loaded.png", width=64)
    st.markdown("### **SmartCart Intelligence**")
    st.caption("E-Commerce Customer Behavioral Segmentation Engine")
    st.markdown("---")
    
    st.subheader("⚙️ Clustering Configuration")
    algorithm = st.selectbox(
        "Clustering Algorithm",
        options=["Agglomerative", "KMeans"],
        index=0,
        help="Agglomerative Hierarchical Clustering (Ward linkage) was selected in the project notebook. You can also experiment with K-Means."
    )
    
    n_clusters = st.slider(
        "Number of Clusters (K)",
        min_value=2,
        max_value=8,
        value=4,
        help="Notebook discovered K=4 as the optimal elbow and silhouette segmentation point."
    )
    
    st.markdown("---")
    st.subheader("🔍 Exploration Filters")
    
    income_min = int(df_full_raw["Income"].min())
    income_max = int(df_full_raw["Income"].max())
    selected_income = st.slider(
        "Household Income ($)",
        min_value=income_min,
        max_value=income_max,
        value=(income_min, income_max),
        step=2000
    )
    
    spend_min = int(df_full_raw["Total_Spending"].min())
    spend_max = int(df_full_raw["Total_Spending"].max())
    selected_spend = st.slider(
        "Total Spending ($)",
        min_value=spend_min,
        max_value=spend_max,
        value=(spend_min, spend_max),
        step=50
    )
    
    selected_education = st.multiselect(
        "Education Tier",
        options=sorted(df_full_raw["Education"].unique()),
        default=sorted(df_full_raw["Education"].unique())
    )
    
    selected_living = st.multiselect(
        "Living Status",
        options=sorted(df_full_raw["Living_With"].unique()),
        default=sorted(df_full_raw["Living_With"].unique())
    )
    
    st.markdown("---")
    st.caption(f"📊 Dataset Size: **{len(df_full_raw):,}** active customer records")
    st.caption(f"🧠 PCA Explained Variance: **{np.sum(pca.explained_variance_ratio_)*100:.1f}%**")


# Model Fitting & Persona Association
@st.cache_resource(show_spinner=False)
def get_trained_cluster_model(X_pca_arr, algo, k):
    return pipeline.fit_cluster_model(X_pca_arr, algorithm=algo, n_clusters=k)

labels, knn_model = get_trained_cluster_model(X_pca, algorithm, n_clusters)

# Attach cluster to dataframe
df_full = df_full_raw.copy()
df_full["Cluster"] = labels
df_full["PC1"] = X_pca[:, 0]
df_full["PC2"] = X_pca[:, 1]
df_full["PC3"] = X_pca[:, 2]

# Personas mapping
personas = pipeline.generate_personas_for_clusters(df_full, cluster_col="Cluster")
df_full["Persona_Name"] = df_full["Cluster"].map(lambda c: personas.get(c, {}).get("name", f"Cluster {c}"))
df_full["Persona_Color"] = df_full["Cluster"].map(lambda c: personas.get(c, {}).get("color", "#6366F1"))

# Filtered view for interactive explorations
filter_mask = (
    (df_full["Income"] >= selected_income[0]) & (df_full["Income"] <= selected_income[1]) &
    (df_full["Total_Spending"] >= selected_spend[0]) & (df_full["Total_Spending"] <= selected_spend[1]) &
    (df_full["Education"].isin(selected_education)) &
    (df_full["Living_With"].isin(selected_living))
)
df_filtered = df_full[filter_mask].copy()


# Main Header Banner
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🛒 SmartCart Customer Segmentation & Behavioral Intelligence</div>
        <div class="hero-subtitle">
            Unsupervised Machine Learning platform converting multidimensional retail transactional data 
            into actionable customer personas, targeted campaign playbooks, and real-time inference.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Top KPI Metric Cards Row
col1, col2, col3, col4, col5 = st.columns(5)
total_cust = len(df_filtered)
total_rev = df_filtered["Total_Spending"].sum()
avg_inc = df_filtered["Income"].mean()
avg_spend = df_filtered["Total_Spending"].mean()
campaign_resp_rate = df_filtered["Response"].mean() * 100

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Active Customers</div>
            <div class="metric-value">{total_cust:,}</div>
            <div class="metric-subtext">{(total_cust/len(df_full))*100:.1f}% of cohort</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Cumulative Spend</div>
            <div class="metric-value">${total_rev/1e6:.2f}M</div>
            <div class="metric-subtext">Total Gross Volume</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Avg Household Income</div>
            <div class="metric-value">${avg_inc:,.0f}</div>
            <div class="metric-subtext">Cohort Baseline</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Avg Customer Spend</div>
            <div class="metric-value">${avg_spend:,.0f}</div>
            <div class="metric-subtext">Customer Lifetime Value</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Campaign Response</div>
            <div class="metric-value">{campaign_resp_rate:.1f}%</div>
            <div class="metric-subtext">Latest Campaign Conversion</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# Tabs Navigation
tab_overview, tab_clustering, tab_personas, tab_campaigns, tab_classifier, tab_data = st.tabs([
    "📊 Executive EDA",
    "🔬 PCA & Cluster Space",
    "👥 Customer Personas",
    "📈 Campaign Performance & ROI",
    "🔮 Live Customer Classifier",
    "📁 Data Records & Batch Upload"
])


# ==========================================
# TAB 1: EXECUTIVE EDA
# ==========================================
with tab_overview:
    st.subheader("Customer Cohort & Purchasing Distributions")
    
    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        fig_inc = px.histogram(
            df_filtered,
            x="Income",
            nbins=40,
            title="Household Income Distribution ($)",
            color_discrete_sequence=["#38BDF8"],
            marginal="box"
        )
        fig_inc.update_layout(template="plotly_dark", height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_inc, use_container_width=True)
        
    with col_dist2:
        fig_spend = px.histogram(
            df_filtered,
            x="Total_Spending",
            nbins=40,
            title="Total Discretionary Spending ($)",
            color_discrete_sequence=["#818CF8"],
            marginal="box"
        )
        fig_spend.update_layout(template="plotly_dark", height=340, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_spend, use_container_width=True)
        
    st.write("")
    col_cat, col_chan = st.columns([1.2, 1])
    
    with col_cat:
        st.markdown("#### 🍷 Product Category Revenue Breakdown")
        product_cols = {
            "MntWines": "Wines",
            "MntMeatProducts": "Meat Products",
            "MntGoldProds": "Gold Products",
            "MntFishProducts": "Fish Products",
            "MntSweetProducts": "Sweets & Confectionery",
            "MntFruits": "Fruits"
        }
        product_sums = {label: df_filtered[col].sum() for col, label in product_cols.items()}
        df_prod_rev = pd.DataFrame(list(product_sums.items()), columns=["Category", "Total_Revenue"]).sort_values("Total_Revenue", ascending=False)
        
        fig_prod = px.bar(
            df_prod_rev,
            x="Category",
            y="Total_Revenue",
            color="Category",
            text_auto="$,.0f",
            color_discrete_sequence=px.colors.qualitative.Prism,
            title="Total Spend per Product Line ($)"
        )
        fig_prod.update_layout(template="plotly_dark", height=360, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_prod, use_container_width=True)
        
    with col_chan:
        st.markdown("#### 🛍️ Purchasing Channels Utilization")
        channel_cols = {
            "NumStorePurchases": "Physical Store",
            "NumWebPurchases": "Web E-Commerce",
            "NumCatalogPurchases": "Catalog Orders",
            "NumDealsPurchases": "Deal Promotions"
        }
        channel_sums = {label: df_filtered[col].sum() for col, label in channel_cols.items()}
        df_channels = pd.DataFrame(list(channel_sums.items()), columns=["Channel", "Total_Orders"])
        
        fig_chan = px.pie(
            df_channels,
            names="Channel",
            values="Total_Orders",
            hole=0.45,
            color_discrete_sequence=["#10B981", "#38BDF8", "#F59E0B", "#EC4899"],
            title="Share of Purchase Volume by Channel"
        )
        fig_chan.update_layout(template="plotly_dark", height=360, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_chan, use_container_width=True)
        
    st.write("")
    with st.expander("🔥 Correlation Heatmap Analysis (Notebook Verification)"):
        corr_cols = ["Income", "Total_Spending", "Age", "Total_Children", "Recency", "Customer_Tenure_Days", "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Response"]
        corr_matrix = df_filtered[corr_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="Feature Correlation Matrix (Cleaned Cohort)"
        )
        fig_corr.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_corr, use_container_width=True)


# ==========================================
# TAB 2: PCA & CLUSTER SPACE
# ==========================================
with tab_clustering:
    st.subheader("Dimensionality Reduction & Latent Space Analysis")
    
    # Diagnostics row: Elbow Curve & Silhouette Score
    diag = get_diagnostics(X_pca)
    
    col_d1, col_d2, col_d3 = st.columns([1, 1, 0.9])
    with col_d1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=diag["k_range_wcss"],
            y=diag["wcss"],
            mode="lines+markers",
            marker=dict(size=8, color="#38BDF8"),
            line=dict(color="#38BDF8", width=2),
            name="WCSS (Inertia)"
        ))
        fig_elbow.add_vline(x=diag["optimal_k"], line_dash="dash", line_color="#F43F5E", annotation_text=f"Elbow K={diag['optimal_k']}")
        fig_elbow.update_layout(
            template="plotly_dark",
            title="Elbow Method (WCSS vs K)",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Within-Cluster SSE",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        
    with col_d2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=diag["k_range_sil"],
            y=diag["silhouette_scores"],
            mode="lines+markers",
            marker=dict(size=8, color="#F59E0B"),
            line=dict(color="#F59E0B", width=2),
            name="Silhouette Score"
        ))
        fig_sil.update_layout(
            template="plotly_dark",
            title="Silhouette Analysis across K",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Silhouette Coefficient",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_sil, use_container_width=True)
        
    with col_d3:
        var_ratios = pca.explained_variance_ratio_
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.6); padding: 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); height: 300px; display: flex; flex-direction: column; justify-content: space-around;">
                <div style="font-weight: 700; color: #f8fafc; font-size: 1rem;">PCA Latent Components</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">
                    <div>• <b>PC1 Variance:</b> <span style="color:#38bdf8">{var_ratios[0]*100:.1f}%</span> (Spending & Purchasing power)</div>
                    <div style="margin-top: 6px;">• <b>PC2 Variance:</b> <span style="color:#818cf8">{var_ratios[1]*100:.1f}%</span> (Household & Family dynamics)</div>
                    <div style="margin-top: 6px;">• <b>PC3 Variance:</b> <span style="color:#c084fc">{var_ratios[2]*100:.1f}%</span> (Deal seeking & Web visits)</div>
                    <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1);">
                        <b>Total Explained Variance:</b> <span style="color:#10b981; font-weight:700;">{np.sum(var_ratios)*100:.1f}%</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.write("")
    st.markdown("#### 🌐 Interactive 3D Latent PCA Customer Space")
    st.caption("Rotate, zoom, and hover over individual customer points. Markers are sized proportionally to customer spending.")
    
    # 3D PCA Scatter Plot
    fig_3d = px.scatter_3d(
        df_filtered,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Persona_Name",
        size="Total_Spending",
        size_max=18,
        opacity=0.85,
        hover_data={
            "ID": True,
            "Income": ":$,.0f",
            "Total_Spending": ":$,.0f",
            "Age": True,
            "Education": True,
            "Living_With": True,
            "Total_Children": True,
            "PC1": False,
            "PC2": False,
            "PC3": False
        },
        color_discrete_map={p["name"]: p["color"] for p in personas.values()}
    )
    fig_3d.update_layout(
        template="plotly_dark",
        height=620,
        scene=dict(
            xaxis_title="PC1 (Affluence & Spend)",
            yaxis_title="PC2 (Family & Household)",
            zaxis_title="PC3 (Deals & Channel)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.write("")
    st.markdown("#### 🎯 Bivariate Business Feature Projection")
    st.caption("Compare how clusters separate across any two business metrics.")
    
    biv_cols = ["Income", "Total_Spending", "Age", "Total_Children", "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", "Recency"]
    col_bx, col_by = st.columns(2)
    with col_bx:
        axis_x = st.selectbox("Select X-Axis", options=biv_cols, index=1)
    with col_by:
        axis_y = st.selectbox("Select Y-Axis", options=biv_cols, index=0)
        
    fig_biv = px.scatter(
        df_filtered,
        x=axis_x,
        y=axis_y,
        color="Persona_Name",
        hover_data=["ID", "Education", "Living_With", "Total_Children"],
        color_discrete_map={p["name"]: p["color"] for p in personas.values()},
        title=f"{axis_y} vs {axis_x} by Customer Segment"
    )
    fig_biv.update_layout(template="plotly_dark", height=480, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_biv, use_container_width=True)


# ==========================================
# TAB 3: CUSTOMER PERSONAS & PROFILES
# ==========================================
with tab_personas:
    st.subheader("Customer Personas & Deep-Dive Segment Characterization")
    
    # Display Persona Cards in a Grid
    num_personas = len(personas)
    card_cols = st.columns(min(num_personas, 4))
    for i, (cid, pdata) in enumerate(personas.items()):
        col_target = card_cols[i % len(card_cols)]
        cluster_subset = df_full[df_full["Cluster"] == cid]
        with col_target:
            st.markdown(
                f"""
                <div class="persona-card" style="border-left: 5px solid {pdata['color']};">
                    <span class="badge-pill" style="background-color: {pdata['color']}22; color: {pdata['color']}; border: 1px solid {pdata['color']};">
                        {pdata['badge']} • {len(cluster_subset)} Users ({(len(cluster_subset)/len(df_full))*100:.1f}%)
                    </span>
                    <h4 style="margin: 0 0 8px 0; color: #f8fafc;">{pdata['name']}</h4>
                    <p style="font-size: 0.83rem; color: #94a3b8; min-height: 48px;">{pdata['summary']}</p>
                    <div style="font-size: 0.82rem; color: #e2e8f0; margin-top: 10px;">
                        <b>Key Characteristics:</b>
                        <ul style="padding-left: 18px; margin-top: 4px; margin-bottom: 8px;">
                            {''.join([f'<li style="margin-bottom: 2px;">{t}</li>' for t in pdata['traits'][:4]])}
                        </ul>
                    </div>
                    <div style="font-size: 0.8rem; margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.08);">
                        <span style="color: #94a3b8;">Primary Channel:</span> <span style="color: {pdata['color']}; font-weight:600;">{pdata['primary_channels']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    st.write("")
    
    # Radar Chart Comparison across normalized attributes
    st.markdown("#### 🕸️ Multidimensional Segment Radar Chart")
    st.caption("Attributes are normalized (0 to 1 scale) relative to cohort maximums to highlight comparative strengths and affinities.")
    
    radar_metrics = [
        ("Income", "Income"),
        ("Total_Spending", "Spending"),
        ("Total_Children", "Children"),
        ("NumDealsPurchases", "Deal Sensitivity"),
        ("NumWebPurchases", "Web Shopping"),
        ("NumStorePurchases", "Store Shopping"),
        ("NumCatalogPurchases", "Catalog Shopping"),
        ("Response", "Campaign Response")
    ]
    
    radar_cols = [m[0] for m in radar_metrics]
    radar_labels = [m[1] for m in radar_metrics]
    
    cluster_means = df_full.groupby("Cluster")[radar_cols].mean()
    max_vals = df_full[radar_cols].max()
    normalized_means = cluster_means / max_vals
    
    fig_radar = go.Figure()
    for cid, pdata in personas.items():
        if cid in normalized_means.index:
            vals = normalized_means.loc[cid].tolist()
            vals += [vals[0]]  # close the loop
            fig_radar.add_trace(go.Scatterpolar(
                r=vals,
                theta=radar_labels + [radar_labels[0]],
                fill='toself',
                name=pdata["name"],
                line_color=pdata["color"],
                opacity=0.65
            ))
            
    fig_radar.update_layout(
        template="plotly_dark",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)
        ),
        height=480,
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.write("")
    
    # Product Category Spending by Segment (Stacked Bar)
    col_p1, col_p2 = st.columns([1.2, 1])
    with col_p1:
        st.markdown("#### 🍷 Product Mix Spending per Customer ($)")
        cat_spend_df = df_full.groupby("Persona_Name")[list(product_cols.keys())].mean().reset_index()
        cat_spend_melted = cat_spend_df.melt(id_vars="Persona_Name", var_name="Product_Code", value_name="Avg_Spend")
        cat_spend_melted["Product"] = cat_spend_melted["Product_Code"].map(product_cols)
        
        fig_cat_stack = px.bar(
            cat_spend_melted,
            x="Persona_Name",
            y="Avg_Spend",
            color="Product",
            title="Average Spend per Customer by Product Line",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_cat_stack.update_layout(template="plotly_dark", height=400, barmode="stack", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cat_stack, use_container_width=True)
        
    with col_p2:
        st.markdown("#### 📦 Channel Orders per Customer")
        chan_spend_df = df_full.groupby("Persona_Name")[list(channel_cols.keys())].mean().reset_index()
        chan_melted = chan_spend_df.melt(id_vars="Persona_Name", var_name="Channel_Code", value_name="Avg_Orders")
        chan_melted["Channel"] = chan_melted["Channel_Code"].map(channel_cols)
        
        fig_chan_bar = px.bar(
            chan_melted,
            x="Persona_Name",
            y="Avg_Orders",
            color="Channel",
            barmode="group",
            title="Average Purchases by Channel",
            color_discrete_sequence=["#10B981", "#38BDF8", "#F59E0B", "#EC4899"]
        )
        fig_chan_bar.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_chan_bar, use_container_width=True)
        
    st.write("")
    st.markdown("#### 📋 Detailed Segment Benchmark Statistics")
    summary_table = df_full.groupby("Persona_Name").agg(
        Customers=("ID", "count"),
        Avg_Income=("Income", "mean"),
        Avg_Spending=("Total_Spending", "mean"),
        Avg_Age=("Age", "mean"),
        Avg_Children=("Total_Children", "mean"),
        Web_Purchases=("NumWebPurchases", "mean"),
        Store_Purchases=("NumStorePurchases", "mean"),
        Catalog_Purchases=("NumCatalogPurchases", "mean"),
        Deal_Purchases=("NumDealsPurchases", "mean"),
        Response_Rate=("Response", lambda x: f"{x.mean()*100:.1f}%")
    ).reset_index()
    
    st.dataframe(
        summary_table.style.format({
            "Avg_Income": "${:,.0f}",
            "Avg_Spending": "${:,.0f}",
            "Avg_Age": "{:.1f}",
            "Avg_Children": "{:.2f}",
            "Web_Purchases": "{:.2f}",
            "Store_Purchases": "{:.2f}",
            "Catalog_Purchases": "{:.2f}",
            "Deal_Purchases": "{:.2f}"
        }),
        use_container_width=True
    )


# ==========================================
# TAB 4: CAMPAIGN PERFORMANCE & ROI
# ==========================================
with tab_campaigns:
    st.subheader("Marketing Campaign Historical Performance & Strategic Playbook")
    
    cmp_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response"]
    cmp_labels = ["Campaign 1", "Campaign 2", "Campaign 3", "Campaign 4", "Campaign 5", "Final Campaign"]
    
    cmp_summary = df_full.groupby("Persona_Name")[cmp_cols].mean() * 100
    cmp_summary.columns = cmp_labels
    
    col_c1, col_c2 = st.columns([1.2, 1])
    with col_c1:
        fig_cmp = px.imshow(
            cmp_summary,
            text_auto=".1f",
            color_continuous_scale="Viridis",
            labels=dict(x="Campaign", y="Segment", color="Conversion Rate (%)"),
            title="Conversion Rate (%) by Campaign & Customer Segment"
        )
        fig_cmp.update_layout(template="plotly_dark", height=380, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cmp, use_container_width=True)
        
    with col_c2:
        st.markdown("#### 🎯 Key Campaign Takeaways")
        st.markdown(
            """
            * **Elite Affluent Singles** generated the highest campaign responsiveness (~32.2% on final campaign and top conversions in Cmp 1 & 5).
            * **Affluent Power Couples** responded enthusiastically to Campaign 5 and the final campaign, demonstrating strong appetite for curated gourmet & wine events.
            * **Value-Conscious Singles** showed a notable spike in Campaign 3 and 4, responding well to discount-driven promotions and free shipping perks.
            * **Budget Family Partners** have the lowest response to generic email blasts (~7%), requiring targeted family bundles and deal pricing to activate.
            """
        )
        
    st.markdown("---")
    st.subheader("💰 Targeted Campaign ROI Simulator")
    st.caption("Simulate expected marketing revenue and net ROI when directing dedicated campaigns to selected personas.")
    
    sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 1.2])
    with sim_col1:
        target_persona_name = st.selectbox(
            "Select Target Customer Segment",
            options=sorted(df_full["Persona_Name"].unique())
        )
        target_subset = df_full[df_full["Persona_Name"] == target_persona_name]
        target_size = len(target_subset)
        st.info(f"Target Audience Size: **{target_size:,}** customers")
        
    with sim_col2:
        contact_cost = st.number_input("Campaign Cost per Contact ($)", min_value=0.5, max_value=50.0, value=3.0, step=0.5)
        default_resp = float(target_subset["Response"].mean() * 100)
        expected_conv_rate = st.slider("Expected Conversion Rate (%)", min_value=1.0, max_value=60.0, value=max(5.0, min(50.0, default_resp)), step=1.0)
        avg_order_val = st.number_input("Average Order Value from Campaign ($)", min_value=20.0, max_value=1500.0, value=float(target_subset["Total_Spending"].mean() * 0.35), step=25.0)
        
    with sim_col3:
        total_campaign_cost = target_size * contact_cost
        expected_buyers = int(target_size * (expected_conv_rate / 100.0))
        projected_gross_rev = expected_buyers * avg_order_val
        net_profit = projected_gross_rev - total_campaign_cost
        roi_pct = (net_profit / total_campaign_cost) * 100 if total_campaign_cost > 0 else 0
        
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.7); padding: 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 0.82rem; color: #94a3b8; text-transform: uppercase; font-weight:700;">Simulated Campaign Outcomes</div>
                <div style="margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div>
                        <div style="font-size: 0.75rem; color: #94a3b8;">Total Campaign Cost</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #F43F5E;">${total_campaign_cost:,.0f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: #94a3b8;">Expected Orders</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #38BDF8;">{expected_buyers:,}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: #94a3b8;">Gross Revenue</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #10B981;">${projected_gross_rev:,.0f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; color: #94a3b8;">Projected Net Profit</div>
                        <div style="font-size: 1.25rem; font-weight: 700; color: {'#10B981' if net_profit >= 0 else '#F43F5E'};">${net_profit:,.0f}</div>
                    </div>
                </div>
                <div style="margin-top: 12px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.08); font-size: 0.95rem;">
                    <b>Estimated Marketing ROI:</b> <span style="color: {'#10B981' if roi_pct >= 0 else '#F43F5E'}; font-weight: 800; font-size: 1.15rem;">{roi_pct:,.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ==========================================
# TAB 5: LIVE CUSTOMER CLASSIFIER
# ==========================================
with tab_classifier:
    st.subheader("Real-Time Customer Persona Classifier")
    st.caption("Input or configure a customer profile to instantly classify them into the corresponding segment using the trained PCA + KNN model.")
    
    with st.form("customer_classifier_form"):
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            st.markdown("##### 👤 Demographics")
            inp_age = st.slider("Customer Age", min_value=18, max_value=85, value=45)
            inp_income = st.number_input("Household Income ($)", min_value=5000, max_value=250000, value=58000, step=2500)
            inp_education = st.selectbox("Education Level", options=["Undergraduate", "Graduate", "Postgraduate"], index=1)
            inp_living = st.selectbox("Living Arrangement", options=["Partner", "Alone"], index=0)
            inp_kids = st.number_input("Children at Home (Kidhome)", min_value=0, max_value=3, value=0, step=1)
            inp_teens = st.number_input("Teens at Home (Teenhome)", min_value=0, max_value=3, value=1, step=1)
            
        with col_f2:
            st.markdown("##### 🛒 Product Spending ($)")
            inp_wines = st.number_input("Spend on Wines ($)", min_value=0, max_value=2000, value=300, step=25)
            inp_meats = st.number_input("Spend on Meat ($)", min_value=0, max_value=1500, value=180, step=20)
            inp_fish = st.number_input("Spend on Fish ($)", min_value=0, max_value=400, value=40, step=10)
            inp_sweets = st.number_input("Spend on Sweets ($)", min_value=0, max_value=400, value=30, step=10)
            inp_gold = st.number_input("Spend on Gold Products ($)", min_value=0, max_value=400, value=50, step=10)
            inp_fruits = st.number_input("Spend on Fruits ($)", min_value=0, max_value=400, value=25, step=5)
            
        with col_f3:
            st.markdown("##### 📱 Channel Behavior")
            inp_store = st.slider("Store Purchases", min_value=0, max_value=20, value=6)
            inp_web = st.slider("Web Purchases", min_value=0, max_value=20, value=5)
            inp_catalog = st.slider("Catalog Purchases", min_value=0, max_value=15, value=2)
            inp_deals = st.slider("Deal Purchases", min_value=0, max_value=15, value=2)
            inp_web_visits = st.slider("Web Visits / Month", min_value=0, max_value=20, value=5)
            inp_recency = st.slider("Recency (Days since last purchase)", min_value=0, max_value=100, value=35)
            inp_response = st.selectbox("Previous Campaign Accepted?", options=[0, 1], index=0)
            
        submit_btn = st.form_submit_button("🚀 Classify Customer Segment", use_container_width=True)
        
    if submit_btn:
        total_inp_spending = inp_wines + inp_meats + inp_fish + inp_sweets + inp_gold + inp_fruits
        total_inp_children = inp_kids + inp_teens
        
        customer_dict = {
            "Income": inp_income,
            "Age": inp_age,
            "Total_Spending": total_inp_spending,
            "Total_Children": total_inp_children,
            "Education": inp_education,
            "Living_With": inp_living,
            "NumStorePurchases": inp_store,
            "NumWebPurchases": inp_web,
            "NumCatalogPurchases": inp_catalog,
            "NumDealsPurchases": inp_deals,
            "NumWebVisitsMonth": inp_web_visits,
            "Recency": inp_recency,
            "Response": inp_response,
            "Customer_Tenure_Days": medians.get("Customer_Tenure_Days", 350)
        }
        
        pred_c, x_pca_single, probs = pipeline.predict_single_customer(
            customer_dict=customer_dict,
            feature_cols=feature_cols,
            feature_medians=medians,
            scaler=scaler,
            pca=pca,
            knn_model=knn_model
        )
        
        predicted_persona = personas.get(pred_c, {})
        
        st.write("")
        st.markdown("### 🎯 Classification Result")
        
        res_col1, res_col2 = st.columns([1, 1.2])
        with res_col1:
            st.markdown(
                f"""
                <div class="persona-card" style="border: 2px solid {predicted_persona.get('color', '#6366F1')};">
                    <span class="badge-pill" style="background-color: {predicted_persona.get('color', '#6366F1')}22; color: {predicted_persona.get('color', '#6366F1')}; border: 1px solid {predicted_persona.get('color', '#6366F1')};">
                        Predicted Segment #{pred_c} • {predicted_persona.get('badge', 'Classified')}
                    </span>
                    <h3 style="margin: 4px 0 8px 0; color: #f8fafc;">{predicted_persona.get('name', f'Cluster {pred_c}')}</h3>
                    <p style="color: #94a3b8; font-size: 0.9rem;">{predicted_persona.get('summary', '')}</p>
                    <div style="margin-top: 12px; font-size: 0.85rem; color: #f1f5f9;">
                        <b>Top Recommended Marketing Actions:</b>
                        <ul style="padding-left: 20px; margin-top: 6px;">
                            {''.join([f'<li style="margin-bottom: 4px;">{act}</li>' for act in predicted_persona.get('marketing_actions', [])])}
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with res_col2:
            st.markdown("##### 📍 PCA Projection Coordinates")
            st.write(f"• **PC1:** `{x_pca_single[0,0]:.2f}` | **PC2:** `{x_pca_single[0,1]:.2f}` | **PC3:** `{x_pca_single[0,2]:.2f}`")
            
            # Comparison against cluster centroid
            cluster_avg = df_full[df_full["Cluster"] == pred_c][["Income", "Total_Spending", "NumStorePurchases", "NumWebPurchases", "NumDealsPurchases"]].mean()
            comparison_df = pd.DataFrame({
                "Metric": ["Income ($)", "Total Spend ($)", "Store Orders", "Web Orders", "Deal Orders"],
                "Input Customer": [inp_income, total_inp_spending, inp_store, inp_web, inp_deals],
                "Segment Average": [cluster_avg["Income"], cluster_avg["Total_Spending"], cluster_avg["NumStorePurchases"], cluster_avg["NumWebPurchases"], cluster_avg["NumDealsPurchases"]]
            })
            
            fig_comp = go.Figure(data=[
                go.Bar(name='Input Customer', x=comparison_df['Metric'], y=comparison_df['Input Customer'], marker_color='#38BDF8'),
                go.Bar(name='Segment Average', x=comparison_df['Metric'], y=comparison_df['Segment Average'], marker_color=predicted_persona.get('color', '#818CF8'))
            ])
            fig_comp.update_layout(
                barmode='group',
                template="plotly_dark",
                height=300,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_comp, use_container_width=True)


# ==========================================
# TAB 6: DATA RECORDS & BATCH UPLOAD
# ==========================================
with tab_data:
    st.subheader("Segmented Customer Records & Batch File Processing")
    
    col_d_act1, col_d_act2 = st.columns([1.5, 1])
    with col_d_act1:
        st.markdown("#### 📄 Filtered Cohort Records")
        display_cols = ["ID", "Cluster", "Persona_Name", "Income", "Total_Spending", "Age", "Education", "Living_With", "Total_Children", "NumStorePurchases", "NumWebPurchases", "NumDealsPurchases", "Response"]
        st.dataframe(df_filtered[display_cols].head(100), use_container_width=True)
        
        # CSV Export
        csv_data = df_filtered[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Segmented Cohort (CSV)",
            data=csv_data,
            file_name="smartcart_segmented_customers.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    with col_d_act2:
        st.markdown("#### 📤 Batch Predict New Customer CSV")
        st.caption("Upload a new customer CSV file with the same columns to automatically clean, project, and label with cluster personas.")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
        
        if uploaded_file is not None:
            try:
                new_df_full, new_df_encoded, _, _ = pipeline.load_and_preprocess_data(uploaded_file)
                new_scaled = scaler.transform(new_df_encoded)
                new_pca = pca.transform(new_scaled)
                new_preds = knn_model.predict(new_pca)
                
                new_df_full["Cluster"] = new_preds
                new_df_full["Persona_Name"] = new_df_full["Cluster"].map(lambda c: personas.get(c, {}).get("name", f"Cluster {c}"))
                
                st.success(f"Successfully processed {len(new_df_full):,} customer records!")
                st.write(new_df_full[["ID", "Cluster", "Persona_Name", "Income", "Total_Spending"]].head())
                
                batch_csv = new_df_full.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Enriched Batch (CSV)",
                    data=batch_csv,
                    file_name="batch_segmented_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error processing uploaded CSV: {e}")
