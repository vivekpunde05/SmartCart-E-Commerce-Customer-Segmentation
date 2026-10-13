import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from typing import Dict, Tuple, Any, List
import warnings

# Curated business personas for the notebook's baseline 4-cluster model
DEFAULT_4_CLUSTER_PERSONAS = {
    0: {
        "name": "Budget Family Partners",
        "badge": "Family Value Hunter",
        "color": "#38BDF8",  # Sky Blue
        "summary": "Couples & families with children, modest household income, highly price-conscious deal hunters.",
        "traits": [
            "100% Living with a Partner",
            "High child ratio (Avg 1.24 kids/teens)",
            "Moderate income (~$40.7K) & low discretionary spend (~$231)",
            "Frequent deal-seeking purchases (2.59 deals/customer)",
            "Moderate digital browsing (~6.18 web visits/month)",
            "Low marketing campaign responsiveness (~7.0%)"
        ],
        "marketing_actions": [
            "Offer family bundle discounts, BOGO promotions, and back-to-school value packages",
            "Promote essential pantry staples, family groceries, and mid-tier product lines",
            "Send digital discount vouchers via email & SMS push notifications",
            "Emphasize cost savings, loyalty reward points, and bulk pricing"
        ],
        "primary_channels": "Physical Retail Stores & Deal Pages",
        "top_categories": "Affordable Wines, Everyday Staples, Basic Groceries"
    },
    1: {
        "name": "Affluent Power Couples",
        "badge": "High Net-Worth VIP",
        "color": "#10B981",  # Emerald Green
        "summary": "Wealthy partner households with few or no kids, commanding top spending across premium categories.",
        "traits": [
            "100% Living with a Partner",
            "Low child ratio (Avg ~0.49 kids/teens)",
            "High household income (~$72.6K) & top spend (~$1,270)",
            "Heavy in-store (~8.7) and catalog (~5.6) buyer",
            "Infrequent deal hunters (<1.9 deals), valuing quality over discounts",
            "Solid campaign conversion rate (~18.3%)"
        ],
        "marketing_actions": [
            "Deliver VIP concierge services, private tastings, and reserved seasonal allocations",
            "Curate high-end wine club memberships and gourmet food gift hampers",
            "Target with high-gloss print catalogs and personalized email invitations",
            "Avoid heavy discount framing; focus on craftsmanship, heritage, and exclusivity"
        ],
        "primary_channels": "In-Store Boutiques & Premium Catalogs",
        "top_categories": "Fine Wines, Premium Meats, Luxury Goods"
    },
    2: {
        "name": "Value-Conscious Singles",
        "badge": "Budget Digital Savvy",
        "color": "#F59E0B",  # Amber
        "summary": "Solo heads of household or single parents balancing budget constraints with high digital browsing.",
        "traits": [
            "99%+ Living Alone / Single Parents",
            "High child ratio (Avg 1.24 kids/teens)",
            "Modest income (~$39.5K) & low spending (~$267)",
            "Highest digital browsing activity (~6.59 web visits/month)",
            "Strong deal-hunter behavior (~2.68 deals purchases)",
            "Moderate campaign receptiveness (~15.9%)"
        ],
        "marketing_actions": [
            "Deploy mobile-first push notifications and flash discount alerts",
            "Incentivize with free shipping thresholds and cart-abandonment coupons",
            "Promote budget-friendly meal kits and affordable child-friendly goods",
            "Focus on web UX and quick 1-click checkout options"
        ],
        "primary_channels": "E-Commerce Website & Deal Section",
        "top_categories": "Budget Essentials, Household Groceries, Discount Deals"
    },
    3: {
        "name": "Elite Affluent Singles",
        "badge": "Gourmet Trendsetters",
        "color": "#8B5CF6",  # Vibrant Purple
        "summary": "High-income single professionals or empty nesters with the highest campaign response rate and premium taste.",
        "traits": [
            "100% Living Alone (Single, Divorced, Widowed)",
            "Fewest dependents (Avg ~0.38 children)",
            "High income (~$72.3K) & high spending (~$1,201)",
            "Highest campaign response rate of all segments (32.2%!)",
            "Lowest deal-hunting behavior (1.59 deals)",
            "Balanced omni-channel buyers (Web, Store, Catalog)"
        ],
        "marketing_actions": [
            "Prioritize for new product launches, limited releases, and premium campaigns",
            "Deliver tailored lifestyle and culinary experiences (artisanal wines, gourmet dining)",
            "Leverage personalized digital retargeting and premium social channels",
            "Provide frictionless high-speed checkout and luxury unboxing experiences"
        ],
        "primary_channels": "Omnichannel (Web, Store & Catalog)",
        "top_categories": "Fine Wines, Organic/Artisanal Meats, Luxury Gold Goods"
    }
}


def load_and_preprocess_data(source="smartcart_customers.csv") -> Tuple[pd.DataFrame, pd.DataFrame, list, dict]:
    """
    Loads raw customer data and performs the exact cleaning and feature engineering 
    defined in the SmartCart project notebook.
    Returns:
        df_full: DataFrame with all original columns + engineered features (cleaned of outliers)
        df_encoded: Numerical & One-Hot Encoded feature matrix ready for PCA / Clustering
        feature_cols: List of column names in df_encoded
        feature_medians: Median dictionary used for default inference values
    """
    if isinstance(source, str):
        df = pd.read_csv(source)
    else:
        df = pd.read_csv(source)
    
    # Missing value imputation (Income median)
    df["Income"] = df["Income"].fillna(df["Income"].median())
    
    # Feature Engineering
    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt_Customer_Parsed"] = pd.to_datetime(df["Dt_Customer"], format='mixed')
    ref_date = df["Dt_Customer_Parsed"].max()
    df["Customer_Tenure_Days"] = (ref_date - df["Dt_Customer_Parsed"]).dt.days
    
    df["Total_Spending"] = (
        df["MntWines"] + 
        df["MntFruits"] + 
        df["MntMeatProducts"] + 
        df["MntFishProducts"] + 
        df["MntSweetProducts"] + 
        df["MntGoldProds"]
    )
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    
    # Education regrouping
    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate",
        "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate",
        "PhD": "Postgraduate"
    })
    
    # Marital Status regrouping
    df["Living_With"] = df["Marital_Status"].replace({
        "Married": "Partner",
        "Together": "Partner",
        "Single": "Alone",
        "Divorced": "Alone",
        "Widow": "Alone",
        "Absurd": "Alone",
        "YOLO": "Alone"
    })
    
    # Total Campaign Acceptance Count
    df["Total_Accepted_Campaigns"] = (
        df["AcceptedCmp1"] + 
        df["AcceptedCmp2"] + 
        df["AcceptedCmp3"] + 
        df["AcceptedCmp4"] + 
        df["AcceptedCmp5"] + 
        df["Response"]
    )
    
    # Outlier removal: Age < 90 and Income < 600,000 (Notebook cell 23)
    mask_valid = (df["Age"] < 90) & (df["Income"] < 600_000)
    df_full = df[mask_valid].copy()
    
    # Columns to drop for clustering feature matrix (as per notebook cell 18)
    cols_to_drop = [
        "ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", 
        "Dt_Customer", "Dt_Customer_Parsed",
        "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", 
        "MntSweetProducts", "MntGoldProds", "Total_Accepted_Campaigns"
    ]
    df_clustering = df_full.drop(columns=[c for c in cols_to_drop if c in df_full.columns])
    
    # Categorical One-Hot Encoding
    cat_cols = ["Education", "Living_With"]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    enc_array = ohe.fit_transform(df_clustering[cat_cols])
    enc_cols_names = ohe.get_feature_names_out(cat_cols)
    enc_df = pd.DataFrame(enc_array, columns=enc_cols_names, index=df_clustering.index)
    
    df_encoded = pd.concat([df_clustering.drop(columns=cat_cols), enc_df], axis=1)
    
    # Calculate medians for reliable customer inference fallback
    feature_medians = df_encoded.median().to_dict()
    
    return df_full, df_encoded, df_encoded.columns.tolist(), feature_medians


def compute_pca_pipeline(df_encoded: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler, PCA]:
    """
    Standardizes features and fits a 3-component PCA faithfully.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_scaled, X_pca, scaler, pca


def compute_clustering_diagnostics(X_pca: np.ndarray, max_k: int = 10) -> Dict[str, Any]:
    """
    Computes WCSS and Silhouette Scores across range 2..max_k to find optimal K.
    """
    wcss = []
    k_range = list(range(1, max_k + 1))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca)
        wcss.append(km.inertia_)
        
    silhouette_scores = []
    eval_range = list(range(2, max_k + 1))
    for k in eval_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        silhouette_scores.append(score)
        
    try:
        knee = KneeLocator(k_range, wcss, curve="convex", direction="decreasing")
        optimal_k = knee.elbow if knee.elbow is not None else 4
    except Exception:
        optimal_k = 4
    
    return {
        "k_range_wcss": k_range,
        "wcss": wcss,
        "k_range_sil": eval_range,
        "silhouette_scores": silhouette_scores,
        "optimal_k": optimal_k
    }


def fit_cluster_model(X_pca: np.ndarray, algorithm: str = "Agglomerative", n_clusters: int = 4) -> Tuple[np.ndarray, KNeighborsClassifier]:
    """
    Fits the requested clustering model on PCA space and trains a KNN classifier 
    for fast, consistent inference on new customer profiles.
    """
    if algorithm == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(X_pca)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_pca)
        
    # Classifier to enable instant out-of-sample inference
    knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
    knn.fit(X_pca, labels)
    
    return labels, knn


def generate_personas_for_clusters(df_full: pd.DataFrame, cluster_col: str = "Cluster") -> Dict[int, Dict[str, Any]]:
    """
    Dynamically generates or maps personas for each cluster based on their empirical averages.
    """
    clusters = sorted(df_full[cluster_col].unique())
    n_clusters = len(clusters)
    
    # If standard 4-cluster model with matching profile, use curated personas
    if n_clusters == 4:
        c0_spend = df_full[df_full[cluster_col] == 0]["Total_Spending"].mean()
        c1_spend = df_full[df_full[cluster_col] == 1]["Total_Spending"].mean()
        c2_spend = df_full[df_full[cluster_col] == 2]["Total_Spending"].mean()
        c3_spend = df_full[df_full[cluster_col] == 3]["Total_Spending"].mean()
        
        if c1_spend > c0_spend and c3_spend > c2_spend:
            return DEFAULT_4_CLUSTER_PERSONAS
            
    # Dynamic fallback persona generation for any K or custom clustering
    palette = ["#38BDF8", "#10B981", "#F59E0B", "#8B5CF6", "#EC4899", "#14B8A6", "#F97316", "#6366F1"]
    overall_mean_income = df_full["Income"].mean()
    overall_mean_spend = df_full["Total_Spending"].mean()
    
    personas = {}
    for idx, c in enumerate(clusters):
        sub = df_full[df_full[cluster_col] == c]
        avg_inc = sub["Income"].mean()
        avg_spend = sub["Total_Spending"].mean()
        avg_kids = sub["Total_Children"].mean()
        pct_partner = (sub["Living_With"] == "Partner").mean() * 100
        resp_rate = sub["Response"].mean() * 100
        deal_purchases = sub["NumDealsPurchases"].mean()
        
        # Determine income/spending tier
        if avg_inc > overall_mean_income * 1.15 and avg_spend > overall_mean_spend * 1.15:
            tier = "Affluent"
            badge = "High Value VIP"
        elif avg_inc < overall_mean_income * 0.9:
            tier = "Budget"
            badge = "Value Focused"
        else:
            tier = "Core Mainstream"
            badge = "Mid-Market"
            
        household = "Couples" if pct_partner > 60 else ("Singles" if pct_partner < 40 else "Mixed Households")
        if avg_kids >= 1.0:
            household += " & Families"
            
        name = f"{tier} {household}"
        color = palette[idx % len(palette)]
        
        personas[c] = {
            "name": name,
            "badge": badge,
            "color": color,
            "summary": f"Segment with avg income ${avg_inc:,.0f}, avg spend ${avg_spend:,.0f}, and {avg_kids:.1f} children per home.",
            "traits": [
                f"{pct_partner:.0f}% Living with Partner",
                f"Average {avg_kids:.2f} children / teens",
                f"Household Income: ${avg_inc:,.0f}",
                f"Annual Total Spending: ${avg_spend:,.0f}",
                f"Deals Purchases: {deal_purchases:.2f}",
                f"Campaign Acceptance: {resp_rate:.1f}%"
            ],
            "marketing_actions": [
                f"Target with {'premium, high-touch luxury products' if avg_spend > overall_mean_spend else 'discount bundles and price-conscious incentives'}",
                f"Preferred purchase channels: {'Store & Catalog' if sub['NumStorePurchases'].mean() > 6 else 'Web & Deals'}",
                f"{'VIP loyalty rewards program' if avg_spend > overall_mean_spend else 'Coupon promotions and free delivery thresholds'}"
            ],
            "primary_channels": "In-Store & Web" if sub['NumStorePurchases'].mean() > 4 else "Web & Deals",
            "top_categories": "Wines & Gourmet" if avg_spend > 500 else "Everyday Essentials"
        }
        
    return personas


def predict_single_customer(
    customer_dict: Dict[str, Any],
    feature_cols: list,
    feature_medians: dict,
    scaler: StandardScaler,
    pca: PCA,
    knn_model: KNeighborsClassifier
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Transforms a single customer input dictionary into standardized PCA space
    and predicts their cluster segment.
    """
    # Start with feature medians to ensure robust fallbacks
    row_dict = feature_medians.copy()
    
    # Update provided numerical features
    for col in [
        "Income", "Recency", "NumDealsPurchases", "NumWebPurchases", 
        "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth", 
        "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "AcceptedCmp1", 
        "AcceptedCmp2", "Complain", "Z_CostContact", "Z_Revenue", 
        "Response", "Age", "Customer_Tenure_Days", "Total_Spending", "Total_Children"
    ]:
        if col in customer_dict and customer_dict[col] is not None:
            row_dict[col] = float(customer_dict[col])
            
    # Reset one-hot columns first
    for col in feature_cols:
        if col.startswith("Education_") or col.startswith("Living_With_"):
            row_dict[col] = 0.0
            
    # Set One-Hot encoded categories
    edu_col = f"Education_{customer_dict.get('Education', 'Graduate')}"
    if edu_col in row_dict:
        row_dict[edu_col] = 1.0
        
    living_col = f"Living_With_{customer_dict.get('Living_With', 'Partner')}"
    if living_col in row_dict:
        row_dict[living_col] = 1.0
        
    # Build single-row DataFrame with exact columns to avoid sklearn warning
    df_single = pd.DataFrame([row_dict])[feature_cols]
    
    # Standardize and project into PCA
    X_single_scaled = scaler.transform(df_single)
    X_single_pca = pca.transform(X_single_scaled)
    
    pred_cluster = int(knn_model.predict(X_single_pca)[0])
    probabilities = knn_model.predict_proba(X_single_pca)[0] if hasattr(knn_model, "predict_proba") else None
    
    return pred_cluster, X_single_pca, probabilities
