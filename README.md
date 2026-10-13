<div align="center">

# 🛒 SmartCart: E-Commerce Customer Segmentation & Behavioral Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.54.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-6.9%2B-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

<p align="center">
  <b>Transforming multidimensional customer transactional data into actionable marketing personas, predictive segmentation, and executive revenue intelligence.</b>
</p>

[Key Features](#-key-features) • [Customer Personas](#-customer-personas--behavioral-profiles) • [ML Pipeline](#-machine-learning-architecture) • [Dashboard Overview](#-interactive-streamlit-dashboard) • [Installation & Usage](#-getting-started) • [Dataset Dictionary](#-dataset-dictionary)

</div>

---

## 📌 Executive Summary

Modern e-commerce retail generates vast amounts of omnichannel transactional data. Without segmentation, marketing campaigns rely on generic, one-size-fits-all communications—wasting marketing budget and depressing conversion rates.

**SmartCart Customer Segmentation & Behavioral Intelligence** is an end-to-end data science and machine learning solution that cleans, transforms, reduces, and clusters retail customer personality records (`smartcart_customers.csv`). Using **Principal Component Analysis (PCA)** and **Agglomerative Hierarchical Clustering (Ward's Linkage)**, the platform discovers four distinct behavioral customer personas and provides an interactive, executive-grade **Streamlit Dashboard** complete with real-time customer classification, ROI simulation, and batch scoring.

---

## 🌟 Key Features

- **📊 Executive KPI Analytics**: Real-time evaluation of total gross volume ($1.35M+), cohort size (2,236 customers), average lifetime spend ($607), and baseline campaign conversion (14.9%).
- **🔬 3D Latent PCA Customer Space**: Interactive, rotatable 3D scatter projection of customer embeddings colored by persona and sized by total spend.
- **📈 Unsupervised Clustering Diagnostics**: Automated Elbow Curve analysis with `kneed.KneeLocator` and Silhouette Score evaluation across $K \in [2, 8]$.
- **🕸️ Multidimensional Segment Radar Profiling**: Normalized comparative spider charts revealing purchasing strength across income, spend, family size, deal sensitivity, store shopping, and web engagement.
- **💰 Marketing Campaign Performance & ROI Simulator**: Historical conversion heatmaps across 6 marketing campaigns with a simulation calculator projecting net profit and ROI for targeted initiatives.
- **🔮 Real-Time Customer Persona Classifier**: Instant out-of-sample inference utilizing a distance-weighted $K$-Nearest Neighbors model on PCA coordinates to classify new prospective customers and suggest immediate marketing strategies.
- **📁 Filterable Cohort Browser & Batch CSV Processing**: Interactive data table with single-click CSV export and batch file ingestion for scoring new customer files.

---
## 💻 Interactive Streamlit Dashboard

The web application [`app.py`](app.py) provides a multi-tab interface styled in an executive dark theme:

| Tab | Name | Description |
|:---:|:---|:---|
| **1** | **📊 Executive EDA** | Income and spending distribution histograms, product mix breakdown (Wines, Meats, etc.), channel share, and correlation heatmaps. |
| **2** | **🔬 PCA & Cluster Space** | Diagnostic Elbow and Silhouette charts, rotatable 3D PCA customer scatter plot, and bivariate metric explorer. |
| **3** | **👥 Customer Personas** | Deep-dive persona cards, normalized segment radar chart, product category spending per segment, and channel benchmarks. |
| **4** | **📈 Campaign ROI** | Historical conversion matrix across Campaigns 1–5 and Response, paired with an interactive marketing ROI and profit simulator. |
| **5** | **🔮 Live Classifier** | Form for testing any customer profile with immediate cluster prediction, PCA coordinates, and comparison against segment centroids. |
| **6** | **📁 Data & Batch Processing** | Filterable cohort data table, CSV download, and automated batch CSV scoring engine. |

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/smartcart-customer-segmentation.git
   cd "smartcart-customer-segmentation"
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit dashboard**:
   ```bash
   streamlit run app.py
   ```

5. **Access the application**:
   Open your browser and navigate to:
   ```
   http://localhost:8501
   ```

---

## 📖 Dataset Dictionary

The dataset [`smartcart_customers.csv`](smartcart_customers.csv) contains 2,240 customer records across 29 original attributes:

| Category | Attribute | Description |
|:---|:---|:---|
| **Customer Profile** | `ID` | Unique customer identifier |
| | `Year_Birth` | Customer birth year (used to derive `Age`) |
| | `Education` | Educational qualification (`Basic`, `2n Cycle`, `Graduation`, `Master`, `PhD`) |
| | `Marital_Status` | Marital status (`Single`, `Together`, `Married`, `Divorced`, `Widow`) |
| | `Income` | Annual household income in USD |
| | `Kidhome` / `Teenhome` | Number of small children / teenagers in customer household |
| | `Dt_Customer` | Date of customer registration with SmartCart |
| | `Recency` | Number of days since customer's last purchase |
| | `Complain` | 1 if customer complained in the last 2 years, 0 otherwise |
| **Product Spend ($)** | `MntWines` | Amount spent on wine products in the last 2 years |
| | `MntFruits` | Amount spent on fruit products |
| | `MntMeatProducts` | Amount spent on meat products |
| | `MntFishProducts` | Amount spent on fish products |
| | `MntSweetProducts` | Amount spent on sweets and confectionery |
| | `MntGoldProds` | Amount spent on gold and luxury items |
| **Channels & Deals** | `NumDealsPurchases` | Number of purchases made with discount deals |
| | `NumWebPurchases` | Number of purchases made through company's website |
| | `NumCatalogPurchases` | Number of purchases made using product catalogs |
| | `NumStorePurchases` | Number of purchases made directly in physical stores |
| | `NumWebVisitsMonth` | Number of visits to company's website in the last month |
| **Campaigns** | `AcceptedCmp1` – `5` | 1 if customer accepted campaign offer in campaigns 1 through 5 |
| | `Response` | 1 if customer accepted offer in the final marketing campaign |

---

## 🗂️ Project Structure

```
SmartCart E_Commerce Customer Segmentation/
├── app.py                     # Main interactive Streamlit dashboard application
├── pipeline.py                # Machine learning pipeline: data prep, PCA, clustering & inference
├── smartcart_customers.csv    # Source customer dataset (2,240 records)
├── smartcart.ipynb            # Original exploratory Jupyter Notebook
├── requirements.txt           # Python dependency specifications
└── README.md                  # Comprehensive project documentation
```

---

## 📦 Dependencies

- **[Streamlit](https://streamlit.io/)**: Interactive web application framework.
- **[Scikit-Learn](https://scikit-learn.org/)**: Preprocessing, PCA, Agglomerative Clustering, KMeans, and KNN.
- **[Plotly](https://plotly.com/)**: Dynamic 3D scatter plots, radar charts, heatmaps, and distribution plots.
- **[Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)**: Data manipulation, aggregation, and linear algebra.
- **[Kneed](https://github.com/arvkevi/kneed)**: Automated knee/elbow detection for optimal cluster selection.
- **[Seaborn](https://seaborn.pydata.org/) & [Matplotlib](https://matplotlib.org/)**: Exploratory visual analytics.

---

## 🤝 Contributing

Contributions, feedback, and feature requests are welcome:
1. Fork the Project repository.
2. Create your Feature Branch (`git checkout -b feature/NewSegmentationModel`).
3. Commit your Changes (`git commit -m 'Add Gaussian Mixture Model option'`).
4. Push to the Branch (`git push origin feature/NewSegmentationModel`).
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
#