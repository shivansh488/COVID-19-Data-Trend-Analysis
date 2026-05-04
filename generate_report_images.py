import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath("."))
from src import data_ingest, eda, standard_techniques, etf, visualization

# Create directory for report images
os.makedirs("report_images", exist_ok=True)

# Load data
print("Loading data...")
df = data_ingest.load_merged()
top_countries = data_ingest.get_top_countries(df, n=30)
top_10 = top_countries[:10]

# 1. STL Decomposition
print("Generating STL...")
stl_res = standard_techniques.stl_decompose(df, "US", period=7)
fig1 = visualization.plot_stl(stl_res, "US")
fig1.savefig("report_images/stl_decomposition.png", dpi=300, bbox_inches="tight")
plt.close(fig1)

# 2. Correlation Matrix
print("Generating Correlation Matrix...")
corr = standard_techniques.correlation_matrix(df, top_10)
fig2 = visualization.plot_correlation_heatmap(corr, title="Correlation Matrix of Daily Cases (Top 10)")
fig2.savefig("report_images/correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.close(fig2)

# 3. K-Means Elbow
print("Generating K-Means Elbow...")
features = standard_techniques.build_country_features(df)
elbow = standard_techniques.elbow_search(features)
fig3 = visualization.plot_elbow(elbow)
fig3.savefig("report_images/kmeans_elbow.png", dpi=300, bbox_inches="tight")
plt.close(fig3)

# 4. K-Means Clusters
print("Generating K-Means Clusters...")
labels, pca_2d, sil = standard_techniques.kmeans_cluster(features, k=5)
fig4 = visualization.plot_kmeans_clusters(pca_2d, labels, features.index.tolist())
fig4.savefig("report_images/kmeans_clusters.png", dpi=300, bbox_inches="tight")
plt.close(fig4)

# 5. ETF Heatmap & Clusters
print("Generating ETF Plots...")
etf_result = etf.run_etf(df, top_countries, n_components=2, min_cluster_size=3)
fig5 = visualization.plot_dtw_heatmap(etf_result["dtw_matrix"], etf_result["countries"])
fig5.savefig("report_images/etf_dtw_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig5)

fig6 = visualization.plot_etf_clusters(etf_result["results_df"])
fig6.savefig("report_images/etf_clusters.png", dpi=300, bbox_inches="tight")
plt.close(fig6)

fig7 = visualization.plot_cluster_curves(etf_result["curves"], etf_result["results_df"])
if fig7:
    fig7.savefig("report_images/etf_trajectory_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig7)

# 6. Lagged Correlation
print("Generating Lagged Correlation...")
lag_corr = standard_techniques.lagged_cross_correlation(df, 'US', 'India', max_lag=30)
fig8, ax = plt.subplots(figsize=(10, 4))
ax.bar(lag_corr.index, lag_corr.values, width=0.8, alpha=0.8, color='teal')
ax.set_xlabel('Lag (days)')
ax.set_ylabel('Pearson r')
ax.set_title('Lagged Cross-Correlation: US vs India')
ax.axhline(0, color='gray', linestyle='--')
best_lag = lag_corr.idxmax()
ax.axvline(best_lag, color='red', linestyle=':', label=f'Best lag = {best_lag}')
ax.legend()
plt.tight_layout()
fig8.savefig("report_images/lagged_correlation.png", dpi=300, bbox_inches="tight")
plt.close(fig8)

print("All images generated successfully in report_images/ directory.")
