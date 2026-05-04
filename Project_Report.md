<div align="center">

# COVID-19 Data Trend Analysis

**A Comprehensive Study on Epidemic Trajectories using Classical Data Mining and Novel Pattern Recognition Techniques**

---

**Course:** Data Mining and Data Warehouse  
**Project Report**  

</div>

<br>
<br>

---

## 1. Introduction

### 1.1 Background
The outbreak of the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2) in late 2019 precipitated a global health crisis unprecedented in modern history. As the COVID-19 pandemic swept across the globe, it laid bare the vulnerabilities in international public health infrastructure and drastically disrupted social and economic systems. Amidst the chaos, data emerged as one of the most critical weapons in the fight against the virus. Governments, healthcare systems, and research institutions rapidly mobilized to track, aggregate, and publish daily statistics on confirmed cases, fatalities, and recoveries.

### 1.2 Importance of Data Mining in Epidemiology
Epidemiological data is inherently complex. It is characterized by high volatility, non-linear growth patterns, spatial dependencies, and temporal lags. Raw data alone is insufficient to drive policy decisions such as the implementation of lockdowns, resource allocation for hospitals, or the rollout of vaccination programs. 

Data mining and data warehousing techniques bridge the gap between raw statistics and actionable intelligence. By applying algorithms designed to uncover hidden patterns, researchers can:
- Forecast future infection rates.
- Understand the efficacy of non-pharmaceutical interventions.
- Categorize regions based on their epidemic trajectories to apply targeted strategies.

### 1.3 Context of the Study
This project delves into the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE) COVID-19 dataset. The dataset provides a granular, day-by-day account of the pandemic's progression across over 200 countries and regions. We apply classical data mining techniques—such as Seasonal-Trend Decomposition using LOESS (STL), Pearson Correlation, K-Means Clustering, and Apriori Association Mining—to establish a baseline understanding of the global trends.

Crucially, this project introduces a **novel hybrid data mining technique**: **Epidemic Trajectory Fingerprinting (ETF)**. ETF addresses the fundamental shortcomings of traditional scalar-based clustering algorithms by preserving the temporal shape of epidemic waves, rendering it invariant to the onset time of the infection.

---

## 2. Objective

The overarching objective of this project is to construct a robust, reproducible data analysis pipeline that transforms raw COVID-19 time-series data into meaningful public health insights.

### 2.1 Primary Objectives
1. **Data Integration and Cleaning**: To build an automated pipeline that ingests, cleans, and reshapes the wide-format JHU CSSE datasets into a normalized, analysis-ready structure.
2. **Exploratory Data Analysis (EDA)**: To compute foundational statistical metrics, identify the most affected regions, and visualize the temporal progression of the virus.
3. **Application of Standard Techniques**: To deploy established data mining methods (moving averages, decomposition, correlation matrices, K-Means clustering) to extract initial patterns regarding wave synchronicity and geographic impact.

### 2.2 Secondary Objective (The Novel Contribution)
4. **Development of Epidemic Trajectory Fingerprinting (ETF)**: To design, implement, and validate a novel, hybrid pattern recognition algorithm. This objective aims to cluster countries not by the sheer volume of cases, but by the *shape and structural evolution* of their epidemic waves, independent of when those waves occurred.

---

## 3. Methodology

### 3.1 Dataset Description
The project utilizes the widely cited **JHU CSSE COVID-19 Time-Series Dataset**.
- **Source**: [JHU CSSE GitHub Repository](https://github.com/CSSEGISandData/COVID-19)
- **Time Range**: January 22, 2020 – March 9, 2023 (Dataset archived).
- **Format**: Wide-format CSV files where rows represent regions and columns represent consecutive dates.
- **Metrics Tracked**: Cumulative Confirmed Cases, Cumulative Deaths, Cumulative Recovered Cases.

### 3.2 Data Preprocessing and Cleaning
Data warehouses require structured, clean, and normalized data. The raw JHU data presents several challenges:
1. **Wide-to-Long Transformation**: The raw data contains dates as columns. We utilized the `pd.melt()` function to pivot the DataFrame into a tidy long format `(Country, Date, Metric)`.
2. **Aggregation**: Sub-national data (Provinces/States) was aggregated to the national level using a `groupby().sum()` operation to ensure a uniform level of geographic granularity.
3. **Daily Delta Calculation**: Since the data is cumulative, we engineered daily metrics by calculating the discrete difference $C_t - C_{t-1}$.
4. **Anomaly Correction**: Negative daily values (caused by retrospective corrections by health ministries) were handled using a bounding function: `daily_cases = max(0, daily_cases)`.

```python
# Code Snippet: Daily Delta Calculation and Anomaly Correction
for col in ["confirmed", "deaths"]:
    daily_col = f"daily_{col}"
    merged[daily_col] = merged.groupby("country")[col].diff().fillna(0)
    merged[daily_col] = merged[daily_col].clip(lower=0).astype(int)
```

### 3.3 Exploratory Data Analysis (EDA)
EDA forms the foundation of our analytical pipeline. We computed:
- **Daily Growth Rate**: $r = \frac{C_t - C_{t-1}}{C_{t-1}}$
- **Doubling Time**: $T_d = \frac{\ln(2)}{\ln(1 + r)}$
- **Fatality and Recovery Rates**: Calculated at the peak of the pandemic to gauge healthcare system strain.

Visualizations such as choropleth maps and daily distribution histograms were utilized to establish a macro-level view of the data.

### 3.4 Standard Data Mining Techniques

#### 3.4.1 Rolling Averages and STL Decomposition
Daily COVID-19 reporting is notoriously noisy due to weekend reporting lags. We applied a 7-day and 14-day Simple Moving Average (SMA) to extract the underlying signal.
To further dissect the time-series, we applied **STL (Seasonal-Trend decomposition using LOESS)**:
$$Y_t = T_t + S_t + R_t$$
Where $T_t$ represents the underlying epidemic wave (trend), $S_t$ captures the 7-day reporting cycle (seasonality), and $R_t$ isolates true anomalies (residuals).

#### 3.4.2 Pearson and Lagged Cross-Correlation
To determine wave synchronicity between countries, we computed the Pearson correlation coefficient $r$. Furthermore, to identify "leading" and "lagging" countries in the context of viral spread, we computed the cross-correlation at various time lags $\tau \in [-30, 30]$:
$$r_{xy}(\tau) = \frac{\sum (x(t) - \bar{x})(y(t+\tau) - \bar{y})}{\sqrt{\sum(x(t) - \bar{x})^2 \sum(y(t+\tau) - \bar{y})^2}}$$

#### 3.4.3 K-Means Clustering
We utilized K-Means to cluster countries based on scalar features: Total Cases, Peak Daily Cases, Days to Peak, and Fatality Rate. The optimal number of clusters ($K$) was determined using a dual-axis evaluation of the **Elbow Method (Inertia)** and the **Silhouette Score**. Data was standardized using Z-score normalization prior to clustering.

#### 3.4.4 Apriori Association Mining
To uncover temporal associations, daily case counts were discretized into categorical bands (Low, Medium, High, Surge). The Apriori algorithm was then applied to discover rules formatted as:
`{Country_A: Surge} => {Country_B: High}`
We evaluated rules based on **Support**, **Confidence**, and **Lift**.

---

### 3.5 Novel Technique: Epidemic Trajectory Fingerprinting (ETF)

The standard K-Means approach relies entirely on scalar aggregates. If Country A has 1 million cases in a single 2-month spike, and Country B has 1 million cases spread over 2 years, standard K-Means often groups them together. 

To solve this, we designed **Epidemic Trajectory Fingerprinting (ETF)**, a novel, 5-stage hybrid pipeline.

#### Step 1: Curve Normalization
Each country's daily-case time-series vector $\vec{X}$ is Min-Max normalized to scale the wave amplitude between 0 and 1, ensuring the algorithm focuses purely on the *shape* of the curve.
$$X'_{t} = \frac{X_t - X_{min}}{X_{max} - X_{min}}$$

#### Step 2: Multi-Scale Wavelet Decomposition
Time-series data contains localized features (sudden outbreaks) and global features (long-term plateaus). We apply a **Discrete Wavelet Transform (DWT)** using the Daubechies-4 (`db4`) mother wavelet up to 3 levels. We concatenate the approximation coefficients and detail coefficients to form a multi-scale feature vector.

#### Step 3: Dynamic Time Warping (DTW)
Standard Euclidean distance aligns $X_t$ with $Y_t$. This fails if Country A's wave occurs in March and Country B's wave occurs in June. We apply **Dynamic Time Warping (DTW)** to calculate a time-shift invariant distance matrix $D$. DTW finds the optimal non-linear alignment between two sequences to minimize the distance measure.

#### Step 4: Spectral Embedding (Laplacian Eigenmaps)
The DTW distance matrix $D$ is converted into a Gaussian Affinity Matrix $A$:
$$A_{i,j} = \exp\left(-\frac{D_{i,j}^2}{2\sigma^2}\right)$$
We then perform spectral embedding to map the high-dimensional, non-linear trajectory relationships into a dense 2D space.

#### Step 5: HDBSCAN Clustering
Finally, we apply **Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN)** to the spectral embedding. Unlike K-Means, HDBSCAN:
1. Does not require the user to pre-define $K$.
2. Can identify arbitrarily shaped clusters.
3. Elegantly handles noise (countries with unique trajectories that do not fit a family).

```python
# Code Snippet: The ETF Pipeline Execution
def run_etf(df, countries):
    curves = prepare_curves(df, countries)
    wt_curves = wavelet_transform(curves, wavelet='db4', level=3)
    D, names = compute_dtw_matrix(wt_curves)
    A = dtw_to_affinity(D)
    emb = spectral_embed(A, n_components=2)
    labels = cluster_hdbscan(emb, min_cluster_size=3)
    return labels
```

---

## 4. Results and Discussion

### 4.1 Exploratory Data Analysis Results
The EDA revealed extreme skewness in the global distribution of the virus. The top 10 countries (led by the US, India, France, Germany, and Brazil) accounted for a vast majority of the total global cases. The daily distribution analysis highlighted the hyper-volatility of the pandemic, with maximum daily spikes frequently exceeding the 75th percentile (Q3) by factors of 10 or more, underscoring the "wave-like" nature of the contagion.

### 4.2 Time Series & Decomposition Results
The application of the 7-day and 14-day Simple Moving Averages effectively smoothed the administrative artifacts present in the raw data (e.g., the infamous "weekend dip" where testing centers reported fewer results).

The **STL Decomposition** yielded profound insights. Visual analysis of the US decomposition (`report_images/stl_decomposition.png`) clearly delineated three massive macro-trends:
1. The initial winter 2020-2021 surge.
2. The Delta variant wave in late summer 2021.
3. The unprecedented Omicron vertical spike in early 2022.
The seasonal component perfectly isolated a rigid 7-day oscillation with an amplitude of tens of thousands of cases, mathematically proving the impact of weekend reporting delays.

### 4.3 Correlation and Lag Analysis
The **Pearson Correlation Matrix** highlighted structural similarities between neighboring European nations (e.g., France and Germany showing $r > 0.7$). 

The **Lagged Cross-Correlation** between the United States and India demonstrated the value of temporal shifting. While the simultaneous correlation (lag=0) was modest, applying a lag of approximately $\pm 20$ to $40$ days resulted in a sharp spike in the correlation coefficient. This mathematically corroborates the timeline of the global spread, showing that waves hitting the US often preceded or followed waves in India by a mathematically quantifiable margin.

### 4.4 K-Means Clustering Outputs
Using standard scalar features, K-Means was heavily influenced by the sheer scale of the population. The optimal cluster count determined via the Silhouette Score and Elbow Method was $K=4$.
- **Cluster 0**: Massive total case counts (e.g., US, India).
- **Cluster 1**: High fatality rates relative to cases (older demographic nations).
- **Cluster 2 & 3**: Developing nations and island nations with delayed onset.

While useful for resource allocation, K-Means failed to group countries by the *behavior* of the virus.

### 4.5 Association Rules
The Apriori algorithm successfully generated actionable association rules. By discretizing case counts into bands, the algorithm identified patterns such as:
`{France_Surge, Italy_High} => {Germany_Surge}` 
Rules boasting a **Confidence > 80%** and **Lift > 2.0** mathematically demonstrated regional contagion synchronicity. When one major European hub entered a surge, the probability of neighbors entering a surge within the same localized timeframe was highly elevated.

---

### 4.6 Epidemic Trajectory Fingerprinting (ETF) Results
The novel ETF technique fundamentally altered our clustering perspective. 

1. **DTW Heatmap Analysis**: The Dynamic Time Warping distance matrix (`report_images/etf_dtw_heatmap.png`) showed distinct, dark "blocks" of structural similarity between countries that were grouped together regardless of whether their peak occurred in 2020 or 2021.
2. **Spectral Manifold**: The spectral embedding successfully unfolded the complex wavelet data into a 2-dimensional space, where dense pockets of countries formed.
3. **HDBSCAN Clustering**: The algorithm autonomously discovered specific "Trajectory Families" without human intervention:
   - **The "Sharp Spike" Family**: Countries that managed to suppress the virus entirely for over a year, only to experience one massive, vertical Omicron spike (e.g., Australia, New Zealand, South Korea).
   - **The "Multi-Wave" Family**: Countries that experienced cyclical, sustained waves every 4-6 months (e.g., US, UK, France).
   - **The "Plateau" Family**: Nations where testing limitations or specific containment strategies resulted in long, flat, sustained case loads rather than sharp spikes.

By combining Wavelets (multi-scale resolution) and DTW (time-shift invariance), ETF successfully solved the exact limitations of standard K-Means.

---

## 5. Conclusion

### 5.1 Summary of Findings
This project successfully applied a comprehensive suite of data mining methodologies to the JHU CSSE COVID-19 dataset. We transitioned from basic exploratory data analysis to complex time-series decomposition, proving that the pandemic's data is heavily influenced by both biological transmission waves and administrative reporting cycles. We utilized K-Means to cluster nations by aggregate impact, and Apriori association mining to mathematically prove the geographic synchronicity of regional surges.

### 5.2 Impact of the Novel Technique
The introduction of the **Epidemic Trajectory Fingerprinting (ETF)** method stands as the major contribution of this study. While existing literature frequently utilizes K-Means or basic hierarchical clustering, those methods natively fail to handle the temporal displacement of epidemic waves. 

ETF's elegant integration of:
1. **Wavelet Transforms** to handle multi-scale epidemic features.
2. **Dynamic Time Warping** to align temporally shifted waves.
3. **Spectral Embedding and HDBSCAN** to discover non-linear, arbitrarily shaped density clusters.

...resulted in a paradigm shift. Instead of asking, *"Which countries had the most cases?"*, ETF allowed the data to answer, *"Which countries fought the virus in the exact same structural manner, regardless of when the battle occurred?"*

### 5.3 Future Scope
The ETF pipeline is highly modular and holds immense potential for future epidemiological research. Future iterations of this project could:
- Integrate global vaccination rollout data to see if trajectory shapes altered post-vaccine.
- Apply the ETF methodology to sub-national data (e.g., US States or Indian States) to track localized epidemic behaviors.
- Adapt the ETF pipeline to forecast future anomalies by comparing incoming real-time data streams against the historical "fingerprints" generated in this study.

<br>
<br>

---
<div align="center">
<i>Report generated utilizing Python Data Stack (pandas, scipy, scikit-learn, statsmodels).<br>
Dataset courtesy of Johns Hopkins University CSSE.</i>
</div>
