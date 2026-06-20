# Walmart-Store-Cluster-Analysis🛒📊
Walmart Store Cluster Analysis is a Python project that segments 45 Walmart stores into meaningful groups based on sales performance and macroeconomic indicators — average weekly sales, temperature, fuel price, CPI, and unemployment. The project compares three different unsupervised learning algorithms and evaluates them with the silhouette score, then presents the full analysis as an interactive HTML report. 

🚀 Features
Store profiling — weekly records aggregated into one profile per store (mean sales, temperature, fuel price, CPI, unemployment)
Standardization — all features scaled with StandardScaler so no variable dominates due to scale
Silhouette analysis — K tested from 2 to 10 to justify the final choice of K=4
Three clustering algorithms compared:
- K-Means
- Hierarchical clustering (Ward linkage), with dendrogram
- DBSCAN, with outlier detection

📈 Visual outputs — 5 charts generated automatically: silhouette curve, K-Means scatter, Ward dendrogram, DBSCAN scatter, final algorithm comparison

🌐 Interactive HTML report — full write-up of the analysis with embedded charts and explanations

⚙️ Requirements
- Python 3.10+
- pandas
- numpy
- matplotlib
- scikit-learn
- scipy

▶️ How to Run
Download the dataset and place it in the project root as Walmart_Store_sales.csv
Run python clustering.py
The script will:
- Load and aggregate the data by store
- Run silhouette analysis to confirm K
- Fit K-Means, Ward, and DBSCAN
- Print cluster sizes and silhouette scores for each algorithm
- Save 5 PNG charts to the project folder

🗃️ Data Sources
The CSV file used in this project is the Walmart Store Sales dataset, sourced from Kaggle: https://www.kaggle.com/datasets/yasserh/walmart-dataset


👤 Author
Aurora Michielin. Created for the AI e Machine Learning per il Marketing course at IULM University.
