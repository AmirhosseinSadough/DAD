import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare
import scipy.stats
from scikit_posthocs import posthoc_nemenyi_friedman
sns.set()
from scipy.spatial import ConvexHull

from sklearn.cluster import KMeans
from matplotlib.patches import Rectangle
import sys

dataset_mode = "synthetic" # "synthetic", "benchmark", "damadics"

figure_dir = ""
table_dir = ""
prune = "datasets"        
evaluation_mode = "average" # "maximum", "average", "default", "hpt"

excluding_method = ["DECODE"]

if evaluation_mode == "maximum":
    result_dir = "results_max_mean/csvresult_dir"
    wc_dir = "results_max_mean/wc_dir"
    if excluding_method[0] == "DECODE_s":
        figure_dir = "results_max_mean/maximum/figures_DECODE"
        table_dir = "results_max_mean/maximum/tables_DECODE"
    elif excluding_method[0] == "DECODE":
        figure_dir = "results_max_mean/maximum/figures_DECODE_s"
        table_dir = "results_max_mean/maximum/tables_DECODE_s"
elif evaluation_mode == "average":
    result_dir = "results_max_mean/csvresult_dir"
    wc_dir = "results_max_mean/wc_dir"
    if excluding_method[0] == "DECODE_s":
        figure_dir = "results_max_mean/average/figures_DECODE"
        table_dir = "results_max_mean/average/tables_DECODE"
    elif excluding_method[0] == "DECODE":
        figure_dir = "results_max_mean/average/figures_DECODE_s"
        table_dir = "results_max_mean/average/tables_DECODE_s"
elif evaluation_mode == "default":
    result_dir = "results_default/csvresult_dir"
    wc_dir = "results_default/wc_dir"
    if excluding_method[0] == "DECODE_s":
        figure_dir = "results_default/figures_DECODE"
        table_dir = "results_default/tables_DECODE"
    elif excluding_method[0] == "DECODE":
        figure_dir = "results_default/figures_DECODE_s"
        table_dir = "results_default/tables_DECODE_s"    
elif evaluation_mode == "hpt":
    result_dir = "results_hpt/csvresult_dir"
    wc_dir = "results_hpt/wc_dir"
    if excluding_method[0] == "DECODE_s":
        figure_dir = "results_hpt/figures_DECODE"
        table_dir = "results_hpt/tables_DECODE"
    elif excluding_method[0] == "DECODE":
        figure_dir = "results_hpt/figures_DECODE_s"
        table_dir = "results_hpt/tables_DECODE_s"    

if dataset_mode == "synthetic":
    result_dir = "results_synthetic_data/csvresult_dir"
    wc_dir = "results_synthetic_data/wc_dir"
elif dataset_mode == "damadics":
    result_dir = "results_damadics/csvresult_dir"
    wc_dir = "results_damadics/wc_dir"
else:
    result_dir = result_dir
    wc_dir = wc_dir
    
os.makedirs(table_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

if evaluation_mode == "default":
    method_blacklist = excluding_method
else:
    method_blacklist = ["DynamicHBOS"] + excluding_method

#TODO: What to do with the large_dataset_blacklist? Currently it is not in sync with the actual paper
large_dataset_blacklist = ["celeba", "backdoor", "fraud"]
double_dataset_blacklist = [] 
unsolvable_dataset_blacklist = ["hrss_anomalous_standard", "wpbc"]
# unsolvable_dataset_blacklist = []
dataset_blacklist = large_dataset_blacklist + unsolvable_dataset_blacklist + double_dataset_blacklist 

rename_datasets = {"hrss_anomalous_optimized":"hrss"}
# rename_datasets = {"hrss_anomalous_optimized":"hrss_optimized", "hrss_anomalous_standard":"hrss_standard"}

evaluation_metrics = ["ROC/AUC","R_precision", "adjusted_R_precision", "average_precision", "adjusted_average_precision"]
#%%
def score_to_rank(score_df): #for example score_to_rank(metric_dfs["ROC/AUC"])
    return(score_df.rank(ascending=False).transpose())

def friedman(rank_df):
    return(friedmanchisquare(*[rank_df[col] for col in rank_df.columns]))

def iman_davenport(rank_df): #could also return p-value, but would have to find F value table
    friedman_stat, _ = friedman(rank_df)
    
    N, k = rank_df.shape
    
    iman_davenport_stat = ((N-1)*friedman_stat)/(N*(k-1)-friedman_stat)
    return(iman_davenport_stat)

def iman_davenport_critical_value(rank_df):
    
    N, k = rank_df.shape
        
    return(scipy.stats.f.ppf(0.05, k-1, (k-1)*(N-1)))
        
    

#%%

#First find all datasets and methods used:
datasets = set(os.listdir(result_dir)) - set(dataset_blacklist)
    
methods_per_dataset = []

method_count_per_dataset = {}
max_methods = 0
for dataset in datasets:
    method_folders = os.listdir(os.path.join(result_dir, dataset))
    
    unique_datasets = set(method_folders)-set(method_blacklist)
    
    methods_per_dataset.append(unique_datasets)
    
    method_count_per_dataset[dataset] = len(unique_datasets)
    
    if method_count_per_dataset[dataset] > max_methods:
        max_methods = method_count_per_dataset[dataset]


if prune == "methods":
    methods = set.intersection(*methods_per_dataset)
    
    incomplete_methods = set([x for xs in methods_per_dataset for x in xs]).difference(methods)
    
    if len(incomplete_methods) > 0:
        print("The following methods were not calculated for each dataset:")
        print(incomplete_methods)
    
    methods = list(methods)
elif prune == "datasets":
    methods = set.union(*methods_per_dataset)
    
    datasets = [m  for m in method_count_per_dataset if method_count_per_dataset[m] == max_methods]
    
    incomplete_datasets = list(set(os.listdir(result_dir)) - set(dataset_blacklist) - set(datasets))
    
    if len(incomplete_datasets) > 0:
        print("The following datasets were not calculated for each method:")
        print(incomplete_datasets)



#%% Read all metrics from files

#contains the averaged results
metric_dfs = {}

#contains the full results of all hyperparameters
full_metric_dfs = {}

wallclock_dfs = {}
wallclock_metric = 'S'

for evaluation_metric in evaluation_metrics:
    # metric_dfs[evaluation_metric] = pd.DataFrame(index=methods,columns=datasets)
    # full_metric_dfs[evaluation_metric] = pd.DataFrame(index=methods,columns=datasets)
    metric_dfs[evaluation_metric] = pd.DataFrame(index=list(methods), columns=datasets)
    full_metric_dfs[evaluation_metric] = pd.DataFrame(index=list(methods), columns=datasets)
    wallclock_dfs[wallclock_metric] = pd.DataFrame(index=list(methods),columns=datasets)


for dataset_name in datasets:
    for method_name in methods:
        
            result_folder_path = os.path.join(result_dir, dataset_name, method_name)
            
            hyperparameter_csvs = os.listdir(result_folder_path)
            hyperparameter_settings = [filename.replace(".csv", "") for filename in hyperparameter_csvs]
            
            wc_folder_path = os.path.join(wc_dir, dataset_name, method_name)
            wc_csvs = os.listdir(wc_folder_path)
            wc_settings = [filename.replace(".csv", "") for filename in wc_csvs]

            results_per_setting = {}
            for hyperparameter_csv, hyperparameter_setting in zip(hyperparameter_csvs, hyperparameter_settings):
                
                full_path_filename = os.path.join(result_folder_path, hyperparameter_csv)
                
                #results_per_setting[hyperparameter_setting] = pickle.load(open(full_path_filename, 'rb'))
                results_per_setting[hyperparameter_setting] = pd.read_csv(full_path_filename)

            wc_per_setting = {}
            for wc_csv, wc_setting in zip(wc_csvs, wc_settings):
                
                full_path_filename = os.path.join(wc_folder_path, wc_csv)
                wc_per_setting[wc_setting] = pd.read_csv(full_path_filename)     

            for evaluation_metric in evaluation_metrics: 
                metric_per_setting = {setting:results[evaluation_metric].values[0] for setting, results in results_per_setting.items()}
                
                wc_per_setting = {
                    setting: float(str(results).rsplit(',', 1)[-1].split(']')[0].strip()) 
                    for setting, results in wc_per_setting.items()
                }
                average_time = np.mean(np.fromiter(wc_per_setting.values(), dtype=float))

                if evaluation_mode == "maximum":
                    average_metric = np.max(np.fromiter(metric_per_setting.values(), dtype=float))
                else:
                    average_metric = np.mean(np.fromiter(metric_per_setting.values(), dtype=float))

                metric_dfs[evaluation_metric][dataset_name][method_name] = average_metric
                full_metric_dfs[evaluation_metric][dataset_name][method_name] = metric_per_setting
                wallclock_dfs[wallclock_metric][dataset_name][method_name] = average_time
        
#%% optional: filter either datasets or methods for which not all methods are in:
    # Also filter blacklisted items.


        
for evaluation_metric in evaluation_metrics:
    #metric_dfs[evaluation_metric].drop(method_blacklist, axis=0, inplace=True, errors="ignore")
    #metric_dfs[evaluation_metric].drop(dataset_blacklist,axis=1,inplace=True, errors="ignore")
        
    if prune == "methods":
        metric_dfs[evaluation_metric].dropna(axis=0, inplace=True)#drop columns first, as datasets are processed in inner loop, methods in outer..
    elif prune == "datasets":
        metric_dfs[evaluation_metric].dropna(axis=1, inplace=True)#drop columns first, as datasets are processed in inner loop, methods in outer..
    #elif prune == "running":
        #running_dataset = metric_dfs[evaluation_metric].isna().sum().idxmax() 
        #metric_dfs[evaluation_metric].drop(running_dataset, axis=1, inplace=True)
        #metric_dfs[evaluation_metric].dropna(axis=0, inplace=True)#drop columns first, as datasets are processed in inner loop, methods in outer..
    metric_dfs[evaluation_metric].rename(columns=rename_datasets, inplace=True)


for wallclock_met in wallclock_metric:
    if prune == "methods":
        wallclock_dfs[wallclock_met].dropna(axis=0, inplace=True)
    elif prune == "datasets":
        wallclock_dfs[wallclock_met].dropna(axis=1, inplace=True)
    wallclock_dfs[wallclock_met].rename(columns=rename_datasets, inplace=True)

#%%
score_df_2 = metric_dfs["ROC/AUC"]
wallclock_df_2 = wallclock_dfs[wallclock_metric] 

visualization_mode = 0
# Scale data based on visualization mode
if visualization_mode == 0:
    scaled_df = (score_df_2 / score_df_2.max()) * 100
    scaled_wallclock_df = (wallclock_df_2 / wallclock_df_2.max())
    auc_label = 'Percentage of Maximum ROC/AUC (median)'
    time_label = 'Average Wall-Clock Time (normalized)'
else:
    scaled_df = score_df_2 
    scaled_wallclock_df = wallclock_df_2
    auc_label = 'ROC/AUC'
    time_label = 'Wall-Clock Time (seconds)'

# Melt DataFrames to long format
plot_df = scaled_df.melt(var_name="dataset", ignore_index=False).reset_index().rename(columns={"index": "method", "value": "auc"})
plot_wallclock_df = scaled_wallclock_df.melt(var_name="dataset", ignore_index=False).reset_index().rename(columns={"index": "method", "value": "time"})

# Merge AUC and wall-clock time data
combined_df = pd.merge(plot_df, plot_wallclock_df, on=['method', 'dataset'])

# Compute median AUC and average wall-clock time for each method
method_stats = combined_df.groupby('method').agg({
    'auc': 'median',
    'time': 'mean'
}).reset_index()
method_stats.columns = ['method', 'auc_median', 'time_avg']
# Pivot data for clustering
auc_pivot = combined_df.pivot(index='dataset', columns='method', values='auc').fillna(0)
time_pivot = combined_df.pivot(index='dataset', columns='method', values='time').fillna(0)

# Function to estimate optimal number of clusters using Elbow Method
def estimate_clusters(data, max_clusters=10):
    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    # Find the "elbow" by calculating the second derivative (acceleration)
    diffs = np.diff(inertias)
    diffs2 = np.diff(diffs)
    optimal_k = np.argmax(diffs2) + 2  # +2 because diff reduces length and we want k, not index
    return optimal_k, inertias

# Estimate clusters for AUC and Time
max_clusters = min(len(auc_pivot.T), 10)  # Limit by number of methods or 10
optimal_k_auc, auc_inertias = estimate_clusters(auc_pivot.T, max_clusters)
optimal_k_time, time_inertias = estimate_clusters(time_pivot.T, max_clusters)

# Perform clustering with optimal number of clusters
kmeans_auc = KMeans(n_clusters=optimal_k_auc, random_state=42)
method_stats['auc_cluster'] = kmeans_auc.fit_predict(auc_pivot.T)

kmeans_time = KMeans(n_clusters=optimal_k_time, random_state=42)
method_stats['time_cluster'] = kmeans_time.fit_predict(time_pivot.T)

# Define cluster colors (extendable for variable cluster numbers)
auc_colors = sns.color_palette("Pastel1", optimal_k_auc)
time_colors = sns.color_palette("Pastel2", optimal_k_time)

# Create the scatter plot
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor('white')  
ax.set_facecolor('white')  

ax.spines['top'].set_edgecolor('black')
ax.spines['bottom'].set_edgecolor('black')
ax.spines['left'].set_edgecolor('black')
ax.spines['right'].set_edgecolor('black')

ax.spines['top'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)

# Add background strips for AUC clusters (horizontal)
for cluster in range(optimal_k_auc):
    cluster_data = method_stats[method_stats['auc_cluster'] == cluster]['auc_median']
    if not cluster_data.empty:
        min_auc, max_auc = cluster_data.min(), cluster_data.max()
        ax.axhspan(min_auc, max_auc, facecolor=auc_colors[cluster], alpha=0.3, zorder=0)

# Add background strips for Time clusters (vertical)
for cluster in range(optimal_k_time):
    cluster_data = method_stats[method_stats['time_cluster'] == cluster]['time_avg']
    if not cluster_data.empty:
        min_time, max_time = cluster_data.min(), cluster_data.max()
        ax.axvspan(min_time, max_time, facecolor=time_colors[cluster], alpha=0.5, zorder=0)

# Plot scatter points
for idx, row in method_stats.iterrows():
    if row['method'] in ['DECODE_s', 'kth-NN']:
        print(row['method'], row['auc_median'], row['time_avg'])
    if row['method'] in ['DECODE', 'DECODE_s']:
        plt.scatter(row['time_avg'], row['auc_median'], color=sns.color_palette("husl", len(method_stats))[idx],
                    s=200, alpha=1.0, edgecolor='black', linewidth=2, zorder=2)
    else:
        plt.scatter(row['time_avg'], row['auc_median'], color=sns.color_palette("husl", len(method_stats))[idx], 
                    s=200, alpha=1.0, edgecolor='gray', linewidth=1, zorder=2)
        
    plt.text(row['time_avg'], row['auc_median'] - 1.3, row['method'], fontsize=14, ha='left', va='bottom', 
             color='black', transform=ax.transData, zorder=3)

# Customize the plot
plt.xlabel(time_label, fontsize=22)
plt.ylabel(auc_label, fontsize=22)
plt.tick_params(axis='both', labelsize=18)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.2))

plt.xscale('log')  # Uncomment if log scale is needed for time

# Add a rectangular box to the plot
rect = Rectangle((0.0001, 85), 0.01 - 0.0001, 100 - 85, linewidth=1, edgecolor='black', facecolor='none')
plt.gca().add_patch(rect)
rect = Rectangle((0.0001, 82.5), 0.11 - 0.0001, 100 - 82.5, linewidth=1, edgecolor='black', facecolor='none')
plt.gca().add_patch(rect)

# Title and layout
plt.tight_layout()

# Save the main plot
plt.savefig(f"{figure_dir}/median_auc_vs_avg_time_auto_clusters.eps", format="eps", bbox_inches="tight")
plt.savefig(f"{figure_dir}/median_auc_vs_avg_time_auto_clusters.png", format="png", bbox_inches="tight")
plt.savefig(f"{figure_dir}/median_auc_vs_avg_time_auto_clusters.pdf", format="pdf", bbox_inches="tight")
#%% see whether datasets are "solvable", and whether they might need to be inverted:
temp_df = metric_dfs["ROC/AUC"]

low_max_datasets= temp_df.columns[temp_df.max() < 0.6]

invertable_datasets = temp_df.columns[np.logical_and(temp_df.max() < 0.6, temp_df.min() < 0.4)]
#list minima:
print("invertable datasets:")
print(invertable_datasets)
print("minima:")
print(temp_df.min().loc[invertable_datasets])
print("maxima:")
print(temp_df.max().loc[invertable_datasets])

unsolvable_datasets = temp_df.columns[np.logical_and(temp_df.max() < 0.6, temp_df.min() >= 0.4)]

print("Unsolvable datasets:")
print(unsolvable_datasets)
print("minima:")
print(temp_df.min().loc[unsolvable_datasets])
print("maxima:")
print(temp_df.max().loc[unsolvable_datasets])
#%% calculate friedman  nemenyi and write to table
#TODO: Calculate Friedman using Tom's exact implementation

#https://stackoverflow.com/questions/6913532/display-a-decimal-in-scientific-notation
def format_e(n):
    a = '%E' % n
    return a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]


def p_value_to_string(p_value, n_decimals):
    if p_value < 1.0/(10**n_decimals):
        return "<" + format_e(1.0/(10**n_decimals))
    else:
        return str(round(p_value, n_decimals))

#def p_value_marker(val):


#    bold = 'bold' if float(val) < 0.05 else ''


#    return 'font-weight: %s' % bold
n_decimals = 3

score_df = metric_dfs["ROC/AUC"]
# n_columns_first_half = int(len(score_df.columns)/2)

# header = ["\\rot{"+column+"}" for column in score_df.columns[:n_columns_first_half]]
# table_file = open(f"{table_dir}/AUC_all_datasets_first_half.tex","w")

# # Create header only for columns that exist in score_df
# header = [f"\\rot{{{col}}}" for col in score_df.columns if col in header]

# # Ensure the correct number of headers
# if len(header) == len(score_df.columns):
#     # Now you can export the LaTeX table
#     score_df.iloc[:, :n_columns_first_half].astype(float).round(2).to_latex(table_file, header=header, escape=False)
# else:
#     print("Column/header mismatch, please check alignment.")

# print(score_df.columns)

# score_df.iloc[:,:n_columns_first_half].astype(float).round(2).to_latex(table_file, header=header, escape=False)
# table_file.close()

# header = ["\\rot{"+column+"}" for column in score_df.columns[n_columns_first_half:]]
# table_file = open(f"{table_dir}/AUC_all_datasets_second_half.tex","w")
# score_df.iloc[:,n_columns_first_half:].astype(float).round(2).to_latex(table_file, header=header, escape=False)
# table_file.close()


rank_df = score_to_rank(score_df)

friedman_score = friedman(rank_df)

print(friedman_score)

iman_davenport_score = iman_davenport(rank_df)

print("iman davenport score: " + str(iman_davenport_score))

print("Critical value: " + str(iman_davenport_critical_value(rank_df)))

nemenyi_table = posthoc_nemenyi_friedman(rank_df)
nemenyi_table_copy = nemenyi_table.copy(deep=True)
nemenyi_table_copy.columns = ["\\rot{"+column+"}" for column in nemenyi_table_copy.columns] 
nemenyi_formatted = nemenyi_table_copy.applymap(lambda x: p_value_to_string(x, n_decimals)).style.apply(lambda x: ["textbf:--rwrap" if float(v) < 0.05 else "" for v in x])

#table_file = open(f"{table_dir}/nemenyi_table_all_datasets.tex","w")
nemenyi_formatted.to_latex(f"{table_dir}/nemenyi_table_all_datasets.tex", hrules=True)
#table_file.close()

#%% Make table summarizing significance and performance results

p_value_threshold = 0.05

result_df = pd.DataFrame()

result_df["Mean Performance"] = score_df.transpose().mean()

result_df["Performance std"] = score_df.transpose().std()

result_df["Performance Range"] = (score_df.transpose().max() - score_df.transpose().min()).astype(float)

method_outperforms = []
for method in result_df.index:
    outperforming_methods = []
    for competing_method in result_df.index:
        if nemenyi_table[method][competing_method] < p_value_threshold and result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
            outperforming_methods.append(competing_method)
    method_outperforms.append(", ".join(outperforming_methods))

result_df["Outperforms"] = method_outperforms

result_df = result_df.sort_values(by="Mean Performance", ascending=False).round(4)

table_file = open(f"{table_dir}/significance_results_all_datasets.tex","w")
result_df.to_latex(table_file)
table_file.close()

#%% plot average percentage of maximum for all datasets

scaled_df = score_df/score_df.max()*100

reordered_index_all = score_df.transpose().mean().sort_values(ascending=False).index

palette = dict(zip(reordered_index_all, sns.color_palette("husl", n_colors=len(reordered_index_all))))

plot_df = (scaled_df).melt(var_name="dataset", ignore_index=False).reset_index().rename(columns={"index":"method"})
plt.figure()
ax = sns.boxplot(x="method",y="value",data=plot_df, order=reordered_index_all, palette=palette)
labels = ax.get_xticklabels()
for label in labels:
    if label.get_text() == "DECODE" or label.get_text() == "DECODE_s":
        label.set_fontweight('bold') 
        label.set_fontsize(12)       
ax.set_xticklabels(labels)
ax.set_title("Percentage of maximum performance (ROC/AUC)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{figure_dir}/ROCAUC_boxplot_all_datasets.eps",format="eps")
plt.savefig(f"{figure_dir}/ROCAUC_boxplot_all_datasets.png",format="png")
plt.savefig(f"{figure_dir}/ROCAUC_boxplot_all_datasets.pdf",format="pdf")
plt.show()

# plt.figure()
# palette = dict(zip(reordered_index_all, sns.color_palette("husl", n_colors=len(reordered_index_all))))
# ax = sns.violinplot(x="method", y="value", data=plot_df, order=reordered_index_all, palette=palette, inner=None)
# sns.stripplot(x="method", y="value", data=plot_df, order=reordered_index_all, color="black", size=2, alpha=0.35, ax=ax)
# labels = ax.get_xticklabels()
# for label in labels:
#     if label.get_text() == "DECODE" or label.get_text() == "DECODE_s":
#         label.set_fontweight('bold')
#         label.set_fontsize(12)
# ax.set_xticklabels(labels)
# ax.set_title("Percentage of maximum performance (ROC/AUC)")
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig(f"{figure_dir}/ROCAUC_violin_all_datasets.eps", format="eps")
# plt.savefig(f"{figure_dir}/ROCAUC_violin_all_datasets.png", format="png")
# plt.savefig(f"{figure_dir}/ROCAUC_violin_all_datasets.pdf", format="pdf")
# plt.show()

plt.figure()
palette = dict(zip(reordered_index_all, sns.color_palette("husl", n_colors=len(reordered_index_all))))
ax = sns.violinplot(x="method", y="value", data=plot_df, order=reordered_index_all, palette=palette, inner=None)
sns.stripplot(x="method", y="value", data=plot_df, order=reordered_index_all, color="black", size=2, alpha=0.35, ax=ax)
counts = plot_df[plot_df['value'] == 100]['method'].value_counts().reindex(reordered_index_all, fill_value=0)
for i, method in enumerate(reordered_index_all):
    count = counts[method]
    ax.text(i, 115, f'{count}', ha='right', va='bottom', fontsize=10, color='blue')

labels = ax.get_xticklabels()
for label in labels:
    if label.get_text() == "DECODE" or label.get_text() == "DECODE_s":
        label.set_fontweight('bold')
        label.set_fontsize(12)
ax.set_xticklabels(labels)

ax.set_title("Percentage of maximum performance (ROC/AUC)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{figure_dir}/ROCAUC_violin_all_datasets.eps", format="eps")
plt.savefig(f"{figure_dir}/ROCAUC_violin_all_datasets.png", format="png")
plt.savefig(f"{figure_dir}/ROCAUC_violin_all_datasets.pdf", format="pdf")
plt.show()
#%% clustermap
#Do clustering on percentage of performance, rather than straight AUC

plot_df = metric_dfs["ROC/AUC"].astype(float)

clustermap = sns.clustermap(plot_df.transpose().iloc[:,:], method="average",metric="correlation", figsize=(15,15))
clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(), fontsize=18)
clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.get_yticklabels(), fontsize=18)
clustermap.savefig(f"{figure_dir}/clustermap_all_datasets.eps",format="eps", dpi=1000)
clustermap.savefig(f"{figure_dir}/clustermap_all_datasets.png",format="png")
clustermap.savefig(f"{figure_dir}/clustermap_all_datasets.pdf",format="pdf")
plt.show()

#%% Make heatmap/table showing significance results at p < 0.05, p < 0.10, p>=0.10
#import matplotlib as mpl

# cmap = sns.color_palette("flare")
# cmap = mpl.cm.viridis
# cmap = mpl.colors.ListedColormap(sns.color_palette("flare").as_hex())
# cmap = mpl.colors.ListedColormap([[1,1,1], [0.4,0,0.4], [0,0,1]]).reversed()
# bounds = [0, 0.05, 0.10, 1]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')

# sns.heatmap(nemenyi_table[reordered_index_global].loc[reordered_index_global], cmap = cmap, norm=norm, cbar_kws={"label":"p-value"})
# plt.show()

significance_table = nemenyi_table.astype(str)

for method in nemenyi_table.columns:
    for competing_method in nemenyi_table.columns:
        if nemenyi_table[method].loc[competing_method] < 0.10:
            if nemenyi_table[method].loc[competing_method] < 0.05:
                if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
                    significance_table.loc[method,competing_method] = "++"
                else:
                    significance_table.loc[method,competing_method] = "-{}-"
            else:
                if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
                    significance_table.loc[method,competing_method] = "+"
                else:
                    significance_table.loc[method,competing_method] = "-"
        else:
            significance_table.loc[method,competing_method] = ""
            
# significance_table = nemenyi_table.astype(str)

# for method in nemenyi_table.columns:
#     for competing_method in nemenyi_table.columns:
#         if nemenyi_table[method].loc[competing_method] <= 0.10:
#             if nemenyi_table[method].loc[competing_method] < 0.01:
#                 if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
#                     significance_table.loc[method,competing_method] = "+++"
#                 else:
#                     significance_table.loc[method,competing_method] = "-{}-{}-"
#             elif nemenyi_table[method].loc[competing_method] < 0.05:
#                 if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
#                     significance_table.loc[method,competing_method] = "++"
#                 else:
#                     significance_table.loc[method,competing_method] = "-{}-"
#             else:
#                 if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
#                     significance_table.loc[method,competing_method] = "+"
#                 else:
#                     significance_table.loc[method,competing_method] = "-"
#         else:
#             significance_table.loc[method,competing_method] = ""
            
   

significance_table = significance_table[reversed(reordered_index_all)].loc[reordered_index_all]
significance_table["Mean AUC"] = result_df["Mean Performance"].map(lambda x: f"{x:.4f}")
significance_table.index = significance_table.index.map(lambda x: x.replace("_", "\\_"))
significance_table.columns = significance_table.columns.map(lambda x: x.replace("_", "\\_"))

significance_table.columns = significance_table.columns.map(lambda x: "\\rotatebox{90}{"+x+"}")

significance_table.columns = significance_table.columns.map(lambda x: x.replace("Mean AUC", "\\textbf{Mean AUC}"))
table_file = open(f"{table_dir}/nemenyi_summary.tex","w")
significance_table.to_latex(table_file)
table_file.close()

# significance_table_truncated = significance_table.loc[:, (significance_table == "++").any() | (significance_table == "+").any()]
# significance_table_truncated["Mean Performance"] = score_df.transpose().mean().astype(float).sort_values(ascending=False).round(3)
# significance_table_truncated["Mean Performance"] = score_df.transpose().mean().sort_values(ascending=False).round(3)
# table_file = open(f"{table_dir}/nemenyi_summary_truncated.tex","w")
# column_format = "l" + "c"*(len(significance_table_truncated.columns)-1) +"|r"
# header = ["\\rot{"+column+"}" for column in significance_table_truncated.columns[:-1]] + ["\\rot{\\shortstack[l]{\\textbf{Mean}\\\\\\textbf{AUC}}}"]
# significance_table_truncated.to_latex(table_file, column_format=column_format, header=header, escape=False)
# table_file.close()


if dataset_mode != "benchmark":
    sys.exit()  # Terminate the program
#%% Redo nemenyi test and pairwise testing based on the clustering

#%% Local datasets

local_datasets = ["skin", "ionosphere", "glass", "landsat", "fault", "vowels", "pen-local", "letter", "wilt", "nasa", "parkinson", "waveform", "magic.gamma", "pima", "internetads", "speech", "aloi"]#["parkinson", "wilt", "aloi", "vowels", "letter", "pen-local", "glass", "ionosphere", "nasa", "fault", "landsat", "donors"]

#check if all local datasets have been calculated/are not in blacklist:
local_datasets = [dataset for dataset in local_datasets if dataset in metric_dfs["ROC/AUC"].columns]

score_df = metric_dfs["ROC/AUC"][local_datasets]

rank_df = score_to_rank(score_df)

friedman_score = friedman(rank_df)

print("local:")
print(friedman_score)

iman_davenport_score = iman_davenport(rank_df)

print ("iman davenport score local: " + str(iman_davenport_score))
print("Critical value: " + str(iman_davenport_critical_value(rank_df)))

nemenyi_table = posthoc_nemenyi_friedman(rank_df)
nemenyi_table_copy = nemenyi_table.copy(deep=True)
nemenyi_table_copy.columns = ["\\rot{"+column+"}" for column in nemenyi_table_copy.columns] 
nemenyi_formatted = nemenyi_table_copy.applymap(lambda x: p_value_to_string(x, n_decimals)).style.apply(lambda x: ["textbf:--rwrap" if float(v) < 0.05 else "" for v in x])

#table_file = open(f"{table_dir}/nemenyi_table_local.tex","w")
nemenyi_formatted.to_latex(f"{table_dir}/nemenyi_table_local.tex", hrules=True)
#table_file.close()

#%% Make table summarizing significance and performance results for local datasets

p_value_threshold = 0.05

result_df = pd.DataFrame()

result_df["Mean Performance"] = score_df.transpose().mean()

result_df["Performance std"] = score_df.transpose().std()

result_df["Performance Range"] = (score_df.transpose().max() - score_df.transpose().min()).astype(float)

method_outperforms = []
for method in result_df.index:
    outperforming_methods = []
    for competing_method in result_df.index:
        if nemenyi_table[method][competing_method] < p_value_threshold and result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
            outperforming_methods.append(competing_method)
    method_outperforms.append(", ".join(outperforming_methods))

result_df["Outperforms"] = method_outperforms

result_df = result_df.sort_values(by="Mean Performance", ascending=False).round(4)

table_file = open(f"{table_dir}/significance_results_local.tex","w")
result_df.to_latex(table_file)
table_file.close()

#%% Make boxplot for local datasets
scaled_df = score_df/score_df.max()*100

reordered_index_local = score_df.transpose().mean().sort_values(ascending=False).index



plot_df = (scaled_df).melt(var_name="dataset", ignore_index=False).reset_index().rename(columns={"index":"method"})
plt.figure()
ax = sns.boxplot(x="method",y="value",data=plot_df, order=reordered_index_local, palette=palette)
labels = ax.get_xticklabels()
for label in labels:
    if label.get_text() == "DECODE" or label.get_text() == "DECODE_s":
        label.set_fontweight('bold')  
        label.set_fontsize(12)        
ax.set_xticklabels(labels)
ax.set_title("Percentage of maximum performance (ROC/AUC)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{figure_dir}/ROCAUC_boxplot_local_datasets.eps",format="eps")
plt.savefig(f"{figure_dir}/ROCAUC_boxplot_local_datasets.png",format="png")
plt.savefig(f"{figure_dir}/ROCAUC_boxplot_local_datasets.pdf",format="pdf")
plt.show()

# plt.figure()
# palette = dict(zip(reordered_index_local, sns.color_palette("husl", n_colors=len(reordered_index_local))))
# ax = sns.violinplot(x="method", y="value", data=plot_df, order=reordered_index_local, palette=palette, inner=None)
# sns.stripplot(x="method", y="value", data=plot_df, order=reordered_index_local, color="black", size=2, alpha=0.35, ax=ax)
# labels = ax.get_xticklabels()
# for label in labels:
#     if label.get_text() == "DECODE" or label.get_text() == "DECODE_s":
#         label.set_fontweight('bold')
#         label.set_fontsize(12)
# ax.set_xticklabels(labels)
# ax.set_title("Percentage of maximum performance (ROC/AUC)")
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig(f"{figure_dir}/ROCAUC_violin_local_datasets.eps", format="eps")
# plt.savefig(f"{figure_dir}/ROCAUC_violin_local_datasets.png", format="png")
# plt.savefig(f"{figure_dir}/ROCAUC_violin_local_datasets.pdf", format="pdf")
# plt.show()

plt.figure()
palette = dict(zip(reordered_index_local, sns.color_palette("husl", n_colors=len(reordered_index_local))))
ax = sns.violinplot(x="method", y="value", data=plot_df, order=reordered_index_local, palette=palette, inner=None)
sns.stripplot(x="method", y="value", data=plot_df, order=reordered_index_local, color="black", size=2, alpha=0.35, ax=ax)
counts = plot_df[plot_df['value'] == 100]['method'].value_counts().reindex(reordered_index_local, fill_value=0)
for i, method in enumerate(reordered_index_local):
    count = counts[method]
    ax.text(i, 115, f'{count}', ha='right', va='bottom', fontsize=10, color='blue')

labels = ax.get_xticklabels()
for label in labels:
    if label.get_text() == "DECODE" or label.get_text() == "DECODE_s":
        label.set_fontweight('bold')
        label.set_fontsize(12)
ax.set_xticklabels(labels)

ax.set_title("Percentage of maximum performance (ROC/AUC)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{figure_dir}/ROCAUC_violin_local_datasets.eps", format="eps")
plt.savefig(f"{figure_dir}/ROCAUC_violin_local_datasets.png", format="png")
plt.savefig(f"{figure_dir}/ROCAUC_violin_local_datasets.pdf", format="pdf")
plt.show()
#%%
plot_df = metric_dfs["ROC/AUC"][local_datasets].astype(float)

clustermap = sns.clustermap(plot_df.transpose().iloc[:,:], method="average",metric="correlation", figsize=(15,15))
clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(), fontsize=18)
clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.get_yticklabels(), fontsize=18)
clustermap.savefig(f"{figure_dir}/clustermap_local_datasets.eps",format="eps", dpi=1000)
clustermap.savefig(f"{figure_dir}/clustermap_local_datasets.png",format="png")
clustermap.savefig(f"{figure_dir}/clustermap_local_datasets.pdf",format="pdf")
plt.show()
#%% Make heatmap/table showing significance results at p < 0.05, p < 0.10, p>=0.10
#import matplotlib as mpl

# cmap = sns.color_palette("flare")
# cmap = mpl.cm.viridis
# cmap = mpl.colors.ListedColormap(sns.color_palette("flare").as_hex())
# cmap = mpl.colors.ListedColormap([[1,1,1], [0.4,0,0.4], [0,0,1]]).reversed()
# bounds = [0, 0.05, 0.10, 1]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')

# sns.heatmap(nemenyi_table[reordered_index_global].loc[reordered_index_global], cmap = cmap, norm=norm, cbar_kws={"label":"p-value"})
# plt.show()

significance_table = nemenyi_table.astype(str)

for method in nemenyi_table.columns:
    for competing_method in nemenyi_table.columns:
        if nemenyi_table[method].loc[competing_method] < 0.10:
            if nemenyi_table[method].loc[competing_method] < 0.05:
                if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
                    significance_table.loc[method,competing_method] = "++"
                else:
                    significance_table.loc[method,competing_method] = "-{}-"
            else:
                if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
                    significance_table.loc[method,competing_method] = "+"
                else:
                    significance_table.loc[method,competing_method] = "-"
        else:
            significance_table.loc[method,competing_method] = ""
            

# for method in nemenyi_table.columns:
#     for competing_method in nemenyi_table.columns:
#         if nemenyi_table[method].loc[competing_method] <= 0.10:
#             if nemenyi_table[method].loc[competing_method] < 0.01:
#                 if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
#                     significance_table.loc[method,competing_method] = "+++"
#                 else:
#                     significance_table.loc[method,competing_method] = "-{}-{}-"
#             elif nemenyi_table[method].loc[competing_method] < 0.05:
#                 if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
#                     significance_table.loc[method,competing_method] = "++"
#                 else:
#                     significance_table.loc[method,competing_method] = "-{}-"
#             else:
#                 if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
#                     significance_table.loc[method,competing_method] = "+"
#                 else:
#                     significance_table.loc[method,competing_method] = "-"
#         else:
#             significance_table.loc[method,competing_method] = ""
            
   

significance_table = significance_table[reversed(reordered_index_local)].loc[reordered_index_local]
significance_table["Mean AUC"] = result_df["Mean Performance"].map(lambda x: f"{x:.4f}")
significance_table.index = significance_table.index.map(lambda x: x.replace("_", "\\_"))
significance_table.columns = significance_table.columns.map(lambda x: x.replace("_", "\\_"))

significance_table.columns = significance_table.columns.map(lambda x: "\\rotatebox{90}{"+x+"}")

significance_table.columns = significance_table.columns.map(lambda x: x.replace("Mean AUC", "\\textbf{Mean AUC}"))
table_file = open(f"{table_dir}/nemenyi_summary_local.tex","w")
significance_table.to_latex(table_file)
table_file.close()

significance_table_truncated = significance_table.loc[:, (significance_table == "++").any() | (significance_table == "+").any()]
# significance_table_truncated["Mean Performance"] = score_df.transpose().mean().sort_values(ascending=False).round(3)
# table_file = open(f"{table_dir}/nemenyi_summary_local_truncated.tex","w")
# column_format = "l" + "c"*(len(significance_table_truncated.columns)-1) +"|r"
# header = ["\\rot{"+column+"}" for column in significance_table_truncated.columns[:-1]] + ["\\rot{\\shortstack[l]{\\textbf{Mean}\\\\\\textbf{AUC}}}"]
# significance_table_truncated.to_latex(table_file, column_format=column_format, header=header, escape=False)
# table_file.close()


#%% Global datasets
non_cluster_datasets = ["vertebral"]
score_df = metric_dfs["ROC/AUC"]
global_datasets = score_df.columns.difference(local_datasets+non_cluster_datasets)
score_df = score_df[global_datasets]

rank_df = score_to_rank(score_df)

friedman_score = friedman(rank_df)

print("global:")
print(friedman_score)

iman_davenport_score = iman_davenport(rank_df)

print ("iman davenport score global: " + str(iman_davenport_score))
print("Critical value: " + str(iman_davenport_critical_value(rank_df)))

nemenyi_table = posthoc_nemenyi_friedman(rank_df)
nemenyi_table_copy = nemenyi_table.copy(deep=True)
nemenyi_table_copy.columns = ["\\rot{"+column+"}" for column in nemenyi_table_copy.columns] 
nemenyi_formatted = nemenyi_table_copy.applymap(lambda x: p_value_to_string(x, n_decimals)).style.apply(lambda x: ["textbf:--rwrap" if float(v) < 0.05 else "" for v in x])

#table_file = open(f"{table_dir}/nemenyi_table_global.tex","w")
nemenyi_formatted.to_latex(f"{table_dir}/nemenyi_table_global.tex", hrules=True)
#table_file.close()




#%% Make table summarizing significance and performance results for global datasets

p_value_threshold = 0.05

result_df = pd.DataFrame()

result_df["Mean Performance"] = score_df.transpose().mean()

result_df["Performance std"] = score_df.transpose().std()

result_df["Performance Range"] = (score_df.transpose().max() - score_df.transpose().min()).astype(float)

method_outperforms = []
for method in result_df.index:
    outperforming_methods = []
    for competing_method in result_df.index:
        if nemenyi_table[method][competing_method] < p_value_threshold and result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
            outperforming_methods.append(competing_method)
    method_outperforms.append(", ".join(outperforming_methods))

result_df["Outperforms"] = method_outperforms

result_df = result_df.sort_values(by="Mean Performance", ascending=False).round(4)

table_file = open(f"{table_dir}/significance_results_global.tex","w")
result_df.to_latex(table_file)
table_file.close()

#%% Make boxplot for global datasets
scaled_df = score_df/score_df.max()*100

reordered_index_global = score_df.transpose().mean().sort_values(ascending=False).index

#scaled_df = scaled_df.loc[reordered_index]

plot_df = (scaled_df).melt(var_name="dataset", ignore_index=False).reset_index().rename(columns={"index":"method"})
plt.figure()
ax = sns.boxplot(x="method",y="value",data=plot_df, order=reordered_index_global, palette=palette)
ax.set_title("Percentage of maximum performance (ROC/AUC)")
labels = ax.get_xticklabels()
for label in labels:
    if label.get_text() == "DECODE" or label.get_text() == "DECODE_s":
        label.set_fontweight('bold')  
        label.set_fontsize(12)       
ax.set_xticklabels(labels)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{figure_dir}/ROCAUC_boxplot_global_datasets.eps",format="eps")
plt.savefig(f"{figure_dir}/ROCAUC_boxplot_global_datasets.png",format="png")
plt.savefig(f"{figure_dir}/ROCAUC_boxplot_global_datasets.pdf",format="pdf")
plt.show()

# plt.figure()
# palette = dict(zip(reordered_index_global, sns.color_palette("husl", n_colors=len(reordered_index_global))))
# ax = sns.violinplot(x="method", y="value", data=plot_df, order=reordered_index_global, palette=palette, inner=None)
# sns.stripplot(x="method", y="value", data=plot_df, order=reordered_index_global, color="black", size=2, alpha=0.35, ax=ax)
# labels = ax.get_xticklabels()
# for label in labels:
#     if label.get_text() == "DECODE" or label.get_text() == "DECODE_s":
#         label.set_fontweight('bold')
#         label.set_fontsize(12)
# ax.set_xticklabels(labels)
# ax.set_title("Percentage of maximum performance (ROC/AUC)")
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig(f"{figure_dir}/ROCAUC_violin_global_datasets.eps", format="eps")
# plt.savefig(f"{figure_dir}/ROCAUC_violin_global_datasets.png", format="png")
# plt.savefig(f"{figure_dir}/ROCAUC_violin_global_datasets.pdf", format="pdf")
# plt.show()

plt.figure()
palette = dict(zip(reordered_index_global, sns.color_palette("husl", n_colors=len(reordered_index_global))))
ax = sns.violinplot(x="method", y="value", data=plot_df, order=reordered_index_global, palette=palette, inner=None)
sns.stripplot(x="method", y="value", data=plot_df, order=reordered_index_global, color="black", size=2, alpha=0.35, ax=ax)
counts = plot_df[plot_df['value'] == 100]['method'].value_counts().reindex(reordered_index_global, fill_value=0)
for i, method in enumerate(reordered_index_global):
    count = counts[method]
    ax.text(i, 115, f'{count}', ha='right', va='bottom', fontsize=10, color='blue')

labels = ax.get_xticklabels()
for label in labels:
    if label.get_text() == "DECODE" or label.get_text() == "DECODE_s":
        label.set_fontweight('bold')
        label.set_fontsize(12)
ax.set_xticklabels(labels)

ax.set_title("Percentage of maximum performance (ROC/AUC)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{figure_dir}/ROCAUC_violin_global_datasets.eps", format="eps")
plt.savefig(f"{figure_dir}/ROCAUC_violin_global_datasets.png", format="png")
plt.savefig(f"{figure_dir}/ROCAUC_violin_global_datasets.pdf", format="pdf")
plt.show()
#%%
plot_df = metric_dfs["ROC/AUC"].drop(columns=local_datasets + non_cluster_datasets).astype(float)

clustermap = sns.clustermap(plot_df.transpose().iloc[:,:], method="average",metric="correlation", figsize=(15,15))
clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(), fontsize=22)
clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.get_yticklabels(), fontsize=18)
clustermap.savefig(f"{figure_dir}/clustermap_global_datasets.eps", format="eps", dpi=1000)
clustermap.savefig(f"{figure_dir}/clustermap_global_datasets.png",format="png")
clustermap.savefig(f"{figure_dir}/clustermap_global_datasets.pdf",format="pdf")
plt.show()

#%% Make heatmap/table showing significance results at p < 0.05, p < 0.10, p>=0.10
#import matplotlib as mpl

# cmap = sns.color_palette("flare")
# cmap = mpl.cm.viridis
# cmap = mpl.colors.ListedColormap(sns.color_palette("flare").as_hex())
# cmap = mpl.colors.ListedColormap([[1,1,1], [0.4,0,0.4], [0,0,1]]).reversed()
# bounds = [0, 0.05, 0.10, 1]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='neither')

# sns.heatmap(nemenyi_table[reordered_index_global].loc[reordered_index_global], cmap = cmap, norm=norm, cbar_kws={"label":"p-value"})
# plt.show()

significance_table = nemenyi_table.astype(str)

for method in nemenyi_table.columns:
    for competing_method in nemenyi_table.columns:
        if nemenyi_table[method].loc[competing_method] < 0.10:
            if nemenyi_table[method].loc[competing_method] < 0.05:
                if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
                    significance_table.loc[method,competing_method] = "++"
                else:
                    significance_table.loc[method,competing_method] = "-{}-"
            else:
                if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
                    significance_table.loc[method,competing_method] = "+"
                else:
                    significance_table.loc[method,competing_method] = "-"
        else:
            significance_table.loc[method,competing_method] = ""

# for method in nemenyi_table.columns:
#     for competing_method in nemenyi_table.columns:
#         if nemenyi_table[method].loc[competing_method] <= 0.10:
#             if nemenyi_table[method].loc[competing_method] < 0.01:
#                 if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
#                     significance_table.loc[method,competing_method] = "+++"
#                 else:
#                     significance_table.loc[method,competing_method] = "-{}-{}-"
#             elif nemenyi_table[method].loc[competing_method] < 0.05:
#                 if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
#                     significance_table.loc[method,competing_method] = "++"
#                 else:
#                     significance_table.loc[method,competing_method] = "-{}-"
#             else:
#                 if result_df["Mean Performance"][method] > result_df["Mean Performance"][competing_method]:
#                     significance_table.loc[method,competing_method] = "+"
#                 else:
#                     significance_table.loc[method,competing_method] = "-"
#         else:
#             significance_table.loc[method,competing_method] = ""
            
               

significance_table = significance_table[reversed(reordered_index_global)].loc[reordered_index_global]
significance_table["Mean AUC"] = result_df["Mean Performance"].map(lambda x: f"{x:.4f}")
significance_table.index = significance_table.index.map(lambda x: x.replace("_", "\\_"))
significance_table.columns = significance_table.columns.map(lambda x: x.replace("_", "\\_"))

significance_table.columns = significance_table.columns.map(lambda x: "\\rotatebox{90}{"+x+"}")

significance_table.columns = significance_table.columns.map(lambda x: x.replace("Mean AUC", "\\textbf{Mean AUC}"))
table_file = open(f"{table_dir}/nemenyi_summary_global.tex","w")
significance_table.to_latex(table_file)
table_file.close()

# significance_table_truncated = significance_table.loc[:, (significance_table == "++").any() | (significance_table == "+").any()]
# significance_table_truncated["Mean Performance"] = score_df.transpose().mean().sort_values(ascending=False).round(3)
# table_file = open(f"{table_dir}/nemenyi_summary_global_truncated.tex","w")
# column_format = "l" + "c"*(len(significance_table_truncated.columns)-1) +"|r"
# header = ["\\rot{"+column+"}" for column in significance_table_truncated.columns[:-1]] + ["\\rot{\\shortstack[l]{\\textbf{Mean}\\\\\\textbf{AUC}}}"]
# significance_table_truncated.to_latex(table_file, column_format=column_format, header=header, escape=False)
# table_file.close()