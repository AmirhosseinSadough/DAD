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

import networkx
from matplotlib.backends.backend_agg import FigureCanvasAgg

import argparse

arg_parser = argparse.ArgumentParser(description='produce figures and tables for the results of the outlier detection methods')

# Add the arguments
arg_parser.add_argument('--base_result_dir',
                        dest='base_result_dir',
                        default="results/benchmark/default",
                        type=str,
                        help='Path to the result directory.')

arg_parser.add_argument('--eval_mode',
                        dest='eval_mode',
                        default="hpt",
                        type=str,
                        help='Evaluation modes:"maximum", "average", "default", "hpt".')

arg_parser.add_argument('--dataset',
                        dest='dataset',
                        default="benchmark",
                        type=str,
                        help='Dataset mode: benchmark, synthetic, damadics.')

arg_parser.add_argument('--exclude_methods',
                        dest='exclude_methods',
                        default="",
                        type=str,
                        help='Comma-separated list of methods to exclude from analysis.')

arg_parser.add_argument('--exclude_datasets',
                        dest='exclude_datasets',
                        default="",
                        type=str,
                        help='Comma-separated list of datasets to exclude from analysis.')

# Execute the parse_args() method
parsed_args = arg_parser.parse_args()


result_dir = os.path.join(parsed_args.base_result_dir, "csvresult_dir")
evaluation_mode = parsed_args.eval_mode 
dataset_mode = parsed_args.dataset

wc_dir = os.path.join(parsed_args.base_result_dir, "wc_dir")
figure_dir = os.path.join(parsed_args.base_result_dir, f"figures_{evaluation_mode}")
table_dir = os.path.join(parsed_args.base_result_dir, f"tables_{evaluation_mode}")


prune = "datasets"        
excluding_method = parsed_args.exclude_methods.split(',') if parsed_args.exclude_methods else []

os.makedirs(table_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

if evaluation_mode == "default":
    method_blacklist = excluding_method
else:
    # method_blacklist = ["DynamicHBOS"] + excluding_method
    method_blacklist = excluding_method

#TODO: What to do with the large_dataset_blacklist? Currently it is not in sync with the actual paper
large_dataset_blacklist = ["celeba", "backdoor", "fraud"]
double_dataset_blacklist = [] 
unsolvable_dataset_blacklist = ["hrss_anomalous_standard", "wpbc"]
# unsolvable_dataset_blacklist = []
dataset_blacklist = large_dataset_blacklist + unsolvable_dataset_blacklist + double_dataset_blacklist 

excluding_dataset = parsed_args.exclude_datasets.split(',') if parsed_args.exclude_datasets else []
dataset_blacklist = dataset_blacklist + excluding_dataset

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


# check in metric_dfs and change name kNN to $k$NN and kth-NN to $k$th-NN
for evaluation_metric in evaluation_metrics:
    if "kNN" in metric_dfs[evaluation_metric].index:
        metric_dfs[evaluation_metric].rename(index={"kNN":"$k$-NN"}, inplace=True)
    if "kth-NN" in metric_dfs[evaluation_metric].index:
        metric_dfs[evaluation_metric].rename(index={"kth-NN":"$k$th-NN"}, inplace=True)
    if "ensemble-LOF" in metric_dfs[evaluation_metric].index:
        metric_dfs[evaluation_metric].rename(index={"ensemble-LOF":"ELOF"}, inplace=True)
    if "DynamicHBOS" in metric_dfs[evaluation_metric].index:
        metric_dfs[evaluation_metric].rename(index={"DynamicHBOS":"DHBOS"}, inplace=True)   

for wallclock_met in wallclock_metric:
    if "kNN" in wallclock_dfs[wallclock_met].index:
        wallclock_dfs[wallclock_met].rename(index={"kNN":"$k$-NN"}, inplace=True)
    if "kth-NN" in wallclock_dfs[wallclock_met].index:
        wallclock_dfs[wallclock_met].rename(index={"kth-NN":"$k$th-NN"}, inplace=True)
    if "ensemble-LOF" in wallclock_dfs[wallclock_met].index:
        wallclock_dfs[wallclock_met].rename(index={"ensemble-LOF":"ELOF"}, inplace=True)
    if "DynamicHBOS" in wallclock_dfs[wallclock_met].index:
        wallclock_dfs[wallclock_met].rename(index={"DynamicHBOS":"DHBOS"}, inplace=True)



df = metric_dfs["ROC/AUC"]

if "DADS" in df.index and "DAD" in df.index:
    ordered_methods = ["DADS", "DAD"] + [m for m in df.index if m not in ["DADS", "DAD"]]
elif "DAD_Auto" in df.index:
    ordered_methods = ["DAD_Auto"] + [m for m in df.index if m not in ["DAD_Auto"]]

df = df.loc[ordered_methods]

df = df[sorted(df.columns)]

# Split columns into two halves
half = len(df.columns) // 2
df1 = df.iloc[:, :half]
df2 = df.iloc[:, half:]

def truncate_to_2(x):
    return np.floor(x * 100) / 100

# Step 6: Function to bold max after truncation
def format_bold_max_truncate(df):
    formatted_df = df.copy()
    for col in df.columns:
        truncated_col = df[col].apply(truncate_to_2)
        max_trunc = truncated_col.max()
        formatted_df[col] = df[col].apply(
            lambda x: (
                f"\\textbf{{{truncate_to_2(x):.2f}}}"
                if pd.notna(x) and truncate_to_2(x) == max_trunc
                else f"{truncate_to_2(x):.2f}"
            )
        )
    return formatted_df.astype(str)

# Apply formatting
df1_fmt = format_bold_max_truncate(df1)
df2_fmt = format_bold_max_truncate(df2)


# Convert to LaTeX
latex_table1 = df1_fmt.to_latex(
    index=True,
    escape=False,
    column_format="l" + "r" * len(df1.columns)
)

latex_table2 = df2_fmt.to_latex(
    index=True,
    escape=False,
    column_format="l" + "r" * len(df2.columns)
)


with open(f"{table_dir}/AUC_method_dataset_part1.tex", "w") as f:
    f.write(latex_table1)

with open(f"{table_dir}/AUC_method_dataset_part2.tex", "w") as f:
    f.write(latex_table2)



#%%
score_df_2 = metric_dfs["ROC/AUC"]
wallclock_df_2 = wallclock_dfs[wallclock_metric] 

visualization_mode = 0
if visualization_mode == 0:
    scaled_df = (score_df_2 / score_df_2.max()) * 100
    scaled_wallclock_df = (wallclock_df_2 / wallclock_df_2.max())
    auc_label = 'AUC performance (median)'
    time_label = 'Average wall-clock time (normalized)'
else:
    scaled_df = score_df_2 
    scaled_wallclock_df = wallclock_df_2
    auc_label = 'AUC'
    time_label = 'wall-clock time (seconds)'

# Melt DataFrames to long format
plot_df = scaled_df.melt(var_name="dataset", ignore_index=False).reset_index().rename(columns={"index": "method", "value": "auc"})
plot_wallclock_df = scaled_wallclock_df.melt(var_name="dataset", ignore_index=False).reset_index().rename(columns={"index": "method", "value": "time"})

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
fig, ax = plt.subplots(figsize=(14, 10))
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

for idx, row in method_stats.iterrows():
    print(f"Method: {row['method']}, Median AUC: {row['auc_median']}, Average Time: {row['time_avg']}")



# Plot scatter points
for idx, row in method_stats.iterrows():
    # if row['method'] in ['DADS', '$k$th-NN']:
    #     print(row['method'], row['auc_median'], row['time_avg'])
    if row['method'] in ['DAD', 'DADS']:
        plt.scatter(row['time_avg'], row['auc_median'], color=sns.color_palette("husl", len(method_stats))[idx],
                    s=350, alpha=1.0, edgecolor='black', linewidth=4, zorder=2)
    else:
        plt.scatter(row['time_avg'], row['auc_median'], color=sns.color_palette("husl", len(method_stats))[idx], 
                    s=350, alpha=1.0, edgecolor='gray', linewidth=4, zorder=2)
    
    if row['method'] in ['EIF', 'LUNAR']:
        plt.text(row['time_avg'] - 0.15*row['time_avg'], row['auc_median'] - 2.5, row['method'], fontsize=27, ha='left', va='bottom', 
                color='black', transform=ax.transData, zorder=3)
    else:
        plt.text(row['time_avg'] + 0.1*row['time_avg'], row['auc_median'] - 1.3, row['method'], fontsize=27, ha='left', va='bottom', 
                color='black', transform=ax.transData, zorder=3)
        
# Customize the plot
plt.xlabel(time_label, fontsize=36)
plt.ylabel(auc_label, fontsize=36)
plt.tick_params(axis='both', labelsize=32)

plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray')
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.2))

plt.xscale('log')  # Uncomment if log scale is needed for time

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

# scaled_df = score_df/score_df.max()*100

# reordered_index_all = score_df.transpose().mean().sort_values(ascending=False).index

# palette = dict(zip(reordered_index_all, sns.color_palette("husl", n_colors=len(reordered_index_all))))

# plot_df = (scaled_df).melt(var_name="dataset", ignore_index=False).reset_index().rename(columns={"index":"method"})
# plt.figure()
# ax = sns.boxplot(x="method",y="value",data=plot_df, order=reordered_index_all, palette=palette)
# labels = ax.get_xticklabels()
# for label in labels:
#     if label.get_text() == "DAD" or label.get_text() == "DADS":
#         label.set_fontweight('bold') 
#         label.set_fontsize(12)       
# ax.set_xticklabels(labels)
# ax.set_title("Percentage of maximum AUC performance")
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig(f"{figure_dir}/ROCAUC_boxplot_all_datasets.eps",format="eps")
# plt.savefig(f"{figure_dir}/ROCAUC_boxplot_all_datasets.png",format="png")
# plt.savefig(f"{figure_dir}/ROCAUC_boxplot_all_datasets.pdf",format="pdf")
# plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Preparation ---
# scaled_df = (score_df / score_df.max()) * 100
scaled_df = score_df
reordered_index_all = (
    score_df.transpose().mean().sort_values(ascending=False).index.to_list()
)

plot_df = (
    scaled_df.melt(var_name="dataset", ignore_index=False)
    .reset_index()
    .rename(columns={"index": "method"})
)

palette = dict(
    zip(
        reordered_index_all,
        sns.color_palette("husl", n_colors=len(reordered_index_all)),
    )
)

# --- Plot Setup ---
fig, ax = plt.subplots(1, 1, figsize=(8.6, 3.0))
sns.reset_orig()
sns.set_style("white")

ax.set_facecolor("white")
# Create the boxplot (Updated to fix the palette & hue warning)
ax = sns.boxplot(
    x="method",
    y="value",
    data=plot_df,
    order=reordered_index_all,
    hue="method",  # <-- Fixes the FutureWarning
    palette=palette,
    legend=False,  # <-- Prevents a huge duplicate legend
    showfliers=False,
    meanprops=dict(color="k", linestyle="--"),
    showmeans=True,
    meanline=True,
    ax=ax,
)

for spine in ["top", "bottom", "left", "right"]:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_color("black")
    ax.spines[spine].set_linewidth(1.0)  # Adjust thickness if desired
# --- Styling & Typography ---
# Apply the formatting directly using plt.xticks to bypass the FixedLocator warning
# plt.xticks(
#     ticks=range(len(reordered_index_all)),
#     labels=reordered_index_all,
#     rotation=90,
#     fontsize=12,
# )
# rename DAD_Auto to DAD$_{Auto}$ in the x-axis labels
new_labels = ["DAD$_{Auto}$" if label == "DAD_Auto" else label for label in reordered_index_all
]
plt.xticks(
    ticks=range(len(reordered_index_all)),
    labels=new_labels,
    rotation=90,
    fontsize=12,
)
for tick in ax.get_xticklabels():
    if tick.get_text() in ["DAD", "DADS", "DAD$_{Auto}$"]:
        tick.set_fontweight("bold")

# remove "method" from the x-axis label
ax.set_xlabel("")

# Labels and Title
plt.ylabel("AUC", fontsize=12)
# ax.set_title("Percentage of maximum AUC performance", fontsize=14)

# --- Save and Show (Updated to fix tight_layout & EPS transparency warnings) ---
# bbox_inches='tight' does a better job than plt.tight_layout() and prevents layout warnings
plt.savefig(
    f"{figure_dir}/ROCAUC_boxplot_all_datasets.eps",
    format="eps",
    bbox_inches="tight",
    facecolor="white",
)
plt.savefig(
    f"{figure_dir}/ROCAUC_boxplot_all_datasets.png",
    format="png",
    bbox_inches="tight",
    facecolor="white",
)
plt.savefig(
    f"{figure_dir}/ROCAUC_boxplot_all_datasets.pdf",
    format="pdf",
    bbox_inches="tight",
    facecolor="white",
)
plt.show()



# import seaborn as sns
# import matplotlib.pyplot as plt

# rank_list = sorted_mean_df.index[:]

# fig, ax = plt.subplots(1, 1, figsize=(8.6, 4.5))
# sns.reset_orig()
# rank_list = sorted_mean_df.index.to_list()

# df_acc = df_VUS_PR

# df_acc_plot = df_acc.rename(
#     columns={
#         "DAD_Auto": "DAD$_{Auto}$",
#         "DAD": "DAD",
#     }
# )

# rank_list_plot = [
#     "DAD$_{Auto}$" if x == "DAD_Auto" else  x
#     for x in rank_list
# ]

# ax = sns.boxplot(
#     data=df_acc_plot[rank_list_plot],
#     showfliers=False,
#     meanprops=dict(color='k', linestyle='--'),
#     showmeans=True,
#     meanline=True
# )
# # boldface for DAD variants
# for tick in ax.get_xticklabels():
#     if tick.get_text() in ["DAD$_{Auto}$", "DAD", "DADS"]:
#         tick.set_fontweight("bold")

# plt.xticks(ticks=range(len(rank_list_plot)), labels=rank_list_plot, rotation=90, fontsize=12)
# plt.ylabel('VUS-PR', fontsize=12)
# plt.tight_layout()

# plt.figure()
# palette = dict(zip(reordered_index_all, sns.color_palette("husl", n_colors=len(reordered_index_all))))
# ax = sns.violinplot(x="method", y="value", data=plot_df, order=reordered_index_all, palette=palette, inner=None)
# sns.stripplot(x="method", y="value", data=plot_df, order=reordered_index_all, color="black", size=2, alpha=0.35, ax=ax)
# labels = ax.get_xticklabels()
# for label in labels:
#     if label.get_text() == "DAD" or label.get_text() == "DADS":
#         label.set_fontweight('bold')
#         label.set_fontsize(12)
# ax.set_xticklabels(labels)
# ax.set_title("Percentage of maximum AUC performance")
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig(f"{figure_dir}/ROCAUC_violin_all_datasets.eps", format="eps")
# plt.savefig(f"{figure_dir}/ROCAUC_violin_all_datasets.png", format="png")
# plt.savefig(f"{figure_dir}/ROCAUC_violin_all_datasets.pdf", format="pdf")
# plt.show()
from matplotlib.patches import PathPatch

fig, ax = plt.subplots(figsize=(6, 3))

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

palette = dict(zip(reordered_index_all, sns.color_palette("husl", n_colors=len(reordered_index_all))))
ax = sns.violinplot(x="method", y="value", data=plot_df, order=reordered_index_all, color="white", inner='quartile', width=0.95, linewidth=0.8, linecolor="black")
sns.stripplot(x="method", y="value", data=plot_df, order=reordered_index_all, color="#870000", size=1.5, edgecolor='red', linewidth = 0.08, alpha=0.4, ax=ax)
counts = plot_df[plot_df['value'] == 100]['method'].value_counts().reindex(reordered_index_all, fill_value=0)
for i, method in enumerate(reordered_index_all):
    count = counts[method]
    ax.text(i, 0, f'#{count}', ha='right', va='bottom', fontsize=10, color='black', fontweight='bold', rotation=90, alpha=0.7)

labels = ax.get_xticklabels()
for label in labels:
    if label.get_text() == "DAD" or label.get_text() == "DADS":
        label.set_fontweight('bold')
        label.set_fontsize(12)
ax.set_xticklabels(labels)
# remove y-tick = 120
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_title("Percentage of maximum AUC performance")

# disable the x labels
ax.set_xlabel("")
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig(f"{figure_dir}/ROCAUC_violin_all_datasets.eps", format="eps", bbox_inches='tight')
plt.savefig(f"{figure_dir}/ROCAUC_violin_all_datasets.png", format="png", bbox_inches='tight')
plt.savefig(f"{figure_dir}/ROCAUC_violin_all_datasets.pdf", format="pdf", bbox_inches='tight')
plt.show()
#%% clustermap
#Do clustering on percentage of performance, rather than straight AUC

# Recreate the source dataset and column names based on your synthetic format
synth_datasets_1 = [f"synthetic_2_{j}" for j in range(1, 13)]
synth_datasets_2 = [f"synthetic_8_{j}" for j in range(1, 9)]
synth_datasets = synth_datasets_1 + synth_datasets_2

# Create the target LaTeX names (L_1 to L_12 and H_1 to H_8)
target_columns_1 = [fr"$\mathit{{L_{{{j}}}}}$" for j in range(1, 13)]
target_columns_2 = [fr"$\mathit{{H_{{{j}}}}}$" for j in range(1, 9)]
target_columns = target_columns_1 + target_columns_2

# Map Synthetic_i_j -> L_j / H_j
column_mapping = dict(zip(synth_datasets, target_columns))

# Extract and rename
plot_df = metric_dfs["ROC/AUC"][synth_datasets].astype(float).rename(columns=column_mapping)

# plot_df = metric_dfs["ROC/AUC"].astype(float)
# rename DAD_Auto to DAD$_{Auto}$ in the index
plot_df.rename(index={"DAD_Auto": "DAD$_{Auto}$"}, inplace=True)
# clustermap = sns.clustermap(plot_df.transpose().iloc[:,:], method="average",metric="correlation", figsize=(15,15), cbar_pos=(1.055, 0.1, 0.03, 0.7))
clustermap = sns.clustermap(plot_df.transpose().iloc[:,:], method="average", metric="correlation", figsize=(15,10), cbar_pos=(1.055, 0.1, 0.03, 0.7))

clustermap.ax_cbar.tick_params(labelsize=26)

# clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(), fontsize=18)
# clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.get_yticklabels(), fontsize=18)

clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(), fontsize=30, rotation=90)
clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.get_yticklabels(), fontsize=20, rotation=0)  # Adjust rotation as needed

clustermap.ax_heatmap.grid(False)

# disable cluster connecting lines
clustermap.ax_row_dendrogram.set_visible(False)
clustermap.ax_col_dendrogram.set_visible(False)


for label in clustermap.ax_heatmap.get_xticklabels():
    if label.get_text() == "DAD" or label.get_text() == "DADS" or label.get_text() == "DAD$_{Auto}$":
        label.set_fontweight('bold')  
        label.set_fontsize(30)

# same font size for y-axis labels
for label in clustermap.ax_heatmap.get_yticklabels():
    label.set_fontsize(30)

# rename DAD_Auto to DAD$_{Auto}$ in the x-axis labels
# for label in clustermap.ax_heatmap.get_xticklabels():
#     if label.get_text() == "DAD_Auto":
#         label.set_text("DAD$_{Auto}$")

clustermap.savefig(f"{figure_dir}/clustermap_all_datasets.eps",format="eps", dpi=1000)
clustermap.savefig(f"{figure_dir}/clustermap_all_datasets.png",format="png")
clustermap.savefig(f"{figure_dir}/clustermap_all_datasets.pdf",format="pdf")
plt.show()


# cell_size = 16*3  # pixels per heatmap cell
# hcell_size = 12*16  # pixels per heatmap cell height

# rows, cols = plot_df.shape
# fig_width = cols * cell_size / 100  # convert to inches
# fig_height = rows * hcell_size / 100


# clustermap = sns.clustermap(
#     plot_df.transpose().iloc[:, :],
#     method="average",
#     metric="correlation",
#     figsize=(fig_width, fig_height),
#     cbar_pos=(1.13, 0.24, 0.02, 0.6),
#     dendrogram_ratio=(0.07, 0.07),  # shrink row and column dendrograms
#     colors_ratio=0.01,  # shrink space for colorbar if using col_colors/row_colors
#     xticklabels=True,
#     yticklabels=True
# )

# clustermap.ax_cbar.tick_params(labelsize=36)
# # Rotate x-axis labels (already in your code)
# clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(), fontsize=36, rotation=90)

# # Rotate y-axis labels (updated line)
# clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.get_yticklabels(), fontsize=36, rotation=0)  # Adjust rotation as needed
# # check if xlabel name is DECODE then make it bold
# for label in clustermap.ax_heatmap.get_xticklabels():
#     if label.get_text() == "DAD" or label.get_text() == "DADS" or label.get_text() == "DAD_Auto":
#         label.set_fontweight('bold')  
#         label.set_fontsize(36)

# # Save the figures
# clustermap.savefig(f"{figure_dir}/clustermap_all_datasets.eps",format="eps", dpi=1000)
# clustermap.savefig(f"{figure_dir}/clustermap_all_datasets.png",format="png")
# clustermap.savefig(f"{figure_dir}/clustermap_all_datasets.pdf",format="pdf")
# plt.show()

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
    if label.get_text() == "DAD" or label.get_text() == "DADS":
        label.set_fontweight('bold')  
        label.set_fontsize(12)        
ax.set_xticklabels(labels)
ax.set_title("Percentage of maximum AUC performance")
ax.set_yticks([0, 20, 40, 60, 80, 100])

# disable the x labels
ax.set_xlabel("")

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
#     if label.get_text() == "DAD" or label.get_text() == "DADS":
#         label.set_fontweight('bold')
#         label.set_fontsize(12)
# ax.set_xticklabels(labels)
# ax.set_title("Percentage of maximum AUC performance")
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig(f"{figure_dir}/ROCAUC_violin_local_datasets.eps", format="eps")
# plt.savefig(f"{figure_dir}/ROCAUC_violin_local_datasets.png", format="png")
# plt.savefig(f"{figure_dir}/ROCAUC_violin_local_datasets.pdf", format="pdf")
# plt.show()

# plt.figure()
# palette = dict(zip(reordered_index_local, sns.color_palette("husl", n_colors=len(reordered_index_local))))
# ax = sns.violinplot(x="method", y="value", data=plot_df, order=reordered_index_local, palette=palette, inner=None)
# sns.stripplot(x="method", y="value", data=plot_df, order=reordered_index_local, color="black", size=2, alpha=0.35, ax=ax)
# counts = plot_df[plot_df['value'] == 100]['method'].value_counts().reindex(reordered_index_local, fill_value=0)
# for i, method in enumerate(reordered_index_local):
#     count = counts[method]
#     ax.text(i, 115, f'{count}', ha='right', va='bottom', fontsize=10, color='blue')

fig, ax = plt.subplots(figsize=(6, 3))

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

palette = dict(zip(reordered_index_local, sns.color_palette("husl", n_colors=len(reordered_index_local))))
ax = sns.violinplot(x="method", y="value", data=plot_df, order=reordered_index_local, color="white", inner='quartile', width=0.95, linewidth=0.8, linecolor="black")
sns.stripplot(x="method", y="value", data=plot_df, order=reordered_index_local, color="#870000", size=1.5, edgecolor='red', linewidth = 0.08, alpha=0.4, ax=ax)
counts = plot_df[plot_df['value'] == 100]['method'].value_counts().reindex(reordered_index_local, fill_value=0)

for i, method in enumerate(reordered_index_local):
    count = counts[method]
    ax.text(i, 20, f'#{count}', ha='right', va='bottom', fontsize=10, color='black', fontweight='bold', rotation=90, alpha=0.7)

labels = ax.get_xticklabels()
for label in labels:
    if label.get_text() == "DAD" or label.get_text() == "DADS":
        label.set_fontweight('bold')
        label.set_fontsize(12)
ax.set_xticklabels(labels)

ax.set_title("Percentage of maximum AUC performance")
ax.set_yticks([0, 20, 40, 60, 80, 100])

ax.set_xlabel("")
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig(f"{figure_dir}/ROCAUC_violin_local_datasets.eps", format="eps", bbox_inches='tight')
plt.savefig(f"{figure_dir}/ROCAUC_violin_local_datasets.png", format="png", bbox_inches='tight')
plt.savefig(f"{figure_dir}/ROCAUC_violin_local_datasets.pdf", format="pdf", bbox_inches='tight')
plt.show()
#%%
plot_df = metric_dfs["ROC/AUC"][local_datasets].astype(float)

# clustermap = sns.clustermap(plot_df.transpose().iloc[:,:], method="average",metric="correlation", figsize=(15,15), cbar_pos=(1.055, 0.1, 0.03, 0.7))

# clustermap.ax_cbar.tick_params(labelsize=26)

# # clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(), fontsize=18)
# # clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.get_yticklabels(), fontsize=18)

# clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(), fontsize=30, rotation=90)
# clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.get_yticklabels(), fontsize=20, rotation=0)  # Adjust rotation as needed

# for label in clustermap.ax_heatmap.get_xticklabels():
#     if label.get_text() == "DAD" or label.get_text() == "DADS" or label.get_text() == "DAD_Auto":
#         label.set_fontweight('bold')  
#         label.set_fontsize(30)

cell_size = 16*4  # pixels per heatmap cell
hcell_size = 12*4  # pixels per heatmap cell height

rows, cols = plot_df.shape
fig_width = cols * cell_size / 100  # convert to inches
fig_height = rows * hcell_size / 100


clustermap = sns.clustermap(
    plot_df.transpose().iloc[:, :],
    method="average",
    metric="correlation",
    figsize=(fig_width, fig_height),
    cbar_pos=(1.1, 0.24, 0.02, 0.6),
    dendrogram_ratio=(0.07, 0.07),  # shrink row and column dendrograms
    colors_ratio=0.01,  # shrink space for colorbar if using col_colors/row_colors
    xticklabels=True,
    yticklabels=True
)

clustermap.ax_cbar.tick_params(labelsize=22)
# Rotate x-axis labels (already in your code)
clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(), fontsize=22, rotation=90)

# Rotate y-axis labels (updated line)
clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.get_yticklabels(), fontsize=22, rotation=0)  # Adjust rotation as needed
# check if xlabel name is DECODE then make it bold
for label in clustermap.ax_heatmap.get_xticklabels():
    if label.get_text() == "DAD" or label.get_text() == "DADS":
        label.set_fontweight('bold')  
        label.set_fontsize(22)


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
ax.set_title("Percentage of maximum AUC performance")
labels = ax.get_xticklabels()
for label in labels:
    if label.get_text() == "DAD" or label.get_text() == "DADS":
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
#     if label.get_text() == "DAD" or label.get_text() == "DADS":
#         label.set_fontweight('bold')
#         label.set_fontsize(12)
# ax.set_xticklabels(labels)
# ax.set_title("Percentage of maximum AUC performance")
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.savefig(f"{figure_dir}/ROCAUC_violin_global_datasets.eps", format="eps")
# plt.savefig(f"{figure_dir}/ROCAUC_violin_global_datasets.png", format="png")
# plt.savefig(f"{figure_dir}/ROCAUC_violin_global_datasets.pdf", format="pdf")
# plt.show()

# plt.figure()
# palette = dict(zip(reordered_index_global, sns.color_palette("husl", n_colors=len(reordered_index_global))))
# ax = sns.violinplot(x="method", y="value", data=plot_df, order=reordered_index_global, palette=palette, inner=None)
# sns.stripplot(x="method", y="value", data=plot_df, order=reordered_index_global, color="black", size=2, alpha=0.35, ax=ax)
# counts = plot_df[plot_df['value'] == 100]['method'].value_counts().reindex(reordered_index_global, fill_value=0)
# for i, method in enumerate(reordered_index_global):
#     count = counts[method]
#     ax.text(i, 115, f'{count}', ha='right', va='bottom', fontsize=10, color='blue')

fig, ax = plt.subplots(figsize=(6, 3))

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
palette = dict(zip(reordered_index_global, sns.color_palette("husl", n_colors=len(reordered_index_global))))
ax = sns.violinplot(x="method", y="value", data=plot_df, order=reordered_index_global, color="white", inner='quartile', width=0.95, linewidth=0.8, linecolor="black")
sns.stripplot(x="method", y="value", data=plot_df, order=reordered_index_global, color="#870000", size=1.5, edgecolor='red', linewidth = 0.08, alpha=0.4, ax=ax)
counts = plot_df[plot_df['value'] == 100]['method'].value_counts().reindex(reordered_index_global, fill_value=0)

for i, method in enumerate(reordered_index_global):
    count = counts[method]
    ax.text(i, 0, f'#{count}', ha='right', va='bottom', fontsize=10, color='black', fontweight='bold', rotation=90, alpha=0.7)


labels = ax.get_xticklabels()
for label in labels:
    if label.get_text() == "DAD" or label.get_text() == "DADS":
        label.set_fontweight('bold')
        label.set_fontsize(12)
ax.set_xticklabels(labels)
ax.set_yticks([0, 20, 40, 60, 80, 100])

ax.set_title("Percentage of maximum AUC performance")

ax.set_xlabel("")
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig(f"{figure_dir}/ROCAUC_violin_global_datasets.eps", format="eps", bbox_inches='tight')
plt.savefig(f"{figure_dir}/ROCAUC_violin_global_datasets.png", format="png", bbox_inches='tight')
plt.savefig(f"{figure_dir}/ROCAUC_violin_global_datasets.pdf", format="pdf", bbox_inches='tight')
plt.show()
#%%
plot_df = metric_dfs["ROC/AUC"].drop(columns=local_datasets + non_cluster_datasets).astype(float)

# clustermap = sns.clustermap(plot_df.transpose().iloc[:,:], method="average",metric="correlation", figsize=(15,15), cbar_pos=(1.055, 0.1, 0.03, 0.7))

# clustermap.ax_cbar.tick_params(labelsize=26)

# # clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(), fontsize=18)
# # clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.get_yticklabels(), fontsize=18)

# clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(), fontsize=30, rotation=90)
# clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.get_yticklabels(), fontsize=20, rotation=0)  # Adjust rotation as needed

# for label in clustermap.ax_heatmap.get_xticklabels():
#     if label.get_text() == "DAD" or label.get_text() == "DADS" or label.get_text() == "DAD_Auto":
#         label.set_fontweight('bold')  
#         label.set_fontsize(30)

cell_size = 16*3  # pixels per heatmap cell
hcell_size = 12*7  # pixels per heatmap cell height

rows, cols = plot_df.shape
fig_width = cols * cell_size / 100  # convert to inches
fig_height = rows * hcell_size / 100


clustermap = sns.clustermap(
    plot_df.transpose().iloc[:, :],
    method="average",
    metric="correlation",
    figsize=(fig_width, fig_height),
    cbar_pos=(1.13, 0.24, 0.02, 0.6),
    dendrogram_ratio=(0.07, 0.07),  # shrink row and column dendrograms
    colors_ratio=0.01,  # shrink space for colorbar if using col_colors/row_colors
    xticklabels=True,
    yticklabels=True
)

clustermap.ax_cbar.tick_params(labelsize=28)
# Rotate x-axis labels (already in your code)
clustermap.ax_heatmap.set_xticklabels(clustermap.ax_heatmap.get_xticklabels(), fontsize=28, rotation=90)

# Rotate y-axis labels (updated line)
clustermap.ax_heatmap.set_yticklabels(clustermap.ax_heatmap.get_yticklabels(), fontsize=28, rotation=0)  # Adjust rotation as needed
# check if xlabel name is DECODE then make it bold
for label in clustermap.ax_heatmap.get_xticklabels():
    if label.get_text() == "DAD" or label.get_text() == "DADS":
        label.set_fontweight('bold')  
        label.set_fontsize(28)


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

#%%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import operator
import math
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
import networkx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import seaborn as sns

def Friedman_Nemenyi(alpha=0.05, df_perf=None):
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()
    # Record the maximum number of datasets
    max_nb_datasets = df_counts['count'].max()
    # Create a list of classifiers
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['classifier_name'])

    # print('classifiers: ', classifiers)

    '''
    Expected input format for friedmanchisquare is:
                Dataset1        Dataset2        Dataset3        Dataset4        Dataset5
    classifer1
    classifer2
    classifer3 
    '''

    # Compute friedman p-value
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
        for c in classifiers))[1]

    # Decide whether to reject the null hypothesis
    # If p-value >= alpha: we cannot reject the null hypothesis. No statistical difference.
    if friedman_p_value >= alpha:
        print('No statistical difference...')
        return None,None,None
    # Friedman test OK
    # Prepare input for Nemenyi test
    data = []
    for c in classifiers:
        data.append(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
    data = np.array(data, dtype=np.float64)
    # Conduct the Nemenyi post-hoc test
    # print(classifiers)
    # Order is classifiers' order
    nemenyi = posthoc_nemenyi_friedman(data.T)

    # print(nemenyi)
    
    # Original code: p_values.append((classifier_1, classifier_2, p_value, False)), True: represents there exists statistical difference
    p_values = []

    # Comparing p-values with the alpha value
    for nemenyi_indx in nemenyi.index:
        for nemenyi_columns in nemenyi.columns:
            if nemenyi_indx < nemenyi_columns:
                if nemenyi.loc[nemenyi_indx, nemenyi_columns] < alpha:
                    p_values.append((classifiers[nemenyi_indx], classifiers[nemenyi_columns], nemenyi.loc[nemenyi_indx, nemenyi_columns], True))
                else:
                    p_values.append((classifiers[nemenyi_indx], classifiers[nemenyi_columns], nemenyi.loc[nemenyi_indx, nemenyi_columns], False))
            else: continue

    # Nemenyi test OK

    m = len(classifiers)

    # Sort by classifier name then by dataset name
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)]. \
        sort_values(['classifier_name', 'dataset_name'])

    rank_data = np.array(sorted_df_perf['accuracy']).reshape(m, max_nb_datasets)

    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=np.unique(sorted_df_perf['dataset_name']))

    dfff = df_ranks.rank(ascending=False)
    # compute average rank
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    
    return p_values, average_ranks, max_nb_datasets

def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=200, textspace=1, reverse=False, filename=None, **kwargs):
    
    width = width
    textspace = float(textspace)
    '''l is an array of array 
        [[......]
         [......]
         [......]]; 
    n is an integer'''
    # n th column
    def nth(l, n):
        n = lloc(l, n)
        # Return n th column
        return [a[n] for a in l]
    
    '''l is an array of array 
        [[......]
         [......]
         [......]]; 
    n is an integer'''
    # return an integer, count from front or from back.
    def lloc(l, n):
        if n < 0:
            return len(l[0]) + n
        else:
            return n
    # lr is an array of integers
    # Maximum range start from all zeros. Returns an iterable element of tuple.
    def mxrange(lr):
        # If nothing in the array
        if not len(lr):
            yield ()
        else:
            index = lr[0]
            # Check whether index is an integer.
            if isinstance(index, int):
                index = [index]
            # *index: index must be an iterable []
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    # Form a tuple, and generate an iterable value
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums
    # lowv: low value
    if lowv is None:
        '''int(math.floor(min(ssums))): select the minimum value in ssums and take floor.
           Then compare with 1 to see which one is the minimum.'''
        lowv = min(1, int(math.floor(min(ssums))))
    # highv: high value
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4
    # how many algorithms
    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace
    
    # Position of rank
    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        # Set up the format
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # set up the formats
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant + 2

    # matplotlib figure format setup
    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    hf = 1. / height
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    # Line plots
    def line(l, color='k', **kwargs):
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    # Add text to the plot
    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None

    # [lowv, highv], step size is 0.5
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        # If a is an integer
        if a == int(a):
            tick = bigtick
        # Plot a line
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=0.7)

    # Add text to the plot, only for integer value
    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=16)

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    # Format for the first half of algorithms
    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=linewidth)

        color = 'k'
        text(textspace - 0.2, chei, filter_names(nnames[i]), color=color, ha="right", va="center", size=16)
        # text(textspace - 0.2, chei, filter_names(name_mapping[nnames[i]] if nnames[i] in name_mapping.keys() else nnames[i]), color=color, ha="right", va="center", size=16)


    # Format for the second half of algorithms
    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=linewidth)

        color = 'k'
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]), color=color, ha="left", va="center", size=16)
        # text(textspace + scalewidth + 0.2, chei, filter_names(name_mapping[nnames[i]] if nnames[i] in name_mapping.keys() else nnames[i]), color=color, ha="left", va="center", size=16)
        

    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2

        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                  (rankpos(ssums[r]) + side, start)],
                 linewidth=linewidth_sign)
            start += height
            
    start = cline + 0.2
    side = -0.02
    height = 0.1


    #Generate cliques and plot a line to connect elements in cliques    
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    # Plot a line to connect elements in cliques
    for clq in cliques:
        if len(clq) == 1:
            continue
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        # Test
        # print("ssums[min_idx]: {}; ssums[max_idx]: {}".format(ssums[min_idx], ssums[max_idx]))
        line([(rankpos(ssums[min_idx]) - side, start),
              (rankpos(ssums[max_idx]) + side, start)],
             linewidth=linewidth_sign)
        start += height

def form_cliques(p_values, nnames):
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1
    g = networkx.Graph(g_data)

    #Test
    # print("p_values in form_cliques:\n{}".format(p_values))
    # print("g_data:\n{}".format(g_data))

    # Returns all maximal cliques in an undirected graph.
    return networkx.find_cliques(g)


# ============================================================
# Critical Difference Diagram (Friedman + Nemenyi)
# ============================================================

print("\nGenerating Critical Difference Diagram...")

score_df = metric_dfs["ROC/AUC"].astype(float)

eval_list = []

for dataset in score_df.columns:
    for method in score_df.index:
        eval_list.append([
            method,
            dataset,
            score_df.loc[method, dataset]
        ])

eval_df = pd.DataFrame(
    eval_list,
    columns=["classifier_name", "dataset_name", "accuracy"]
)

# prettier names
eval_df["classifier_name"] = (
    eval_df["classifier_name"]
    .replace("DAD_Auto", r"DAD$_{Auto}$")
    .replace("kNN", r"$k$-NN")
    .replace("kth-NN", r"$k$th-NN")
)

p_values, average_ranks, _ = Friedman_Nemenyi(
    df_perf=eval_df,
    alpha=0.05
)

if p_values is not None:

    ranking = average_ranks.index[::-1].tolist()

    print("Top ranking methods:")
    print(ranking[:10])

    plt.figure()

    graph_ranks(
        average_ranks.values,
        average_ranks.index.values,
        p_values,
        cd=None,
        reverse=True,
        width=12,
        textspace=1.5
    )

    plt.title(
        r"Critical Difference Diagram ($\alpha = 0.05$)",
        fontsize=20
    )

    plt.tight_layout()

    plt.savefig(
        f"{figure_dir}/critical_difference_diagram.pdf",
        dpi=1200,
        bbox_inches="tight"
    )

    plt.savefig(
        f"{figure_dir}/critical_difference_diagram.png",
        dpi=1200,
        bbox_inches="tight"
    )

    plt.savefig(
        f"{figure_dir}/critical_difference_diagram.eps",
        format="eps",
        dpi=1200,
        bbox_inches="tight"
    )

    plt.show()

else:
    print("Friedman test not significant; CD diagram not generated.")