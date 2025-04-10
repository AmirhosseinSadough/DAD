#%% setup
import pickle
import os
import gc
import time
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from pyod.utils.utility import precision_n_scores
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import ParameterGrid
from evaluation_metrics import adjusted_precision_n_scores, adjusted_average_precision

import shlex
import subprocess

import argparse

# formatted_data_dir = "formatted_data"
# base_result_dir = "results_hpt"
result_dir = "result_dir"
csvresult_dir = "csvresult_dir"
score_dir = "score_dir"

result_train_dir = "result_dir_train"
csvresult_train_dir = "csvresult_dir_train"
score_train_dir = "score_dir_train"
best_hyp_train_dir = "best_hyp_train_dir"
best_hyp_val_train_dir = "best_hyp_val_train_dir"
log_dir = "logs"
preprocessed_data_dir = "preprocessed_data"
DeepSVDD_dir = "additional_methods/Deep-SVDD"

DeepSVDD_conda_env = "myenv"

#define score function:
score_functions = {"ROC/AUC": roc_auc_score, 
                   "R_precision": precision_n_scores, 
                   "adjusted_R_precision": adjusted_precision_n_scores, 
                   "average_precision": average_precision_score, 
                   "adjusted_average_precision": adjusted_average_precision}

#%% check filename valid:
    
def fix_filename(filename):
    # if Windows OS, replace : by _
    if os.name == "nt":
        return filename.replace(":", "_")
    else:
        return filename

#%% argument parsing for command line functionality
# Create the parser
arg_parser = argparse.ArgumentParser(description='Run selected methods over all datasets')

# Add the arguments
arg_parser.add_argument('--method',
                       metavar='M',
                       dest='method',
                       default='all',
                       type=str,
                       help='The method that you would like to run')

arg_parser.add_argument('--dataset',
                       metavar='D',
                       dest='dataset',
                       default="all",
                       type=str,
                       help='The dataset you would like to run.')

arg_parser.add_argument('--verbose',
                       metavar='V',
                       dest='verbose',
                       default=1,
                       type=int,
                       help='The verbosity of the pipeline execution.')

arg_parser.add_argument('--input_type',
                       metavar='I',
                       dest='input_type',
                       default="npz",
                       type=str,
                       help='The extension type of the processed data. Can be either "npz" or "pickle".')

arg_parser.add_argument('--skip-CBLOF',
                       metavar='C',
                       dest='skip_CBLOF',
                       default=1,
                       type=int,
                       help='Bool to skip CBLOF execution during method = "all". When CBLOF has been calculated previously, redundant invalid clusterings will be calculated when this is set to 0 (False).')

arg_parser.add_argument('--formatted_data_dir',
                        dest='formatted_data_dir',
                        default="datasets/synthetic",
                        type=str,
                        help='Path to the formatted dataset directory.')

arg_parser.add_argument('--base_result_dir',
                        dest='base_result_dir',
                        default="results/synthetic",
                        type=str,
                        help='Path to the base result directory.')

# Execute the parse_args() method
parsed_args = arg_parser.parse_args()

method_to_run = parsed_args.method
verbose = parsed_args.verbose
skip_CBLOF = parsed_args.skip_CBLOF
include_datasets = parsed_args.dataset
input_type = parsed_args.input_type


formatted_data_dir = parsed_args.formatted_data_dir
base_result_dir = parsed_args.base_result_dir

#%% Define parameter settings and methods

from pyod.models.inne import INNE
from pyod.models.kde import KDE
from pyod.models.gmm import GMM
#from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
#from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN 
#from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.ecod import ECOD
from pyod.models.lunar import LUNAR
from pyod.models.so_gaal import SO_GAAL
#from pyod.models.mo_gaal import MO_GAAL
from pyod.models.combination import maximization

from additional_methods.ensemble import  Ensemble
from additional_methods.wrappers.ExtendedIForest import ExtendedIForest
from additional_methods.ODIN import ODIN
from additional_methods.gen2out.gen2out import gen2Out
#from additional_methods.SVDD.src.BaseSVDD import BaseSVDD
from additional_methods.wrappers.HBOS import DynamicHBOS

from additional_methods.wrappers.AE import AE_wrapper
from additional_methods.wrappers.VAE import VAE_wrapper
#from additional_methods.wrappers.rrcf import rrcf_wrapper
from additional_methods.wrappers.ALAD import ALAD_wrapper

from additional_methods.cof import COF
from additional_methods.abod import ABOD
from additional_methods.sod import SOD
from additional_methods.lmdd import LMDD


from additional_methods.wrappers.DECODE import DECODE

ensemble_LOF_krange = range(5,31,3)

#dict of methods and functions
method_classes = {
        "DECODE":DECODE,
        "DECODE_s":DECODE,
        "GMM":GMM,
        "COF":COF,
        "kNN":KNN,
        "kth-NN":KNN,
        "IF":IForest,
        "ensemble-LOF":Ensemble,
        "LOF":LOF,
        "MCD":MCD,
        "OCSVM":OCSVM,
        "PCA":PCA, 
        "SOD":SOD,
        "EIF":ExtendedIForest,
        "ODIN":ODIN,
        "LUNAR":LUNAR,
        "DynamicHBOS":DynamicHBOS
        }

#dict of methods and parameters
method_parameters = {
        "DECODE":{"lr":[0.8, 0.2, 8e-2, 2e-2, 8e-3, 2e-3, 8e-4, 2e-4, 8e-5, 2e-5, 8e-6, 8e-7], "mom_score":[0.25], "win_size":[0]},
        "DECODE_s":{"lr":[0.8, 0.2, 8e-2, 2e-2, 8e-3, 2e-3, 8e-4, 2e-4, 8e-5, 2e-5, 8e-6, 8e-7], "mom_score":[0.25], "win_size":[0, 1]},
        "GMM":{"n_components":range(2,15)},
        "COF":{"n_neighbors":[10,20,30]},
        "kNN":{"n_neighbors":range(5,31,3), "method":["mean"]},
        "kth-NN":{"n_neighbors":range(5,31,3), "method":["largest"]},
        "IF":{"n_estimators":[1000], "max_samples":[128,256,512,1024]},
        "ensemble-LOF":{"estimators":[[LOF(n_neighbors=k) for k in ensemble_LOF_krange]], "combination_function":[maximization]},
        "LOF":{"n_neighbors":range(5,31,3)},
        "MCD":{"support_fraction":[0.6,0.7,0.8,0.9], "assume_centered":[True]},
        "OCSVM":{"kernel":["rbf"], "gamma":["auto"], "nu":[0.5,0.6,0.7,0.8,0.9]},
        "PCA":{"n_components":[0.3,0.5,0.7,0.9]}, 
        "SOD":{"n_neighbors":[20,30], "ref_set":[10,18], "alpha":[0.7,0.9]},
        "EIF":{"n_estimators":[1000], "max_samples":[128,256,512,1024], "extension_level":[1,2,3]},
        "ODIN":{"n_neighbors":range(5,31,3)},
        "LUNAR":{"n_neighbours":[5, 10, 15, 20, 25 ,30]}, #parameter is inconsistently named n_neighbours 
        "DynamicHBOS":{}
        }

method_parameters_test = {}
#%% 
#sort dataset_names based on size: https://stackoverflow.com/questions/20252669/get-files-from-directory-argument-sorting-by-size
# make a generator for all file paths within dirpath
all_files = ( os.path.join(basedir, filename) for basedir, dirs, files in os.walk(formatted_data_dir) for filename in files   )
sorted_files = sorted(all_files, key = os.path.getsize)
dataset_names = [filename.replace(formatted_data_dir+os.path.sep,"") for filename in sorted_files]
dataset_names = [dataset_name for dataset_name in dataset_names if dataset_name.endswith(input_type)]

#%%
if method_to_run == "all":
    all_methods_to_run = method_classes
else:
    try:
        all_methods_to_run = {method_to_run:method_classes[method_to_run]}
    except KeyError:
        raise KeyError("Specified method is not found in the list of available methods.")


if skip_CBLOF and method_to_run == "all":
    all_methods_to_run.pop("CBLOF", False)
    all_methods_to_run.pop("u-CBLOF", False)
    
if include_datasets == "all":
    pass        
elif include_datasets+"."+input_type in dataset_names:
    dataset_names = [include_datasets+"."+input_type]

#%% manual skip of datasets being calculated on other machines
skip_datasets = ["http","cover", "aloi", "donors", "campaign", "mi-f", "mi-v", "internetads"]
skip_datasets = [dataset+"."+input_type for dataset in include_datasets]
try:
    dataset_names.remove(skip_datasets)
except ValueError:
    pass
#%% loop over all data, but do not reproduce existing results

import ast

target_dir = os.path.join(base_result_dir, result_dir)
target_csvdir = os.path.join(base_result_dir, csvresult_dir)
score_csvdir = os.path.join(base_result_dir, score_dir)

target_train_dir = os.path.join(base_result_dir, result_train_dir)
target_train_csvdir = os.path.join(base_result_dir, csvresult_train_dir)
score_train_csvdir = os.path.join(base_result_dir, score_train_dir)
best_hyp_train_csvdir = os.path.join(base_result_dir, best_hyp_train_dir)
best_hyp_train_val_csvdir = os.path.join(base_result_dir, best_hyp_val_train_dir)
target_wcdir = os.path.join(base_result_dir, "wc_dir")

os.makedirs(target_wcdir, exist_ok=True)

if not os.path.exists(score_csvdir):
    os.makedirs(score_csvdir)

from scipy.spatial.distance import jensenshannon

downsample_rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]  
num_bins = 20  # Number of bins for histograms

for dataset_name in dataset_names:
    
    print("______"+dataset_name+"______train_______")
    
    full_path_filename = os.path.join(formatted_data_dir, dataset_name)
    
    if input_type == "pickle":
        data = pickle.load(open(full_path_filename, 'rb'))
    elif input_type == "npz":
        data  = np.load(open(full_path_filename, 'rb'))
                    
    X, y = data["X"], np.squeeze(data["y"])
    contamination_rate_original = np.sum(y == 1) / len(y)
    dataset_down_contamination = {}
    histogram_distances = {}  # To store histogram comparisons for each downsample rate

    histogram_distances = {}  # To store histogram comparisons for each downsample rate

    # Compute histogram for the full dataset
    full_histograms = [np.histogram(X[:, i], bins=num_bins, density=True)[0] for i in range(X.shape[1])]

    for dr in downsample_rate:
        num_samples = int(dr * X.shape[0])
        indices = np.linspace(0, X.shape[0] - 1, num_samples, dtype=int)
        # Ensure the subset has the correct size by trimming if necessary
        X_d, y_d = X[indices], y[indices]
        if len(X_d) > int(dr * X.shape[0]):
            X_d, y_d = X_d[:int(dr * X.shape[0])], y_d[:int(dr * X.shape[0])]

        # Calculate contamination rate for downsampled data
        contamination_rate_downsample = np.sum(y_d == 1) / len(y_d)
        if contamination_rate_downsample == 0:
            dataset_down_contamination[dr] = 1.0
        else:
            dataset_down_contamination[dr] = np.abs(contamination_rate_downsample - contamination_rate_original)

        # Compute histograms for the downsampled dataset
        down_histograms = [np.histogram(X_d[:, i], bins=num_bins, density=True)[0] for i in range(X.shape[1])]

        # Compute Jensen-Shannon divergence for each feature
        js_distances = [jensenshannon(full_histograms[i], down_histograms[i]) for i in range(X.shape[1])]

        # Average the divergence over all features
        histogram_distances[dr] = np.mean(js_distances)

    print("dataset_down_contamination", dataset_down_contamination)
    print("histogram_distances", histogram_distances)

    # Filter rates ensuring at least one outlier exists
    valid_rates = {
        dr: histogram_distances[dr] for dr in downsample_rate if dataset_down_contamination[dr] != 1.0
    }

    # Find the downsample rate with the minimum histogram distance among valid rates
    if valid_rates:
        best_downsample_rate = min(valid_rates, key=valid_rates.get)
        print(f"Best downsample rate: {best_downsample_rate}")
        print(f"Contamination rate difference: {dataset_down_contamination[best_downsample_rate]}")
        print(f"Histogram distance: {histogram_distances[best_downsample_rate]}")
    else:
        print("No valid downsample rate found with non-zero contamination.")


    dr_selected = min(valid_rates, key=valid_rates.get)
    print(f"Best downsample rate: {dr_selected}")
    #-------------------------------------------------------------------------------------------------------------------
    num_samples = int(dr * X.shape[0])
    indices = np.linspace(0, X.shape[0] - 1, num_samples, dtype=int)    
    X_d, y_d = X[indices], y[indices]
    if len(X_d) > int(dr * X.shape[0]):
        X_d, y_d = X_d[:int(dr * X.shape[0])], y_d[:int(dr * X.shape[0])]        
    X, y = X_d, y_d
    #-------------------------------------------------------------------------------------------------------------------
    max_duplicates = data["max_duplicates"]
    
    #loop over all methods:

    for method_name, OD_class in all_methods_to_run.items():
        print("-" + method_name)
        hyperparameter_grid = method_parameters[method_name]        
        hyperparameter_list = list(ParameterGrid(hyperparameter_grid))
        
        maximum_ROC_hyperparameter = []
        #loop over hyperparameter settings
        for hyperparameter_setting in hyperparameter_list:
            
            if method_name == "ensemble-LOF":              
                hyperparameter_string = str(ensemble_LOF_krange)
            else:
                hyperparameter_string = str(hyperparameter_setting)
                
            if verbose:
                print(hyperparameter_string)
            
            # finding the best hyperparameters for each method per dataset
            besthyp_dir = os.path.join(best_hyp_train_csvdir, dataset_name.replace("."+input_type, ""), method_name, method_name + ".txt")
            directory = os.path.dirname(besthyp_dir)       
            #check whether results have  been calculated
            full_target_dir = os.path.join(target_train_dir, dataset_name.replace("."+input_type, ""), method_name)
            target_file_name = fix_filename(os.path.join(target_train_dir, dataset_name.replace("."+input_type, ""), method_name, hyperparameter_string+".pickle"))
            
            besthyp_dir = os.path.join(best_hyp_train_csvdir, dataset_name.replace("."+input_type, ""), method_name, method_name + ".txt")
            directory = os.path.dirname(besthyp_dir)       

            # if os.path.exists(target_file_name) and os.path.getsize(target_file_name) > 0:
            if os.path.exists(directory) and os.path.getsize(directory) > 0:
                if verbose:
                    print(" results already calculated, skipping recalculation")
            elif method_name == "EIF" and X.shape[1] <= hyperparameter_setting["extension_level"]:
                print("Dimensionality of dataset higher than EIF extension level, skipping...")
            elif (method_name == "kNN" or method_name == "kth-NN" or method_name == "SOD" or method_name == "ODIN") and X.shape[0] < (hyperparameter_setting["n_neighbors"] + 1):
                print("kNN/kth-NN: not enough samples for n_neighbors, skipping...")    
            elif method_name == "LUNAR" and X.shape[0] < (hyperparameter_setting["n_neighbours"]+1):
                print("LUNAR: not enough samples for n_neighbors, skipping...")    
            elif method_name == "GMM"  and X.shape[0] < hyperparameter_setting["n_components"]:
                print("GMM: not enough samples for n_components, skipping...")
            else:

                #use memory efficient COF when too many samples:
                if method_name =="COF" and X.shape[0] > 8000:
                    hyperparameter_setting["method"] = "knn"
                
                #process DeepSVDD differently due to lacking sklearn interface
                #instead: call deepsvdd script from command line with arguments parsed from variables (also needed for custom Conda env)
                if method_name in ["DeepSVDD", "sb-DeepSVDD"]:
                    
                    preprocessed_data_file_name = os.path.join(DeepSVDD_dir, "data", dataset_name)
                    #preprocess data and write to csv:
                    
                    #check if preprocessed data already exists:, if not preprocess and write data
                    if not os.path.exists(preprocessed_data_file_name):
                        scaler = RobustScaler()
                        
                        X_preprocessed = scaler.fit_transform(X)
                        
                        data_dict = {"X": X_preprocessed, "y": y}
                        
                        pickle.dump(data_dict, open(preprocessed_data_file_name, "wb"))    
                    
                    #make shell call to calculate DeepSVDD
                    DeepSVDD_argument_list = shlex.split("conda run -n")
                    DeepSVDD_argument_list.append(DeepSVDD_conda_env)
                    
                    DeepSVDD_argument_list.append("python")
                    DeepSVDD_argument_list.append(os.path.join(DeepSVDD_dir,"src", "main.py"))
                    
                    DeepSVDD_argument_list.append(dataset_name)
                    
                    DeepSVDD_argument_list.append(str(hyperparameter_setting["n_layers"]))
                    DeepSVDD_argument_list.append(str(hyperparameter_setting["shrinkage_factor"]))
                    
                    DeepSVDD_argument_list.append(os.path.join("..", "log", dataset_name))
                    DeepSVDD_argument_list.append(os.path.join(DeepSVDD_dir, "data"))
                    
                    #csv scores
                    full_target_scoredir = os.path.join(score_train_csvdir, dataset_name.replace("."+input_type, ""), method_name)
                    os.makedirs(full_target_scoredir, exist_ok=True)
                    csv_filename = fix_filename(os.path.join(full_target_scoredir, hyperparameter_string+".csv"))
                    DeepSVDD_argument_list.append(csv_filename) #csv
                    
                    
                    #calculate batch size (n_samples % batchsize != 1, otherwise batchnorm breaks)
                    batch_size = 200
                    while X.shape[0] % batch_size == 1:
                        batch_size+=1
                    #append hardcoded arguments:
                    DeepSVDD_argument_list.append("--objective") #csv
                    if method_name == "DeepSVDD":
                        DeepSVDD_argument_list.append("one-class")
                    elif method_name == "sb-DeepSVDD":
                        DeepSVDD_argument_list.append("soft-boundary")
                    DeepSVDD_argument_list += shlex.split("--lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size {0} --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size {0} --ae_weight_decay 0.5e-3 --normal_class 0".format(batch_size))
                                                  
                    subprocess.run(DeepSVDD_argument_list)
                    
                    #read scores, output metrics
                    outlier_scores = np.loadtxt(csv_filename)
                    
                    method_performance = {method_name:{score_name: score_function(y,outlier_scores) for (score_name, score_function) in score_functions.items()}}
                    method_performance_df = pd.DataFrame(method_performance).transpose()
                        
                    os.makedirs(full_target_dir, exist_ok=True)
                    with open(target_file_name, 'wb') as handle:
                        pickle.dump(method_performance_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    #also write csv files for easy manual inspection of metrics
                    full_target_csvdir = os.path.join(target_train_csvdir, dataset_name.replace("."+input_type, ""), method_name)
                    os.makedirs(full_target_csvdir, exist_ok=True)
                    target_csvfile_name = fix_filename(os.path.join(full_target_csvdir, hyperparameter_string+".csv"))
                    method_performance_df.to_csv(target_csvfile_name)
                    
                    
                else:
                    
                    OD_method = OD_class(**hyperparameter_setting)
                    
                    #Temporary fix for ECOD:
                    if method_name == "ECOD" and hasattr(OD_method, "X_train"):
                        delattr(OD_method, "X_train")
                        
                    if method_name == "DECODE":
                        pipeline = make_pipeline(StandardScaler(), OD_method)
                    else:
                        pipeline = make_pipeline(RobustScaler(), OD_method)

            
                    try:
                        pipeline.fit(X)
                    except ValueError as e: #Catch error when CBLOF fails due to configuration
                        if str(e) == "Could not form valid cluster separation. Please change n_clusters or change clustering method":
                            print("Separation invalid, skipping this hyperparameter setting")
                            continue
                        else:
                            raise e
                    #resolve issues with memory leaks with keras
                    if method_name in ["AE", "VAE", "beta-VAE"]:
                        
                        gc.collect() 
                        K.clear_session() 

                    #correct for non pyod-like behaviour from gen2out, needs inversion of scores
                    if method_name == "gen2out":
                        outlier_scores = -pipeline[1].decision_function(RobustScaler().fit_transform(X)) 
                    elif method_name == "SVDD":
                        outlier_scores = -pipeline[1].decision_function(RobustScaler().fit_transform(X)) 
                    else:
                        outlier_scores = pipeline[1].decision_scores_
         

                    method_performance = {method_name:{score_name: score_function(y,outlier_scores) for (score_name, score_function) in score_functions.items()}}
                    method_performance_df = pd.DataFrame(method_performance).transpose()
                    maximum_ROC_hyperparameter.append((hyperparameter_string, method_performance_df["ROC/AUC"].values[0]))    
                    print(f"{method_name}, {hyperparameter_string}, ROC/AUC: {method_performance_df['ROC/AUC'].values[0]}")

                    os.makedirs(full_target_dir, exist_ok=True)
                    with open(target_file_name, 'wb') as handle:
                        pickle.dump(method_performance_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    #also write csv files for easy manual inspection
                    full_target_csvdir = os.path.join(target_train_csvdir, dataset_name.replace("."+input_type, ""), method_name)
                    os.makedirs(full_target_csvdir, exist_ok=True)
                    target_csvfile_name = fix_filename(os.path.join(full_target_csvdir, hyperparameter_string+".csv"))
                    method_performance_df.to_csv(target_csvfile_name)
                    
                    full_target_scoredir = os.path.join(score_train_csvdir, dataset_name.replace("."+input_type, ""), method_name)
                    os.makedirs(full_target_scoredir, exist_ok=True)
                    target_scorefile_name = fix_filename(os.path.join(full_target_scoredir, hyperparameter_string+".csv"))
                    np.savetxt(target_scorefile_name, outlier_scores)
                    
                    #write Keras history for relevant neural methods
                    if method_name in ["VAE", "beta-VAE", "AE", "AnoGAN", "ALAD"]:
                        if method_name == "AnoGAN":
                            history_df = pd.DataFrame({"discriminator_loss":pipeline[1].hist_loss_discriminator, "generator_loss":pipeline[1].hist_loss_generator})
                        elif method_name =="ALAD":
                            history_df = pd.DataFrame({"discriminator_loss":pipeline[1].hist_loss_disc, "generator_loss":pipeline[1].hist_loss_gen})
                        else:
                            history = pipeline[1].history_
                            history_df = pd.DataFrame(history)
                        
                        full_target_dir = os.path.join(log_dir, dataset_name.replace("."+input_type, ""), method_name)
                        target_file_name = fix_filename(os.path.join(log_dir, dataset_name.replace("."+input_type, ""), method_name, hyperparameter_string+"."+input_type))
                        
                        os.makedirs(full_target_dir, exist_ok=True)
                        with open(target_file_name, 'wb') as handle:
                            pickle.dump(history_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
                        full_target_dir = os.path.join(log_dir, dataset_name.replace("."+input_type, ""), method_name)
                        target_file_name = fix_filename(os.path.join(log_dir, dataset_name.replace("."+input_type, ""), method_name, hyperparameter_string+".csv"))
                        
                        os.makedirs(full_target_dir, exist_ok=True)
    
                        history_df.to_csv(target_file_name)

        # finding the best hyperparameters for each method per dataset
        besthyp_dir = os.path.join(best_hyp_train_csvdir, dataset_name.replace("."+input_type, ""), method_name, method_name + ".txt")
        besthyp_val_dir = os.path.join(best_hyp_train_val_csvdir, dataset_name.replace("."+input_type, ""), method_name, method_name + ".txt")
        directory = os.path.dirname(besthyp_dir)       
        directory_val = os.path.dirname(besthyp_val_dir)       
        if os.path.exists(directory) and os.path.getsize(directory) > 0:
            if verbose:
                print("best parameters already calculated, skipping recalculation")
            # read the best hyperparameters from the file
            with open(besthyp_dir, 'r') as f:
                best_hyperparameters = ast.literal_eval(f.read())
                print(f"load: {method_name} Best hyperparameters for {dataset_name} is {best_hyperparameters['best_hyperparameter']}")
                method_parameters_test[method_name] = {best_hyperparameters['best_hyperparameter']}
        else:
            
            if method_name == "DECODE" or method_name =="DECODE_s":
                filtered_hyperparameter = [x for x in maximum_ROC_hyperparameter if x[1] != 0.5]
                maximum_ROC_hyperparameter = sorted(filtered_hyperparameter, key=lambda x: x[1], reverse=True)
            else:
                maximum_ROC_hyperparameter = sorted(maximum_ROC_hyperparameter, key=lambda x: x[1], reverse=True)
            if maximum_ROC_hyperparameter != []:
                print(f"save: {method_name} Best hyperparameters for {dataset_name} is {maximum_ROC_hyperparameter[0][0]}")
                method_parameters_test[method_name] = {maximum_ROC_hyperparameter[0][0]}
                best_hyperparameters = {'best_hyperparameter': maximum_ROC_hyperparameter[0][0]}
                os.makedirs(directory, exist_ok=True)
                with open(besthyp_dir, 'w') as f:
                    f.write(str(best_hyperparameters))
                best_hyperparameters_value = {'best_hyperparameter': maximum_ROC_hyperparameter}
                os.makedirs(directory_val, exist_ok=True)
                with open(besthyp_val_dir, 'w') as f:
                    f.write(str(best_hyperparameters_value))                   
            else:
                print(f"save: {method_name} Best hyperparameters for {dataset_name} is None")
                continue
             
    try:
        besthyp_dir = os.path.join(best_hyp_train_csvdir, dataset_name.replace(f".{input_type}", ""), method_name, method_name + ".txt")
        directory = os.path.dirname(besthyp_dir)

        if not os.path.exists(besthyp_dir) or os.path.getsize(besthyp_dir) == 0:
            print(f"Error: The file {besthyp_dir} does not exist or is empty.")
            continue  # Ensure this is in a loop
        else:
            with open(besthyp_dir, 'r') as f:
                try:
                    best_hyperparameters = ast.literal_eval(f.read())
                    print(f"load again: {method_name} Best hyperparameters for {dataset_name} is {best_hyperparameters['best_hyperparameter']}")
                    method_parameters_test[method_name] = best_hyperparameters['best_hyperparameter']
                except (SyntaxError, ValueError) as e:
                    print(f"Error parsing file {besthyp_dir}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    #print name for reporting purpose
    print("______"+dataset_name+"______test_______")
    
    full_path_filename = os.path.join(formatted_data_dir, dataset_name)
    
    if input_type == "pickle":
        data = pickle.load(open(full_path_filename, 'rb'))
    elif input_type == "npz":
        data  = np.load(open(full_path_filename, 'rb'))
                    
    X, y = data["X"], np.squeeze(data["y"])
    
    max_duplicates = data["max_duplicates"]
    
    #loop over all methods:
    # print("---method_parameters_test 2---", method_parameters_test)
    for method_name, OD_class in all_methods_to_run.items():
        print("-" + method_name)
        
        if method_name not in method_parameters_test or not method_parameters_test[method_name]:
            print(f"Skipping {method_name} as it does not have valid hyperparameters.")
            continue        
        if method_name == "ensemble-LOF":
            hyperparameter_grid = method_parameters[method_name]     
            hyperparameter_list = list(ParameterGrid(hyperparameter_grid))
        else:
            hyperparameter_list = method_parameters_test[method_name]   

        #loop over hyperparameter settings
        for hyperparameter_setting in hyperparameter_list:
            
            if method_name == "ensemble-LOF":               
                hyperparameter_string = str(ensemble_LOF_krange)
            else:
                hyperparameter_string = str(hyperparameter_setting)
                
            if verbose:
                print(hyperparameter_string)

            if method_name == "EIF":
                hs = ast.literal_eval(hyperparameter_setting)
            
            #check whether results have  been calculated
            full_target_dir = os.path.join(target_dir, dataset_name.replace("."+input_type, ""), method_name)
            target_file_name = fix_filename(os.path.join(target_dir, dataset_name.replace("."+input_type, ""), method_name, hyperparameter_string+".pickle"))
            
            if os.path.exists(target_file_name) and os.path.getsize(target_file_name) > 0:
                if verbose:
                    print(" results already calculated, skipping recalculation")
            elif method_name == "EIF" and X.shape[1] <= hs["extension_level"]:
                print("Dimensionality of dataset higher than EIF extension level, skipping...")
            else:
                #process DeepSVDD differently due to lacking sklearn interface
                #instead: call deepsvdd script from command line with arguments parsed from variables (also needed for custom Conda env)
                if method_name in ["DeepSVDD", "sb-DeepSVDD"]:
                    
                    preprocessed_data_file_name = os.path.join(DeepSVDD_dir, "data", dataset_name)
                    #preprocess data and write to csv:
                    
                    #check if preprocessed data already exists:, if not preprocess and write data
                    if not os.path.exists(preprocessed_data_file_name):
                        scaler = RobustScaler()
                        
                        X_preprocessed = scaler.fit_transform(X)
                        
                        data_dict = {"X": X_preprocessed, "y": y}
                        
                        pickle.dump(data_dict, open(preprocessed_data_file_name, "wb"))    
                    
                    #make shell call to calculate DeepSVDD
                    DeepSVDD_argument_list = shlex.split("conda run -n")
                    DeepSVDD_argument_list.append(DeepSVDD_conda_env)
                    
                    DeepSVDD_argument_list.append("python")
                    DeepSVDD_argument_list.append(os.path.join(DeepSVDD_dir,"src", "main.py"))
                    
                    DeepSVDD_argument_list.append(dataset_name)
                    
                    DeepSVDD_argument_list.append(str(hyperparameter_setting["n_layers"]))
                    DeepSVDD_argument_list.append(str(hyperparameter_setting["shrinkage_factor"]))
                    
                    DeepSVDD_argument_list.append(os.path.join("..", "log", dataset_name))
                    DeepSVDD_argument_list.append(os.path.join(DeepSVDD_dir, "data"))
                    
                    #csv scores
                    full_target_scoredir = os.path.join(score_csvdir, dataset_name.replace("."+input_type, ""), method_name)
                    os.makedirs(full_target_scoredir, exist_ok=True)
                    csv_filename = fix_filename(os.path.join(full_target_scoredir, hyperparameter_string+".csv"))
                    DeepSVDD_argument_list.append(csv_filename) #csv
                    
                    
                    #calculate batch size (n_samples % batchsize != 1, otherwise batchnorm breaks)
                    batch_size = 200
                    while X.shape[0] % batch_size == 1:
                        batch_size+=1
                    #append hardcoded arguments:
                    DeepSVDD_argument_list.append("--objective") #csv
                    if method_name == "DeepSVDD":
                        DeepSVDD_argument_list.append("one-class")
                    elif method_name == "sb-DeepSVDD":
                        DeepSVDD_argument_list.append("soft-boundary")
                    DeepSVDD_argument_list += shlex.split("--lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size {0} --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size {0} --ae_weight_decay 0.5e-3 --normal_class 0".format(batch_size))
                                                  
                    subprocess.run(DeepSVDD_argument_list)
                    
                    #read scores, output metrics
                    outlier_scores = np.loadtxt(csv_filename)
                    
                    method_performance = {method_name:{score_name: score_function(y,outlier_scores) for (score_name, score_function) in score_functions.items()}}
                    method_performance_df = pd.DataFrame(method_performance).transpose()
                        
                    os.makedirs(full_target_dir, exist_ok=True)
                    with open(target_file_name, 'wb') as handle:
                        pickle.dump(method_performance_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    #also write csv files for easy manual inspection of metrics
                    full_target_csvdir = os.path.join(target_csvdir, dataset_name.replace("."+input_type, ""), method_name)
                    os.makedirs(full_target_csvdir, exist_ok=True)
                    target_csvfile_name = fix_filename(os.path.join(full_target_csvdir, hyperparameter_string+".csv"))
                    method_performance_df.to_csv(target_csvfile_name)
                    
                    
                else:
                    
                    if method_name == "ensemble-LOF":
                        hyperparameter_setting = hyperparameter_setting
                    elif method_name == "DynamicHBOS" or method_name == "gen2out":
                        hyperparameter_setting = '{}'
                        hyperparameter_setting = ast.literal_eval(hyperparameter_setting)
                    else:
                        hyperparameter_setting = ast.literal_eval(hyperparameter_setting)

                    if method_name =="COF" and X.shape[0] > 8000:
                        hyperparameter_setting["method"] = "knn"

                    OD_method = OD_class(**hyperparameter_setting)
                    
                    #Temporary fix for ECOD:
                    if method_name == "ECOD" and hasattr(OD_method, "X_train"):
                        delattr(OD_method, "X_train")
                        
                    start_time = time.time()
                    
                    if method_name == "DECODE":
                        pipeline = make_pipeline(StandardScaler(), OD_method)
                    else:
                        pipeline = make_pipeline(RobustScaler(), OD_method)
            
                    try:
                        pipeline.fit(X)
                    except ValueError as e: #Catch error when CBLOF fails due to configuration
                        if str(e) == "Could not form valid cluster separation. Please change n_clusters or change clustering method":
                            print("Separation invalid, skipping this hyperparameter setting")
                            continue
                        else:
                            raise e
                    #resolve issues with memory leaks with keras
                    if method_name in ["AE", "VAE", "beta-VAE"]:
                        
                        gc.collect() 
                        K.clear_session() 

                    #correct for non pyod-like behaviour from gen2out, needs inversion of scores
                    if method_name == "gen2out":
                        outlier_scores = -pipeline[1].decision_function(RobustScaler().fit_transform(X)) 
                    elif method_name == "SVDD":
                        outlier_scores = -pipeline[1].decision_function(RobustScaler().fit_transform(X)) 
                    else:
                        outlier_scores = pipeline[1].decision_scores_

                    work_clock_time = time.time() - start_time
                    target_wcfile_name = os.path.join(target_wcdir, dataset_name.replace("."+input_type, ""), method_name, hyperparameter_string + ".csv")
                    os.makedirs(os.path.dirname(target_wcfile_name), exist_ok=True)
                    with open(target_wcfile_name, 'a') as wcfile:
                        wcfile.write(f"{hyperparameter_string},{work_clock_time}\n")
                                      
                    method_performance = {method_name:{score_name: score_function(y,outlier_scores) for (score_name, score_function) in score_functions.items()}}
                    method_performance_df = pd.DataFrame(method_performance).transpose()
                        
                    os.makedirs(full_target_dir, exist_ok=True)
                    with open(target_file_name, 'wb') as handle:
                        pickle.dump(method_performance_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
                    #also write csv files for easy manual inspection
                    full_target_csvdir = os.path.join(target_csvdir, dataset_name.replace("."+input_type, ""), method_name)
                    os.makedirs(full_target_csvdir, exist_ok=True)
                    target_csvfile_name = fix_filename(os.path.join(full_target_csvdir, hyperparameter_string+".csv"))
                    method_performance_df.to_csv(target_csvfile_name)
                    
                    full_target_scoredir = os.path.join(score_csvdir, dataset_name.replace("."+input_type, ""), method_name)
                    os.makedirs(full_target_scoredir, exist_ok=True)
                    target_scorefile_name = fix_filename(os.path.join(full_target_scoredir, hyperparameter_string+".csv"))
                    np.savetxt(target_scorefile_name, outlier_scores)
                    
                    #write Keras history for relevant neural methods
                    if method_name in ["VAE", "beta-VAE", "AE", "AnoGAN", "ALAD"]:
                        if method_name == "AnoGAN":
                            history_df = pd.DataFrame({"discriminator_loss":pipeline[1].hist_loss_discriminator, "generator_loss":pipeline[1].hist_loss_generator})
                        elif method_name =="ALAD":
                            history_df = pd.DataFrame({"discriminator_loss":pipeline[1].hist_loss_disc, "generator_loss":pipeline[1].hist_loss_gen})
                        else:
                            history = pipeline[1].history_
                            history_df = pd.DataFrame(history)
                        
                        full_target_dir = os.path.join(log_dir, dataset_name.replace("."+input_type, ""), method_name)
                        target_file_name = fix_filename(os.path.join(log_dir, dataset_name.replace("."+input_type, ""), method_name, hyperparameter_string+"."+input_type))
                        
                        os.makedirs(full_target_dir, exist_ok=True)
                        with open(target_file_name, 'wb') as handle:
                            pickle.dump(history_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
                        full_target_dir = os.path.join(log_dir, dataset_name.replace("."+input_type, ""), method_name)
                        target_file_name = fix_filename(os.path.join(log_dir, dataset_name.replace("."+input_type, ""), method_name, hyperparameter_string+".csv"))
                        
                        os.makedirs(full_target_dir, exist_ok=True)
    
                        history_df.to_csv(target_file_name)

