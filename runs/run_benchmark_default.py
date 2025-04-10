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


result_dir = "result_dir"
csvresult_dir = "csvresult_dir"
score_dir = "score_dir"
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
# from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
# from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN 
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.sod import SOD
from pyod.models.ecod import ECOD
from pyod.models.lunar import LUNAR
# from pyod.models.so_gaal import SO_GAAL
# from pyod.models.deep_svdd  import DeepSVDD
# from pyod.models.auto_encoder  import auto_encoder
# from pyod.models.vae  import VAE

 

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
# from additional_methods.sod import SOD
# from additional_methods.lmdd import LMDD
from additional_methods.abod import ABOD



from additional_methods.wrappers.DECODE_Auto import DECODE_Auto


ensemble_LOF_krange = range(5,31,3)

#dict of methods and functions
method_classes = {
        "DECODE_Auto":DECODE_Auto,
        "INNE":INNE,
        "GMM":GMM,
        "KDE":KDE,
        "ABOD":ABOD, 
        "CBLOF":CBLOF,
        "COF":COF,
        "COPOD":COPOD,
        "HBOS":HBOS,
        "kNN":KNN,
        "IF":IForest,
        "LMDD":LMDD,
        "LODA":LODA,
        "LOF":LOF,
        "MCD":MCD,
        "OCSVM":OCSVM,
        "PCA":PCA, 
        "SOD":SOD,
        "ECOD":ECOD,
        "LUNAR":LUNAR,
        #------------------------- Implemented methods -------------------------
        "ensemble-LOF":Ensemble,
        "EIF":ExtendedIForest,
        "ODIN":ODIN,        
        "gen2out":gen2Out,
        "DynamicHBOS":DynamicHBOS
        }

#dict of methods and parameters
method_parameters = {
        # -------------- default settings -----------------
        "DECODE_Auto":{},  
        "INNE":{},
        "GMM":{},
        "KDE":{},
        "ABOD":{"method":["fast"], "n_neighbors":[10]}, 
        "CBLOF":{},
        "COF":{"n_neighbors":[20]},
        "COPOD":{},
        "HBOS":{},
        "kNN":{},
        "IF":{},
        "LMDD":{},
        "LODA":{},
        "LOF":{},
        "MCD":{},      
        "OCSVM":{},
        "PCA":{},
        "SOD":{},
        "ECOD":{},
        "LUNAR":{},
        # -------------------------------------------------
        "ensemble-LOF":{},
        "EIF":{"n_estimators":[100], "max_samples":[256], "extension_level":[1]},
        "ODIN":{},
        "gen2out":{},
        "DynamicHBOS":{}
        }

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


target_dir = os.path.join(base_result_dir, result_dir)
target_csvdir = os.path.join(base_result_dir, csvresult_dir)
score_csvdir = os.path.join(base_result_dir, score_dir)
target_wcdir = os.path.join(base_result_dir, "wc_dir")
os.makedirs(target_wcdir, exist_ok=True)

if not os.path.exists(score_csvdir):
    os.makedirs(score_csvdir)
    
for dataset_name in dataset_names:
    
    #check if data path exists, and make it if it doesn't

    
    #print name for reporting purpose
    print("______"+dataset_name+"______")
    
    full_path_filename = os.path.join(formatted_data_dir, dataset_name)
    
    if input_type == "pickle":
        data = pickle.load(open(full_path_filename, 'rb'))
    elif input_type == "npz":
        data  = np.load(open(full_path_filename, 'rb'))
                    
    X, y = data["X"], np.squeeze(data["y"])
    
    max_duplicates = data["max_duplicates"]
    
    #loop over all methods:

    for method_name, OD_class in all_methods_to_run.items():
        print("-" + method_name)
        hyperparameter_grid = method_parameters[method_name]
        
        # #In case max_duplicates > k, these methods need an increase in k:
        # if method_name in ["LOF", "COF", "ABOD"] and \
        # max_duplicates >= min(hyperparameter_grid["n_neighbors"]):
                
        #     hyperparameter_grid["n_neighbors"] = [int(k+max_duplicates) for k in hyperparameter_grid["n_neighbors"]]
        # elif method_name ==  "ensemble-LOF":
        #     if max_duplicates >= min(ensemble_LOF_krange):
        #         temp_ensemble_LOF_krange = [int(k+max_duplicates) for k in ensemble_LOF_krange]
        #         hyperparameter_grid["estimators"] = [[LOF(n_neighbors=k) for k in temp_ensemble_LOF_krange]]
                
        hyperparameter_list = list(ParameterGrid(hyperparameter_grid))
        
        #loop over hyperparameter settings
        for hyperparameter_setting in hyperparameter_list:
            
            if method_name == "ensemble-LOF":
                # if max_duplicates >= min(ensemble_LOF_krange):
                #     #hyperparameter_string = str(temp_ensemble_LOF_krange)
                # else:                    
                hyperparameter_string = str(ensemble_LOF_krange)
            else:
                hyperparameter_string = str(hyperparameter_setting)
                
            if verbose:
                print(hyperparameter_string)
            
            #check whether results have  been calculated
            full_target_dir = os.path.join(target_dir, dataset_name.replace("."+input_type, ""), method_name)
            target_file_name = fix_filename(os.path.join(target_dir, dataset_name.replace("."+input_type, ""), method_name, hyperparameter_string+".pickle"))
            
            if os.path.exists(target_file_name) and os.path.getsize(target_file_name) > 0:
                if verbose:
                    print(" results already calculated, skipping recalculation")
            elif method_name == "EIF" and X.shape[1] <= hyperparameter_setting["extension_level"]:
                print("Dimensionality of dataset higher than EIF extension level, skipping...")
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
                    
                    # DeepSVDD_argument_list.append(str(hyperparameter_setting["n_layers"]))
                    # DeepSVDD_argument_list.append(str(hyperparameter_setting["shrinkage_factor"]))
                    
                    # DeepSVDD_argument_list.append(os.path.join("..", "log", dataset_name))
                    # DeepSVDD_argument_list.append(os.path.join(DeepSVDD_dir, "data"))
                    
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
                    
                    OD_method = OD_class(**hyperparameter_setting)
                    
                    #Temporary fix for ECOD:
                    if method_name == "ECOD" and hasattr(OD_method, "X_train"):
                        delattr(OD_method, "X_train")
                        
                    start_time = time.time()
                    
                    if method_name == "DECODE_Auto":
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

