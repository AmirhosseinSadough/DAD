# main.py
import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Execute scripts from the 'runs' directory with specified dataset and result paths.")
    parser.add_argument('--data_category', type=str, default='benchmark', choices=['benchmark', 'synthetic', 'damadics'], help='Specify the category of the dataset to use.')
    parser.add_argument('--benchmark_method', type=str, default='default', choices=['hpt', 'maximum', 'average', 'default'], help='Specify the method for the benchmark category. Required if data_category is "benchmark".')
    parser.add_argument('--exclude_methods', type=str, default='', help='Comma-separated list of methods to exclude from the analysis.')
    parser.add_argument('--exclude_datasets', type=str, default='', help='Comma-separated list of datasets to exclude from the analysis.')

    args = parser.parse_args()

    # Construct the paths
    formatted_data_dir = os.path.join("datasets", args.data_category)
    if args.data_category == 'benchmark':
        if args.benchmark_method == 'maximum' or args.benchmark_method == 'average':
            base_result_dir = os.path.join("results", "benchmark", "max_mean")
        else:
            base_result_dir = os.path.join("results", "benchmark", args.benchmark_method)
    elif args.data_category == 'damadics':
        base_result_dir = os.path.join("results", "damadics", "max_mean")
    elif args.data_category == 'synthetic':
        base_result_dir = os.path.join("results", "synthetic", "max_mean")

    script_name = 'produce_fig_tab.py'
    # Construct the full path to the script
    script_path = os.path.join('runs', script_name)

    # Check if the script exists before attempting to execute
    if not os.path.exists(script_path):
        print(f"Error: The script '{script_path}' does not exist.")
        return

    # Execute the script with the constructed paths as arguments
    # print(f"Executing {script_path} with dataset path '{formatted_data_dir}' and result path '{base_result_dir}'...")
    subprocess.run(["python", script_path,                    
                    f'--base_result_dir={base_result_dir}',
                    f'--dataset={args.data_category}',
                    f'--eval_mode={args.benchmark_method}',
                    f'--exclude_methods={args.exclude_methods}',
                    f'--exclude_datasets={args.exclude_datasets}'])

if __name__ == "__main__":
    main()

