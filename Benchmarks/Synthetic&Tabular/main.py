# main.py
import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Execute scripts from the 'runs' directory with specified dataset and result paths.")
    parser.add_argument('--data_category', type=str, default='benchmark', choices=['benchmark', 'synthetic', 'damadics'], help='Specify the category of the dataset to use.')
    parser.add_argument('--benchmark_method', type=str, default='default', choices=['hpt', 'max_mean', 'default'], help='Specify the method for the benchmark category. Required if data_category is "benchmark".')
    parser.add_argument('--method', type=str, default='DAD_Auto', help='Specify the method to use.')
    parser.add_argument('--dataset', type=str, default='all', help='Specify the dataset to use (e.g., hepatitis, synthetic_1, damadics_1).')

    args = parser.parse_args()
    # Determine the script name based on category and method
    if args.data_category == 'benchmark':
        if args.benchmark_method is None:
            parser.error("The 'benchmark' category requires a method argument ('hpt', 'max_mean', or 'default').")
        script_name = f"run_benchmark_{args.benchmark_method}.py"
    elif args.data_category == 'synthetic':
        script_name = "run_synthetic.py"
    elif args.data_category == 'damadics':
        script_name = "run_damadics.py"
    else:
        parser.error("Invalid category or method combination.")

    # Construct the paths
    formatted_data_dir = os.path.join("datasets", args.data_category)
    if args.data_category == 'benchmark':
        base_result_dir = os.path.join("results", "benchmark", args.benchmark_method)
    elif args.data_category == 'damadics':
        base_result_dir = os.path.join("results", "damadics", "max_mean")
    elif args.data_category == 'synthetic':
        base_result_dir = os.path.join("results", "synthetic", "max_mean")

    # Construct the full path to the script
    script_path = os.path.join('runs', script_name)

    # Check if the script exists before attempting to execute
    if not os.path.exists(script_path):
        print(f"Error: The script '{script_path}' does not exist.")
        return

    # Execute the script with the constructed paths as arguments
    print(f"Executing {script_path} with dataset path '{formatted_data_dir}' and result path '{base_result_dir}'...")
    subprocess.run(["python", script_path, 
                    f'--formatted_data_dir={formatted_data_dir}', 
                    f'--base_result_dir={base_result_dir}',
                    f'--method={args.method}', 
                    f'--dataset={args.dataset}'])

if __name__ == "__main__":
    main()
