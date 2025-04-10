# main.py
import argparse
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Execute scripts from the 'runs' directory with specified dataset and result paths.")
    parser.add_argument('category', choices=['benchmark', 'synthetic', 'damadics'],
                        help="Specify the category of the script to run.")
    parser.add_argument('method', nargs='?', choices=['hpt', 'max_mean', 'default'],
                        help="Specify the method for the benchmark category. Not required for 'synthetic' or 'damadics' categories.")

    args = parser.parse_args()

    # Determine the script name based on category and method
    if args.category == 'benchmark':
        if args.method is None:
            parser.error("The 'benchmark' category requires a method argument ('hpt', 'max_mean', or 'default').")
        script_name = f"run_benchmark_{args.method}.py"
    elif args.category == 'synthetic':
        script_name = "run_synthetic.py"
    elif args.category == 'damadics':
        script_name = "run_damadics.py"
    else:
        parser.error("Invalid category or method combination.")

    # Construct the paths
    formatted_data_dir = os.path.join("datasets", args.category)
    if args.category == 'benchmark':
        base_result_dir = os.path.join("results", "benchmark", args.method)
    elif args.category == 'damadics':
        base_result_dir = os.path.join("results", "damadics", "max_mean")
    elif args.category == 'synthetic':
        base_result_dir = os.path.join("results", "synthetic", "max_mean")

    # Construct the full path to the script
    script_path = os.path.join('runs', script_name)

    # Check if the script exists before attempting to execute
    if not os.path.exists(script_path):
        print(f"Error: The script '{script_path}' does not exist.")
        return

    # Execute the script with the constructed paths as arguments
    print(f"Executing {script_path} with dataset path '{formatted_data_dir}' and result path '{base_result_dir}'...")
    # subprocess.run(["python", script_path, formatted_data_dir, base_result_dir])
    subprocess.run(["python", script_path, 
                    f'--formatted_data_dir={formatted_data_dir}', 
                    f'--base_result_dir={base_result_dir}'])

if __name__ == "__main__":
    main()
