import pandas as pd
import os

def display_results(csv_path, output_mode='terminal', output_file='results.txt'):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Add extra spacing between columns by adjusting column spacing manually
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.width', 200)

    # Group by Test Name and k
    grouped = df.groupby(['Test Name', 'k'])

    output_lines = []

    for (test_name, k), group in grouped:
        output_lines.append(f"=== Test Name: {test_name} | k: {k} ===")
        # Format with extra spaces between columns manually
        formatted = group.to_string(index=False, col_space=15)
        output_lines.append(formatted)
        output_lines.append("\n")

    output_text = "\n".join(output_lines)

    if output_mode == 'terminal':
        print(output_text)
    elif output_mode == 'file':
        with open(output_file, 'w') as f:
            f.write(output_text)
        print(f"Results written to {output_file}")
    else:
        print("Invalid output mode. Use 'terminal' or 'file'.")


if __name__ == "__main__":
    results_dir = 'results'
    output_dir = 'results_display'
    os.makedirs(output_dir, exist_ok=True)

    file_names = [f.split('.')[0] for f in os.listdir(results_dir) if f.endswith('.csv')]

    for file_name in file_names:
        csv_path = f'{results_dir}/{file_name}.csv'
        output_file = f'{output_dir}/{file_name}.txt'
        display_results(csv_path, output_mode='file', output_file=output_file) 
