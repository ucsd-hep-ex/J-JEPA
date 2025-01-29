import click
import os
import re
import numpy as np

def extract_highest_imtafe(logfile_path):
    """
    Extracts all IMTAFE values from the logfile and returns the highest one.
    """
    imtafe_values = []
    float_pattern = r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+'

    with open(logfile_path, 'r') as file:
        for line in file:
            if 'IMTAFE:' in line:
                match = re.search(r'IMTAFE:\s*(' + float_pattern + r')', line)
                if match:
                    imtafe = float(match.group(1))
                    imtafe_values.append(imtafe)
    if imtafe_values:
        return max(imtafe_values)
    else:
        return None

@click.command()
@click.option('--out-dir', type=click.Path(exists=True, file_okay=False, readable=True), required=True, help='Output directory containing trial subdirectories.')
def main(out_dir):
    highest_imtafe_list = []

    # Iterate over all trial directories
    for trial_dir in os.listdir(out_dir):
        trial_path = os.path.join(out_dir, trial_dir)
        if os.path.isdir(trial_path) and trial_dir.startswith('trial-'):
            logfile_path = os.path.join(trial_path, 'logfile.txt')
            if os.path.isfile(logfile_path):
                highest_imtafe = extract_highest_imtafe(logfile_path)
                if highest_imtafe is not None:
                    highest_imtafe_list.append(highest_imtafe)
                else:
                    print(f"No IMTAFE values found in {logfile_path}")
            else:
                print(f"logfile.txt not found in {trial_path}")

    if highest_imtafe_list:
        mean_imtafe = np.mean(highest_imtafe_list)
        std_imtafe = np.std(highest_imtafe_list)
        print(highest_imtafe_list)
        print(f"Mean of highest IMTAFE values: {mean_imtafe}")
        print(f"Standard deviation: {std_imtafe}")
    else:
        print("No IMTAFE values were found in any trial.")

if __name__ == '__main__':
    main()

