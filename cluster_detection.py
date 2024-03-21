import os
import pandas as pd

# Directory containing the files
directory = '/Users/XXXX/'

# Detect all files containing "unique_pairs_df" in their filenames
file_list = [file for file in os.listdir(directory) if "unique_pairs_df" in file]

# Initialize an empty DataFrame to store the combined results
combined_results_df = pd.DataFrame()

# Process each file
for file in file_list:
    # Load the DataFrame from the file
    filepath = os.path.join(directory, file)
    unique_pairs_df = pd.read_csv(filepath)

    # Ignore rows where either Cell1_Max_Frame or Cell2_Max_Frame is equal to the maximum frame in the DataFrame
    max_frame = unique_pairs_df['FRAME'].max()
    unique_pairs_df = unique_pairs_df[(unique_pairs_df['Cell1_Max_Frame'] != max_frame) & 
                                      (unique_pairs_df['Cell2_Max_Frame'] != max_frame)]

    # Initialize an empty list to store clustered elimination pairs
    clustered_elimination_pairs = []

    # Iterate through each pair of cells
    for index, row in unique_pairs_df.iterrows():
        # Check if either cell ID appears more than once in the DataFrame
        if (unique_pairs_df['Cell1_ID'] == row['Cell1_ID']).sum() > 1 or \
           (unique_pairs_df['Cell1_ID'] == row['Cell2_ID']).sum() > 1 or \
           (unique_pairs_df['Cell2_ID'] == row['Cell1_ID']).sum() > 1 or \
           (unique_pairs_df['Cell2_ID'] == row['Cell2_ID']).sum() > 1:
            # Add the pair to the list of clustered elimination pairs
            clustered_elimination_pairs.append(row)

    # Create a DataFrame from the clustered elimination pairs
    clustered_apo = pd.DataFrame(clustered_elimination_pairs)

    # Save individual results
    clustered_apo.to_csv(os.path.join(directory, file.replace("unique_pairs_df", "clustered_apo")), index=False)

    # Create a new DataFrame entry for the combined results
    if not clustered_apo.empty:
        clustered_apo['filename'] = file  # Add filename column
        combined_results_df = combined_results_df.append(clustered_apo, ignore_index=True)
    else:
        # Add a row with NaN values if no cluster apoptosis is detected
        nan_row = pd.Series([pd.NA] * len(unique_pairs_df.columns), index=unique_pairs_df.columns)
        nan_row['filename'] = file
        combined_results_df = combined_results_df.append(nan_row, ignore_index=True)

# Save the combined results DataFrame
combined_results_df.to_csv(os.path.join(directory, 'combined_results.csv'), index=False)
