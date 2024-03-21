import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import norm
from tqdm import tqdm
from scipy.spatial import Delaunay
import timeit
import random
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.widgets import Slider
import matplotlib.tri as tri
import os
from sklearn.cluster import MeanShift

lec_df = pd.read_csv("/Users/kevin-hgd/Documents/SP5/pnrGAL4 Crosses/minicic-SCAT3/shiTS/oldtrack/shiTS_minicicmscarlet_008.csv", sep=',', index_col=None)


# Simulated Data (Replace with your actual data)
lec_full_df = lec_df[["TRACK_ID", "POSITION_X", "POSITION_Y", "FRAME"]]

# Filter data for early phase (frames 6-24)
early_phase = lec_full_df[(lec_full_df["FRAME"] >= 6) & (lec_full_df["FRAME"] <= 24)]
late_phase = lec_full_df[(lec_full_df["FRAME"] >= 31) & (lec_full_df["FRAME"] <= 75)]

#specify early or late here
full_df = early_phase

remaining_lecs = full_df["FRAME"].value_counts().sort_index()
print(list(remaining_lecs))

#lec_full_df = experimental_data_full.loc[experimental_data_full['FRAME'] == 0].sort_values(by=['TRACK_ID'])
#print(experimental_data_f0)


remaining_cells_list = remaining_lecs.tolist()
#[146, 146, 145, 145, 144, 143, 142, 141, 139, 138, 138, 138, 135, 134, 133, 132, 131, 131, 130]
#experimental_data_full["FRAME"].value_counts().sort_index()


# Function to calculate the number of cell deaths in each frame compared to the previous frame
def calculate_cell_deaths(remaining_cells_list):
    cell_deaths_list = [0]  # Initialize with 0 cell deaths for the first frame
    
    for i in range(1, len(remaining_cells_list)):
        cell_deaths = remaining_cells_list[i - 1] - remaining_cells_list[i]
        cell_deaths_list.append(cell_deaths)
    
    return cell_deaths_list

# Example usage
cell_deaths_list = calculate_cell_deaths(remaining_cells_list)
print("Number of cell deaths in each frame:", cell_deaths_list)

# Define the frame range
frame_range = full_df['FRAME'].unique()  # Frame range from 42 to 60

# Get the minimum and maximum frame numbers
min_frame = min(frame_range)
max_frame = max(frame_range)

# Function to replicate TRACK_ID and XY positions for all frames in a specified range
def replicate_data_for_frames(full_df, frame_range):
    data_for_min_frame = full_df[full_df['FRAME'] == min_frame]  # Extract data for the minimum frame
    replicated_df = pd.DataFrame()  # Initialize an empty DataFrame to store replicated data
    
    # Duplicate the data for each frame in the specified range
    for frame in frame_range:
        data_for_frame = data_for_min_frame.copy()  # Copy data for the minimum frame
        data_for_frame['FRAME'] = frame  # Update the frame number
        replicated_df = pd.concat([replicated_df, data_for_frame], ignore_index=True)  # Concatenate with replicated_df
    
    return replicated_df


# Replicate data for the specified frame range
replicated_df = replicate_data_for_frames(full_df, frame_range)
print(replicated_df)


   
# Function to simulate cell removal based on remaining cells list
def simulate_cell_removal(replicated_df, cell_deaths_list):
    # Initialize an empty DataFrame to store the result
    result_df = pd.DataFrame(columns=['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'FRAME'])
    removed_cells = set()  # Set to store removed cells
    
    # Iterate over cell_deaths_list and remove cells from full_df
    for frame, num_cells_to_remove in enumerate(cell_deaths_list):
#### MAKE SURE TO CHANGE START FRAME ####        
        frame += 6  # Adjust frame index to start from 6 for early and 31 for late
        
        # Get the cells at the current frame
        cells_at_frame = replicated_df[(replicated_df['FRAME'] == frame) & (~replicated_df['TRACK_ID'].isin(removed_cells))]
        
        # Remove num_cells_to_remove random cells from cells_at_frame
        if num_cells_to_remove > 0:
            cells_to_remove = cells_at_frame.sample(n=num_cells_to_remove)
            removed_cells.update(cells_to_remove['TRACK_ID'])  # Update set with removed cells
            result_df = pd.concat([result_df, cells_to_remove[['TRACK_ID', 'POSITION_X', 'POSITION_Y', 'FRAME']]], ignore_index=True)
    
    # Sort by frame
    result_df = result_df.sort_values(by='FRAME').reset_index(drop=True)
    
    return result_df

# Function to remove cells from full_df based on cell removal simulation
def remove_cells(modified_df, result_df):
    removed_indices = set()  # Initialize a set to store the indices of removed cells

    # Iterate over each row in result_df and mark corresponding cells for removal
    for index, row in result_df.iterrows():
        track_id = row['TRACK_ID']
        frame = row['FRAME']
        print(f"Removing cell with TRACK_ID {track_id} on frame {frame}")
        # Find the cells to remove using boolean indexing
        cells_to_remove = (modified_df['TRACK_ID'] == track_id) & (modified_df['FRAME'] == frame)
        # Add the indices of removed cells to the set
        removed_indices.update(modified_df.index[cells_to_remove])

    print("Indices of cells marked for removal:", removed_indices)

    # Filter out removed cells for all frames beyond the initial removal frame
    for frame in range(result_df['FRAME'].min(), result_df['FRAME'].max() + 1):
        frame_removed_indices = set(modified_df[(modified_df['FRAME'] == frame) & modified_df.index.isin(removed_indices)].index)
        removed_indices.update(frame_removed_indices)

    print("Indices of cells marked for removal after filtering frames:", removed_indices)

    # Also remove XY positions for removed cells in subsequent frames
    for idx in removed_indices:
        if idx in modified_df.index:  # Check if index exists before accessing
            track_id = modified_df.loc[idx, 'TRACK_ID']
            frame = modified_df.loc[idx, 'FRAME']
            modified_df.loc[(modified_df['TRACK_ID'] == track_id) & (modified_df['FRAME'] >= frame), ['POSITION_X', 'POSITION_Y']] = np.nan

    return modified_df

# Example usage
# Assuming full_df and cell_deaths_list are defined

# Simulate cell removal
result_df = simulate_cell_removal(replicated_df, cell_deaths_list)

# Display the result DataFrame
print(result_df)

# Call the function to remove cells from full_df based on the simulation
modified_df = remove_cells(replicated_df, result_df)

print("\nModified full_df:")
print(modified_df)

# Function to update the plot for each frame
def update(frame):
    plt.cla()
    plt.scatter(modified_df[modified_df['FRAME'] == frame]['POSITION_X'], 
                modified_df[modified_df['FRAME'] == frame]['POSITION_Y'],
                marker='o',
                color='blue',
                label='Cells')
    plt.xlim(0, 1024)
    plt.ylim(0, 1024)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title(f'Frame {frame}')
    plt.legend()

# Create the initial plot
#fig, ax = plt.subplots()

# Define the animation
#ani = FuncAnimation(fig, update, frames=range(min_frame, max_frame+1), interval=200)

#plt.show()


# Drop rows with NaN values
modified_df = modified_df.dropna()

# Reset the index after dropping rows
modified_df = modified_df.reset_index(drop=True)

# Find the minimum frame value
min_frame = modified_df['FRAME'].min()

# Filter the data for the first frame
first_frame_data = modified_df[modified_df['FRAME'] == min_frame][["TRACK_ID", "POSITION_X", "POSITION_Y"]]

# Debugging - print the number of samples in the first frame
print("Number of samples in the first frame:", len(first_frame_data))

if len(first_frame_data) == 0:
    print("No data available for clustering in the first frame.")
    exit()

# Perform mean shift clustering on the first frame along the Y-axis
bandwidth = 100  # Adjust the bandwidth parameter
ms = MeanShift(bandwidth=bandwidth)
ms.fit(first_frame_data[["POSITION_Y"]])

# Add cluster labels to modified_df
cluster_labels = ms.labels_
first_frame_data['cluster_label'] = cluster_labels
modified_df = modified_df.merge(first_frame_data[['TRACK_ID', 'cluster_label']], on='TRACK_ID', how='left')

# Function to perform triangulation for a given cluster and frame
def perform_triangulation(frame_data):
    points = frame_data[["POSITION_X", "POSITION_Y", "TRACK_ID"]].values
    tri = Delaunay(points[:, :2])  # Considering only X and Y positions for triangulation
    
    # Get indices of triangles
    tri_indices = tri.simplices

    # Create a DataFrame to store pairs of connected cells with their IDs, positions, and cluster assignment
    connected_cells = []
    for simplex in tri_indices:
        for i in range(3):
            cell1_id = int(points[simplex[i], 2])  # Get the ID of the first cell
            cell2_id = int(points[simplex[(i + 1) % 3], 2])  # Get the ID of the second cell
            cell1_x = frame_data.loc[frame_data['TRACK_ID'] == cell1_id, 'POSITION_X'].iloc[0]
            cell1_y = frame_data.loc[frame_data['TRACK_ID'] == cell1_id, 'POSITION_Y'].iloc[0]
            cell2_x = frame_data.loc[frame_data['TRACK_ID'] == cell2_id, 'POSITION_X'].iloc[0]
            cell2_y = frame_data.loc[frame_data['TRACK_ID'] == cell2_id, 'POSITION_Y'].iloc[0]
            cluster_label = frame_data['cluster_label'].iloc[0]  # Assuming all cells in frame_data have the same cluster label
            frame = frame_data['FRAME'].iloc[0]  # Assuming all cells in frame_data have the same frame
            connected_cells.append((cell1_id, cell1_x, cell1_y, cell2_id, cell2_x, cell2_y, cluster_label, frame))

    # Convert the list of connected cells to DataFrame
    connected_cells_df = pd.DataFrame(connected_cells, columns=['Cell1_ID', 'Cell1_X', 'Cell1_Y', 'Cell2_ID', 'Cell2_X', 'Cell2_Y', 'Cluster_Label', 'FRAME'])
    
    return connected_cells_df

# Create a DataFrame to store triangulation information
triangulation_df = pd.DataFrame(columns=['Cell1_ID', 'Cell1_X', 'Cell1_Y', 'Cell2_ID', 'Cell2_X', 'Cell2_Y', 'Cluster_Label', 'FRAME'])

# Iterate over frames and clusters to perform triangulation
frames = modified_df['FRAME'].unique()
clusters = modified_df['cluster_label'].unique()
total_iterations = len(frames) * len(clusters)

with tqdm(total=total_iterations, desc='Triangulating', unit=' iteration') as pbar:
    for frame in frames:
        frame_data = modified_df[modified_df['FRAME'] == frame]
        for cluster_label in clusters:
            cluster_data = frame_data[frame_data['cluster_label'] == cluster_label]
            triangulation_cluster = perform_triangulation(cluster_data)
            triangulation_df = pd.concat([triangulation_df, triangulation_cluster])
            pbar.update(1)

# Display the triangulation DataFrame
print(triangulation_df.head())


# Create a new DataFrame containing "TRACK_ID" and "FRAME" from modified_df
cell_frame_df = modified_df[['TRACK_ID', 'FRAME']].copy()

# Add a new column to cell_frame_df which takes the max value of the frame for every cell
cell_frame_df['Max_Frame'] = cell_frame_df.groupby('TRACK_ID')['FRAME'].transform('max')

print(cell_frame_df)

# Merge triangulation_df and last_frames on 'Cell1_ID' to get the max frame for Cell1
triangulation_df = triangulation_df.merge(cell_frame_df.rename(columns={'TRACK_ID': 'Cell1_ID', 'Max_Frame': 'Cell1_Max_Frame'}), on='Cell1_ID', how='left')

# Merge triangulation_df and last_frames on 'Cell2_ID' to get the max frame for Cell2
triangulation_df = triangulation_df.merge(cell_frame_df.rename(columns={'TRACK_ID': 'Cell2_ID', 'Max_Frame': 'Cell2_Max_Frame'}), on='Cell2_ID', how='left')

# Display the updated triangulation DataFrame
print(triangulation_df[['FRAME_x', 'FRAME_y']].head())


# Filter the triangulation_df to include only rows where max frame of Cell1 equals max frame of Cell2
equal_max_frames_df = triangulation_df[triangulation_df['Cell1_Max_Frame'] == triangulation_df['Cell2_Max_Frame']]

# Display the new DataFrame
print(equal_max_frames_df)

# Select only the necessary columns from the triangulation_df
unique_pairs_df = equal_max_frames_df[['Cell1_ID', 'Cell2_ID', 'Cell1_X', 'Cell1_Y', 'Cell2_X', 'Cell2_Y', 'FRAME', 'Cell1_Max_Frame', 'Cell2_Max_Frame']]

# Sort the IDs to ensure uniqueness
unique_pairs_df[['Cell1_ID', 'Cell2_ID']] = pd.DataFrame(np.sort(unique_pairs_df[['Cell1_ID', 'Cell2_ID']], axis=1), index=unique_pairs_df.index)
unique_pairs_df = unique_pairs_df[unique_pairs_df['FRAME'] == unique_pairs_df['Cell1_Max_Frame']]

# Drop duplicate rows to keep only unique pairs
unique_pairs_df.drop_duplicates(subset=['Cell1_ID', 'Cell2_ID'], inplace=True)


# Reset the index
unique_pairs_df.reset_index(drop=True, inplace=True)

# Display the new DataFrame
print(unique_pairs_df)



# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(10.24, 8.00))
ax.invert_yaxis()  # Invert Y axis
ax.set_xlim(0, 1024)  # Set X axis limits
ax.set_ylim(0, 720)  # Set Y axis limits
ax.set_aspect('equal')  # Set aspect ratio to be equal
ax.xaxis.set_ticks_position('bottom')  # Set X ticks position to bottom

# Plot invisible points with labels for the legend
ax.scatter([], [], color='red', label='Concurrently Dying Cell', alpha=0)

# Function to update the plot for each frame
def update(frame):
    ax.clear()

    # Plot XY coordinates of cells for the current frame
    frame_data = modified_df[modified_df['FRAME'] == frame]
    ax.scatter(frame_data['POSITION_X'], frame_data['POSITION_Y'], label=f'Frame {frame}', color='blue', s=30)

    # Plot pairs of cells with concurrent death, ignoring last frame
    if frame != modified_df['FRAME'].max():
        frame_pairs_data = unique_pairs_df[unique_pairs_df['FRAME'] == frame]
        for i, row in frame_pairs_data.iterrows():
            if frame == row['Cell1_Max_Frame']:
                ax.scatter(row['Cell1_X'], row['Cell1_Y'], color='red', zorder=3, linewidth=2, s=80)
            if frame == row['Cell2_Max_Frame']:
                ax.scatter(row['Cell2_X'], row['Cell2_Y'], color='red', zorder=3, linewidth=2, s=80)

            # Plot line connecting the pair at the exact frames
            if frame == row['Cell1_Max_Frame'] and frame == row['Cell2_Max_Frame']:
                ax.plot([row['Cell1_X'], row['Cell2_X']], [row['Cell1_Y'], row['Cell2_Y']], color='red', zorder=2, linewidth=2)

    plt.xlabel('POSITION_X')
    plt.ylabel('POSITION_Y')
    plt.title(f'Cell Movement with Concurrently Dying Pairs (Frame {frame})')

    # Invert Y axis
    ax.invert_yaxis()

# Create the animation
ani = FuncAnimation(fig, update, frames=modified_df['FRAME'].unique(), interval=200)

# Display the legend
legend = plt.legend(loc='upper right', bbox_to_anchor=(0, 1), frameon=False)
plt.setp(legend.get_texts(), color='black')  # Set legend text color to black

#plt.show()

# Save the dataframes
######DONT FORGET TO CHANGE TO THE CORRECT FILES OR SUFFER THE OVERWRITING######
modified_df.to_csv('/Users/kevin-hgd/Documents/SP5/pnrGAL4 Crosses/minicic-SCAT3/Control/cluster_poisson/shi/shi008/modified_df_shi008_early006.csv', index=False)
#triangulation_df.to_csv('/Users/kevin-hgd/Documents/SP5/pnrGAL4 Crosses/minicic-SCAT3/Control/cluster_poisson/control/triangulation_df_ctrl_003_late001.csv', index=False)
unique_pairs_df.to_csv('/Users/kevin-hgd/Documents/SP5/pnrGAL4 Crosses/minicic-SCAT3/Control/cluster_poisson/shi/shi008/unique_pairs_df_shi008_early006.csv', index=False)

# Save the animation as a video file
#ani.save('/Users/kevin-hgd/Documents/SP5/pnrGAL4 Crosses/minicic-SCAT3/Control/cluster_poisson/control/animation_ctrl_003_late001.mp4', writer='ffmpeg')
ani.save('/Users/kevin-hgd/Documents/SP5/pnrGAL4 Crosses/minicic-SCAT3/Control/cluster_poisson/shi/shi008/animation_shi008_early006_frame_{:04d}.png', writer='pillow', dpi=200)
ani.save('/Users/kevin-hgd/Documents/SP5/pnrGAL4 Crosses/minicic-SCAT3/Control/cluster_poisson/shi/shi008/animation_shi008_early006.gif', writer='pillow', fps=5)

# Display a message indicating successful saving
print("Dataframes and animation saved successfully.")