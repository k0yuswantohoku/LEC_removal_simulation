This repository contains the Supporting Information for the article "Reduction of endocytosis and EGFR signaling is associated with the switch from isolated to clustered apoptosis during epithelial tissue remodeling in Drosophila" by Yuswan, K., Sun, X., Kuranaga, E., and Umetsu, D. in PLOS Biology

DOI 10.5281/zenodo.13290047

The files include the Excel files of all the data points used to generate the figures in the article, and the python files for detailed analyses.

Following are the brief descriptions of the python files:

- "cluster_detection.py" The code uses the "Spots" output of the LEC tracking data from TRACKMATE Plugin in ImageJ/Fiji as the input. The cells, represented by ID labeled points (TRACK_ID column) are triangulated (Delaunay Triangulation) over time to detect pairs of cells which are eliminated at the same frame. The output .csv file is then used to select overlapping LEC pairs, and determine them as a clustered apoptosis.
- "poisson_simulation_trial_iteration.py" The code was built upon the "cluster_detection.py" to additionally generate the order of apoptosis randomly, from the random removals of TRACK_ID over time, based on the experimental rates of apoptosis. The triangulation and cluster detection is then conducted as described above.
