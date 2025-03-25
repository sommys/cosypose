"""Configuration of the BOP Toolkit."""
# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

from cosypose.config import LOCAL_DATA_DIR

######## Basic ########

# Folder with the BOP datasets.
datasets_path = str(LOCAL_DATA_DIR / 'bop_datasets')

# Folder with pose results to be evaluated.
results_path = str(LOCAL_DATA_DIR / '/bop_predictions_csv')

# Folder for the calculated pose errors and performance scores.
eval_path = str(LOCAL_DATA_DIR / 'bop_eval_outputs')

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r'/path/to/output/folder'

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/path/to/bop_renderer/build'

# Executable of the MeshLab server.
meshlab_server_path = r'/path/to/meshlabserver.exe'
