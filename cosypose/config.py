import os
from joblib import Memory
from pathlib import Path
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

COSYPOSE_HOME = os.environ.get('COSYPOSE_HOME', None)
PROJECT_ROOT = Path(COSYPOSE_HOME) if COSYPOSE_HOME is not None else Path(__file__).parent.parent
PROJECT_DIR = PROJECT_ROOT
DATA_DIR = PROJECT_DIR / 'data'
LOCAL_DATA_DIR = PROJECT_DIR / 'local_data'
TEST_DATA_DIR = LOCAL_DATA_DIR
DASK_LOGS_DIR = LOCAL_DATA_DIR / 'dasklogs'
SYNT_DS_DIR = LOCAL_DATA_DIR / 'synt_datasets'
BOP_DS_DIR = LOCAL_DATA_DIR / 'bop_datasets'

BOP_TOOLKIT_DIR = PROJECT_DIR / 'deps' / 'bop_toolkit_cosypose'
BOP_CHALLENGE_TOOLKIT_DIR = PROJECT_DIR / 'deps' / 'bop_toolkit_challenge'

EXP_DIR = LOCAL_DATA_DIR / 'experiments'
RESULTS_DIR = LOCAL_DATA_DIR / 'results'
DEBUG_DATA_DIR = LOCAL_DATA_DIR / 'debug_data'

DEPS_DIR = PROJECT_DIR / 'deps'
CACHE_DIR = LOCAL_DATA_DIR / 'joblib_cache'

assert LOCAL_DATA_DIR.exists()
CACHE_DIR.mkdir(exist_ok=True)
TEST_DATA_DIR.mkdir(exist_ok=True)
DASK_LOGS_DIR.mkdir(exist_ok=True)
SYNT_DS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DEBUG_DATA_DIR.mkdir(exist_ok=True)

ASSET_DIR = DATA_DIR / 'assets'
MEMORY = Memory(CACHE_DIR, verbose=2)
