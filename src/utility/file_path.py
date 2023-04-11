import os.path

DIRECTORY_NAME = "GeneAnalysis"
DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split(os.sep)
INDEX_DIRECTORY_NAME = REPO_PATH.index(DIRECTORY_NAME)
REPO_PATH = os.sep.join(REPO_PATH[:INDEX_DIRECTORY_NAME + 1])

# Path to sub-directories
DATASET_PATH = os.path.join(REPO_PATH, 'data')
RESULT_PATH = os.path.join(REPO_PATH, 'result')
