import sys

from sacred.observers import FileStorageObserver

from src import logger
from src.eval import eval_dmm
from src.experiment import ex

if __name__ == "__main__":
    experiments_dir = sys.argv[1]
    logger.info(f"Writing experiments to {experiments_dir}")
    ex.observers.append(FileStorageObserver(experiments_dir))
    r = ex.run()
    eval_dmm(r.observers[0].dir)
