from sacred.observers import FileStorageObserver

from src.eval import eval_dmm
from src.experiment import ex

if __name__ == "__main__":
    ex.observers.append(FileStorageObserver('/home/tylee/experiments'))
    r = ex.run()
    eval_dmm(r.observers[0].dir)
