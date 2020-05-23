from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment()
from sacred import SETTINGS
SETTINGS['HOST_INFO']['INCLUDE_GPU_INFO'] = False
SETTINGS.HOST_INFO.INCLUDE_GPU_INFO = False  # equivalent
# SETTINGS['CAPTURE_MODE']='no'
@ex.config
def my_config():
    """This is my demo configuration"""

    training_params = {'a': 10, 'b': 20}
    c = 5
    d = 6


@ex.capture
def capture(c, d):
    print("capture ", c, d)


class Model:
    def __init__(self):
        pass


@ex.capture
def get_model():
    return Model()


@ex.main
def my_main(training_params, _config, _run):
    print(training_params)
    print(_config)
    print(_run)
    with open("my_artifact.txt","w") as fptr:
        fptr.write("aaaa")
    _run.add_artifact("my_artifact.txt")
    for i in range(10):
        _run.log_scalar("step_n",i, step=i)

    return {'cc': 5}


if __name__ == '__main__':
    ex.observers.append(FileStorageObserver('my_runs'))
    r = ex.run()
    print(r.result)
    m1= get_model()
    print(m1)
    m2 = get_model()
    print(m2)
    print(r.config)
    print(capture())

    print(r.experiment_info)
    print(r.meta_info)
    print(r.info)
    print(r.run_logger, r.observers[0].run_entry['artifacts'], r.observers[0].dir)
