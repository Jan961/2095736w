import pickle
from typing import List, Dict
import os
from pathlib import Path


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


class ComparisonBase:
    all_datasets_list = ('cube', 'mnist', 'coil20', 'rna N3k', 'airfoil', 'wine quality',
                         'fashion mnist', 'shuttle', 'flow cytometry')

    def __init__(self, experiment_name: str = 'Experiment',
                 dataset_names: List[str] = all_datasets_list,
                 metric_collection_during_layout_creation: Dict[str, int] = None,
                 num_repeats: int = 3,
                 measure_memory_use: bool =True,
                 measure_time: bool = True,
                 iterations: int = 100):

        assert measure_memory_use or measure_time, "Either time or memory use has to be measured"

        self.dataset_names = dataset_names
        self.experiment_name = experiment_name
        self.metric_collection_during_layout_creation = metric_collection_during_layout_creation
        self.num_repeats = num_repeats
        self.measure_memory_use = measure_memory_use
        self.measure_time = measure_time
        self.iterations = iterations

    def create_output_directory(self):

        name = f"{self.experiment_name} - pickled"
        # exception catching bc we don't want to have to re-run the experiment in case of we mess up the naming
        try:
            path = os.path.realpath(os.path.join(os.path.dirname(__file__), '.', name))
            os.mkdir(path)
        except FileExistsError:
            path = os.path.realpath(os.path.join(os.path.dirname(__file__), '.', name + "(1)"))
            os.mkdir(path)

        return path

    def pickle(self, save_to: Path = None):
        if save_to:
            directory = save_to
        else:
            directory = Path(__file__).parent.resolve()

        path_to_pickle = directory.joinpath(Path(f"{self.experiment_name}.pickle"))
        with open(path_to_pickle, 'wb') as pickle_out:
            pickle.dump(self, pickle_out)
        return path_to_pickle

