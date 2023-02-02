import pickle
from typing import List, Dict
import os


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


class ComparisonBase:
    all_datasets_list = ('mnist', 'coil20', 'rna N3k', 'airfoil', 'wine quality',
                         'fashion mnist', 'shuttle', 'flow cytometry')

    def __init__(self, experiment_name: str = 'Experiment', dataset_names: List[str] = all_datasets_list,
                 metric_collection: Dict[str,int] = None, num_repeats: int = 3, record_memory=False,
                 iterations: int = 100):
        self.dataset_names = dataset_names
        self.experiment_name = experiment_name
        self.metric_collection = metric_collection
        self.num_repeats = num_repeats
        self.record_memory = record_memory
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

    def pickle(self):
        directory = os.path.realpath(os.path.dirname(__file__))
        path_to_pickle = os.path.join(directory, f"{self.experiment_name}.pickle")
        with open(path_to_pickle, 'wb') as pickle_out:
            pickle.dump(self, pickle_out)
        return path_to_pickle

