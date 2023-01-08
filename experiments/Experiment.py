import pickle
from typing import List, Dict
import os


def load_object(filename):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

class Experiment:

    all_datasets_list = ('poker', 'mnist', 'bonds', 'coil20', 'rna N3k', 'airfoil', 'wine quality', 'fashion mnist'
                         'shuttle', 'flow cytometry', 'flow cytometry')

    def __init__(self, experiment_name: str, dataset_names : List[str] = all_datasets_list,
                 metric_collection: Dict[str:int] = None, repeat: int = 3):
        self.dataset_names = dataset_names
        self.experiment_name = experiment_name
        self.metric_collection = metric_collection
        self.repeat = repeat
        self.printout_list = []

    def create_output_directory(self):
        # exception catching bc we don't want to have to re-run the experiment
        try:
            path = os.path.realpath(os.path.join(os.path.dirname(__file__),'.' ,self.experiment_name))
            os.mkdir(path)
        except FileExistsError:
            path = os.path.realpath(os.path.join(os.path.dirname(__file__), '.', self.experiment_name + "(1)"))
            os.mkdir(path)

        return path

    def h(self):
        h = '#'*20
        self.printout_list.append(h)
        print(h)

    def pr(self, line_to_print: str):
        self.printout_list.append(line_to_print)
        print(line_to_print)