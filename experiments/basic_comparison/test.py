from SimpleComparison import SimpleComparison
from hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96 import Chalmers96


algo1 = Chalmers96(None, neighbour_set_size=1, sample_set_size=0, use_knnd=True)
algo2 = Chalmers96(None, neighbour_set_size=1, sample_set_size=0, use_knnd=False)

datasets = ['mock data']
    # , 'rna N3k', 'airfoil']

algos = {algo2: 'no knnd'}
metric_collection = { "Average Speed": 3}

experiment = SimpleComparison(algos, experiment_name='test experiment', metric_collection=metric_collection,
                              dataset_names=datasets, num_repeats=1, iterations=1)


experiment.run()
path = experiment.create_output_directory()
experiment.save(path)
print(path)