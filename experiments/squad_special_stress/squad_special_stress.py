from ..Experiment import Experiment


class SquadSpecialStress(Experiment):
    name = "SquadSpecialStress"

    def __init__(self, experiment_name: str):
        super().__init__(experiment_name)

    def run(self):
        for dataset in self.datasets:
            pass








    dataset = DataFetcher().fetch_data('poker', size=1000)
    Xld = PCA(n_components=2, whiten=False, copy=True).fit_transform(dataset.data).astype(np.float64)
    # Xld *= 10/np.std(Xld)

    algo96 = Chalmers96(dataset=dataset, initial_layout=Xld, alpha=0.7, distance_fn=poker_distance,
                        neighbour_set_size=0, sample_set_size=4)

    layout = LowDLayoutCreation().create_layout(algo96, dataset, metric_collection=metric_collection, no_iters=200)

    print(f"iterations stress: {layout.collected_metrics['stress'][0]} \n")
    print(f"iterations velocity: {layout.collected_metrics['average speed'][0]} \n")
    print(f"velocity: {layout.collected_metrics['average speed'][1]} \n")
    print(f" stress: {layout.collected_metrics['stress'][1]} \n")
    print("total time: {}")
    show_layout(layout, use_labels=True)
    show_generation_metrics(layout, average_speed=True)