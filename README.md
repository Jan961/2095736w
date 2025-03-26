
**Datasets used:**
- custom toy dataset of points sampled from a surface of a 3D cube
- custom toy dataset of points sampled from a 3D globe - idea and a significant part of the implementation from  [https://github.com/NikolayOskolkov/tSNE_vs_UMAP_GlobalStructure](https://github.com/NikolayOskolkov/tSNE_vs_UMAP_GlobalStructure)
- [Coli20](https://git-disl.github.io/GTDLBench/datasets/coil20/)
- [Mouse scRNA](https://pubmed.ncbi.nlm.nih.gov/30382198/) as provided by the authors of SQuadMDS who use the pre-pocessing described in [https://www.nature.com/articles/s41467-019-13056-x](https://www.nature.com/articles/s41467-019-13056-x)
- random sample (with reduced resolution) of [MNIST](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)
- random sample (again with reduced resolution) [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)  
  
  



**Alogithms**

Main algorithms (re) implemented in Python here:
- Pierre Lambert, Cyril de Bodt, Michel Verleysen, John A. Lee a c ["SQuadMDS: a lean Stochastic Quartet MDS improving global structure preservation in neighbor embedding"](https://www.sciencedirect.com/science/article/abs/pii/S0925231222008402)
- Matthew Chalmers  ["A linear iteration time layout algorithm for visualising high-dimensional data"](https://ieeexplore.ieee.org/document/567787 )
- Alistair Morrison, Greg Ross, Matthew Chalmers ["A hybrid layout algorithm for sub-quadratic multidimensional scaling"](https://www.researchgate.net/publication/4000921_A_hybrid_layout_algorithm_for_sub-quadratic_multidimensional_scaling)
  
For some (limited) comparison two popular high dimensional data-vis algorithm as used, as implemented in scikit-learn:
- [UMAP](https://arxiv.org/abs/1802.03426) - implementation doc [here](https://umap-learn.readthedocs.io/en/latest/basic_usage.html)
- [t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) ( Barnes-Hut variant ) - implementation doc [here](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) 






It is important to note that the python interpreter for the project must use **Python version 3.10** as some packages used are not compatible with other versions.  

The project contains 3 example scripts - each in duplicate as python file and a jupyter notebook file.


Almost all of the code used to run and process the data in experiments 
is contained in the *experiments* package so that the obtained results can be replicated.

<br>
<br>
Below is a shortened and incomplete description of the API used:
<br>

<br>
<span class = "section"> <b>Dataset object:</b></span><br><br>


## <b> hdimvis.data_fetchers.Dataset.Dataset() </b>


*basic dataset object* <br> <br>
<span class="main"><b>Attributes:</b><br></span>


`data: numpy.ndarray`<br><br>
`labels: numpy.ndarray|None`<br>
contains data labels if the dataset is labeled <br><br>
`name: str`<br>
name of the dataset<br><br>




<span class = "section"> <b>Cube test dataset:</b></span><br><br>

*this isa unique dataset - created and used differently form the one described below*
<br><br>
<span class="main">
## <b>experiments.cube.Cube.Cube</span>( side = 10, angle = 1/6, num_points = 2, distance_axes = 2)</b>
</span>

<br>
<br>
<span class="main">
<b>Parameters:</b><br><br>
</span>
`side : int, optional` <br>
    length of the side of the cube  <br><br>

`angle : float, optional`   <br>
    angle between the bottom side of the cube and the x-y plane,
    the angle is multiplied by pi so that angle=1/3 will
    mean that the angle pi/3 will be used when generation the <br><br>

`num_points: int, optional`<br>
    number of points per side<br><br>

`distance_axes: int, optional`<br>
 distance from the axes, e.g. distance 1 will mean that cube will be translated by the vector (1,1,1) from the position at the origin and then rotated by the `angle`<br><br>
  

<span class="main">
<b>Methods:</b><br><br>
</span>

### <b> plot_3d(title: str = None, axes_labels_off = False, size_inches: int None) </b> <br>

*plot the 3D cube*


`title: str, optional` <br><br>

`axes_labels_off: bool, optional` <br><br>

`size_inches: int, optional` <br>
size of the plot<br><br>

**returns**<br>
None<br><br>


### <b> get_sample_dataset(self, size) </b> <br>

`size: int`<br>
size of the dataset

**returns**<br>
Dataset object

### <b> plot_2d(self, layout = None, layout_points= None, hd_points =None, opacity= 1, title = None, save_to = None) </b>

*plot the 2D embedding of the cube*

`layout: LowDLayoutBase object, optional`<br>
if provided the method extracts all the required data from the Layout object automatically if not `layout_points` and `hd_points` in a corresponding order must be provided  <br><br>

`layout_points: np.ndarray, optional`<br> see above <br><br>

`hd_points : np.ndarray, optional`<br>see above <br><br>

`opacity: float, optional`<br> self-explanatory <br><br>

`title: str, optional`<br> self-explanatory <br><br>

`save_to: Path, optional`<br> self-explanatory <br><br>
**returns**<br>
None


<br><br><br><br>
<span class = "section"> <b>Loading all other datasets:</b></span><br><br>


## <b>hdimvis.data_fetchers.DataFetcher.DataFetcher.fetch_data</span>( dataset_name_raw = 'rna N3k',  **kwargs)</b>

<br>

*static method of the DataFetcher factory class to load any dataset*



`dataset_name_raw: str` <br>
dataset name available datasets are:<br>
*'mnist', 
'bonds', 
'coil20', 
'rna N3k', 
'airfoil', 
'wine quality', 
'fashion mnist'
'shuttle', 
'flow cytometry',
'mock data',
'globe',
'cancer RNA',
'metro'*



`**kwargs` <br>
some dataset have variable size - this can be set here by passing `size` argument to the fetcher, 
the *globe* data set can be loaded as "swiss roll" or continents on a sphere (`swiss_roll: bool = False` parameter) the "swiss roll" also has adjustable number of revolutions (`revolutions: float =2.0`) and tightness (`tightness: float = 1.0`) <br>
this parameter also enables flexibility for adding further datasets whose loading process might be possible to adjust by some further parameter settings

**returns**<br>
Dataset object

<br><br><br><br>
<span class = "section"> <b>Algorithms</b></span><br><br>

*Since for layout creation an algorithm object only needs to be created (with a dataset object passed into its __init__ method) but none of its methods need to be called directly, no descriptions of methods is provided here and only some attributes are mentioned, many parameters are also not described in detail as their names seem to be self-explanatory - for more information see the source code and dissertation* 


## <b>hdimvis.algorithms.spring_force_algos.chalmers96_algo.Chalmers96.Chalmers96</b></span>(<br>
<b>dataset, <br>
record_neighbour_updates = False, <br>
nodes = None,<br>
enable_cache = True,<br>
use_knnd = False, knnd_parameters: Dict = None,  <br>
neighbour_set_size = 5, <br>
sample_set_size = 10, <br>
spring_constant = 0.5, <br>
sc_scaling_factor = None, <br> 
integrate_sum  = True, <br>
damping_constant = 0, <br>
initial_layout = None, <br>
distance_fn = euclidean, <br>
**kwargs)</b><br>



`dataset: Dataset object`<br><br>
`record_neighbour_updates: bool, optional`,<br>
record the absolute number of neighbour set updates on each iteration<br><br>
`nodes: List[Node], optional` ,<br>
provide a list of Node objects (datapoints) to the algorithm<br><br> 
`enable_cache: bool optional`,<br><br>
`use_knnd: bool, optional` , <br>
use the kNND algorithm to pre-compute neighbour sets <br><br>
`knnd_parameters: Dict, optional`,<br>
a dictionary for passing optional arguments to the kNND (pynndescent) algorithm<br><br>
`neighbour_set_size: int, optional`,<br><br>
`sample_set_size: int, optional` ,<br><br>
`spring_constant: float, optional` ,<br><br>
`sc_scaling_factor: float, optional`,<br>
spring constant scaling factor<br><br>
`integrate_sum : bool, optional` ,<br>
use the cumulative force (sum) for position updates if False update the position as soon as new force is calculated <br><br>
`damping_constant: float, optional` ,<br><br>
`initial_layout: np.ndarray, optional` ,<br>
layout positions to initialise the algo<br><br>
`distance_fn: Callable, optional`,<br><br>



<br>
<br>
<br>


## <b>hdimvis.algorithms.spring_force_algos.hybrid_algo.Hybrid.Hybrid</b></span>(
<b>dataset: Dataset, <br>
initial_layout = None, <br>
distance_fn = euclidean <br>
preset_sample  = None, <br>
 interpolation_adjustment_sample_size = 15, <br>
use_correct_interpolation_error = True, <br>
use_random_sample = False, <br> 
num_strata = 20, <br>
nodes = None, <br>
enable_cache = True, <br>
use_knnd = False, <br>
knnd_parameters = None <br>
neighbour_set_size = 5, <br>
sample_set_size = 10, <br>
spring_constant = 0.5, <br>
sc_scaling_factor = None, <br>
integrate_sum  = True <br>
**kwargs)</b>b<br>


<br><br>

`dataset: Dataset`, <br><br>
`initial_layout: np.ndarray, optional`, <br> as above <br><br>
`distance_fn: Callable, optional` <br><br>
`preset_sample : np.ndarray, optional` <br> choose the sample indices manually <br><br>
`interpolation_adjustment_sample_size: int,optional` <br> sample size to be used when adjusting the position of a point interpolated into the layout <br><br>
`use_correct_interpolation_error: bool, optional` <br>
use the corrected error function if False the 2019 error will be used <br><br>
`use_random_sample: bool, optional` <br>
if False stratified sample will be used <br><br>
`num_strata: int, optional` <br> number of divisions to use for the stratified sample <br><br>
`nodes: List[Node], optional` <br>as above<br><br>
`enable_cache: bool, optional` <br><br>
`use_knnd: bool , optional` <br>as above<br><br>
`knnd_parameters: Dict, optional` <br>as above<br><br>
`neighbour_set_size: int, optional` <br><br>
`sample_set_size: int, optional` <br><br>
`spring_constant: float , optional` <br><br>
`sc_scaling_factor: float , optional` <br>as above<br><br>
`integrate_sum : bool , optional` <br>as above<br><br>
`damping_constant: float, optiona`<br><br>
<br><br>



## <b>hdimvis.algorithms.stochastic_ntet_algo.SNeD.SNeD</b></span>(
<b>dataset, <br>
initial_layout = None <br>
distance_fn = euclidean <br>
ntet_size = 4  <br>
use_nesterovs_momentum = False <br>
momentum = 0.6 <br>
is_test = False <br>
use_rbf_adjustment = False  <br>
**kwargs <br>
</b>
<br><br>

`dataset: Dataset`, <br>
`initial_layout: np.ndarray, optional` , <br>as above<br><br>
`distance_fn : Callable, optional`, <br>as above<br><br>
`ntet_size: int, optional`, <br> size of the basic unit for GD and position updates<br><br>
`use_nesterovs_momentum: bool, optional`,<br><br>
`momentum: float, optional`,<br><br>
`is_test: bool, optional` , <br> if True ntet_size must be set to 4 and the algorithm tests if all the vectorised gradient and distance calculations return the same results as the original code provided by the authors <br><br> 
`use_rbf_adjustment: bool, optional` , <br>
use rbf distances for each n_tet  <br><br>

<br><br>

<span class = "section"> <b>Layout creation:</b></span><br><br>



*Static method of the LayoutCreation (factory) class is described below, only an Algorithm object needs to be passed to the method and the number oor iteration or another termination condition) specified to create a layout*


## <b>hdimvis.create_low_d_layout.LayoutCreation.LayoutCreation.create_layout(</b></span><b>( <br>
algorithm: BaseAlgorithm, <br>
num_iters = None , <br>
optional_metric_collection = None,<br>
**additional_parameters<br>
</b>

`algorithm: BaseAlgorithm`, <br><br>
`num_iters: int, optional`,<br>
number of iteration to run the layout creation for <br><br>
`optional_metric_collection: dict[str: int], optional` ,<br>
can be used to specify what metric are to be collected during layout creation and the interval of collection, the format must be: <br>
{metric_name : iteration_interval (e.g. 2 means every 2 iterations)   <br>
if a valid value is passed the layout create will also collect the specified metric at the beginning and after the last iteration regardless of the value of `num_iters` mod iteration_interval <br><br>
`**additional_parameters`<br>
additional parameters specific to each algorithm can be passed here <br><br>

**returns**<br>
Layout object
<br><br><br>

## <b>hdimvis.create_low_d_layout.LayoutBase.LayoutBase</b></span>
<br><br>

*Layout objects are created automatically after running `create_layout` rather than directly, thus the details of its __init__ method are not described here only its most important attributes and the most important method*<br><br>

**attributes:**<br>
</span>
`algorithm`<br> the algorithm object used to create the layout<br> <br> 
`final_positions`<br> 
positions of the point on the layout<br> <br> 
`num_iters `<br> 
number of iterations to be performed
`collected_metrics `<br> 
collected metrics stored in a dict of the form:
{metric_name : tuple(List([iteration_numbers]),List([metric_values]))
where iteration numbers are the numbers of iteration at which the metrics where measured<br> <br> 
`data` <br> 
original HD data<br> <br> 
`labels `<br> 
data labels<br> <br> 

`iteration_number` <br> 
current iteration number<br> <br> 
<span class="main">
**methods:**<br> 
</span>
get_final_stress()

**return**
stress of the layout: float
<br><br><br><br>

<span class = "section"> <b>Layout visualisation:</b></span><br><br>

*Two functions are used to visualise the data contained in a layout object. These are described below*

<br><br>


## hdimvis.visualise_layouts_and_metrics.plot.show_layout</b></span><b>(<br>
layout =None, <br>
use_labels = False, <br>
alpha: float = None,<br>
color_by = None,<br>
color_map = 'rainbow',<br>
size = 3, <br>
title = None,<br>
save_to  =None,<br>
positions =None,<br>
labels= None<br>
</b>
<br><br>

`layout: LowDLayoutBase`, <br>
Layout object, if not provided must provide `positions` to plot and optionally `labels`<br><br>
`use_labels: bool, optional`, <br>
use data labels for colouring<br><br>
`alpha: float, optional`,<br>
transparency of datapoints<br><br>
`color_by: Callable[[np.ndarray],float], optional`,<br>
use a custom function for colouring - e.g. by one of the data dimensions rather than their labels<br><br>
`color_map: str, optional`,<br> name of the Pyplot color map to use<br><br>
`size: float, optional`, <br> size of a datapoint on the plot <br><br>
`title: str, optional`,<br><br>
`save_to: Path, optional`,<br><br>
`positions: np.ndarray`,<br>
if layout object not provided must use this <br><br>
`labels: np.ndarray`<br> if layout does not contain labels or there is no Layout object pass labels them here <br><br>

<br><br>

## <b>hdimvis.visualise_layouts_and_metrics.plot.show_generation_metrics(</b></span><b><br>
layout, 
stress = True,<br>
average_speed = False, <br>
quartet_stress = False,<br>
title = None,<br>
save_to = None, <br>
log_scale = False,<br>
iters_from = None, <br>
iters_to = None<br>
</b>
<br><br>

`layout: LowDLayoutBase`, <br>
must provide a layout object to show collected metrics<br><br>
`stress: bool, optional`,<br>
show collected Kruskal stress measurements<br><br>
`average_speed: bool, optional`, <br>
show average speed (only available for Spring Force Layouts) if 
`stress` is True a twin axis will be added to show both metric on the same plot <br><br>
`quartet_stress: bool, optional`,<br>
similar to above but for SNeD<br><br>
`title: str, optional`,<br><br>
`save_to: Path, optional`, <br><br>
`log_scale: bool, optional`,<br>
use log scale on the y axis<br><br>
`iters_from: int, optional`, <br>
auto adjust the plot to show only metrics collected in the specified range if None all metrics from the start will be showed <br><br>
`iters_to: int, optional` <br>
as above, if None, all measurements until the termination of layout creation will be showed<br><br>
