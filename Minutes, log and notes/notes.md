* Unsolved puzzles in Math,
 What could this be applied to?

* We can evaluate local vs more global structure by comparing with a k-nearest neighbour classifier trained on the embedding  and using cross validation - the measure for comparison can be recall as defined in the k-NN paper
  \
  \
We can make nice swarm plots with accuracy values obtained from cross validation - code available at Lmcinnes github - umap_paper_notebooks
* We can measure the stability by using something called Procrustes distance (normalised) - between the full embedding and samples - and plot the result against sample size 10% 20% etc
* The guy whose code I will be using for k-KNN came up with UMAP
* Use a 3 dimensional colour cube for testing  



* stochastic MDS paper  uses slightly different stress
* they claim that hybrid or pivot (and loads of others) - many parameters add cumbersome complexity
* t-SNE, UMAP don't preserve global structures well
* SQUAD MDS - O(N) - but that's per iteration ( like with all the other algorithms )  
therefore important how many iterations needed for convergence
* one version of Squad MDS uses something called Nesterov momentum 
* Squad MDS initialises the embedding with PCA
* squad uses a different gradient function
* squad evaluation uses something called RNX quality curves 
to evaluate the layouts comparing them to SMACOF algo ( O(n2))
but this method of evaluation does not seem very popular 
otoh the code to produce those curves is freely available
in pypi (nxcurve)
<br/>
<br/>
<br/>
* at least for some cytometric data, initialisation is critical for UMAP and t-SNE
* something called Laplacian Eigenmaps can be used to initialise the embeddings - apart from PCA
<br/>
<br/>
* one way to evaluate the embeddings here -ebecht /DR_benchmark
but it's in R
* also more ideas in: dimensionality reduction for visualizing single-cell data using umap
<br/>
<br/>
* FIt SNE is p(small number) N but takes a lot of memory apparently
* UMAP can be used to reduce dimensions t-SNE - more apparently not computationally feasible to do the same on large datasets
* memory-mapped files to reduce memory use?
* special data structures?
* rss -resident set size mem- can be misleading bc does not include memory "swapped" to the hard drive
* vms - can be misleading because it includes all memory used by the process including shared libs and can become larger that the total aval memory on the machine
* tracemalloc only works for code written in python



velocity x = (xld1 - xld2)* ((D_ld - D_hd)/ D_) * alpha * data_size_factor
