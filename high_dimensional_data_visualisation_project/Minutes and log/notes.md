* Unsolved puzzles in Math,
 What could this be applied to?

* We can evaluate local vs more global structure by comparing with a k-nearest neighbour classifier trained on the embedding  and using cross validation - the measure for comparison can be recall as defined in the k-NN paper
  \
  \
We can make nice swarm plots with accuracy values obtained from cross validation
* We can measure the stability by using something called Procrustes distance (normalised) - between the full embedding and samples - and plot the result against sample size 10% 20% etc
* The guy whose code I will be using for k-KNN came up with UMAP
* Use a 3 dimensional colour cube for testing  



* stochastic MDS paper  uses slightly different stress
* they claim that hybrid or pivot (and loads of others) - many parameters add cumbersome complexity
* t-SNE, UMAP don't preserve global structures well
* SQUAD MDS - O(N) - but that's per iteration ( like with all the other algorithms )  
therefore important how many iterations needed for convergence
* Squad MDS uses something called Nesterov momentum 
* Squad MDS initialises the embedding with PCA