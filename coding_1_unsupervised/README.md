# UK-High value Customers Identification

## 1: My k-means implementation 
For this question, you will submit your "cold" k-means implementation and include 
your justification for the stopping condition(s) that you used in your implementations. 
Your implementation should be called `my_kmeans()` and should set at three inputs **in this 
order** -
* A numpy array
* The number of cluster (ie. _k_)
* The random_state

Your implementation should terminate with output including:
- The cluster centers
- Cluster labels for the data points


## 2: Choosing _k_ using elbowology 
### Part A
In k-means, we supply the number of clusters that we believe our data has. This means that 
the choice of _k_ is made without being directly derived from the data. In this question, 
you will use _elbowology_ to determine the number of clusters that our data falls into. 

For this question, we will use the `students_info.csv` for this question. For each of the 
three variable combinations, please normalize your variables and then do the following:
* Using either _within cluster sum of squares_ or _average cluster cohesion_ as the measure of 
  cluster "goodness", write a function `looping_kmeans` that perform k-means using `sklearn` 
  and computes the "goodness" of clusters for _k=1, k=2, ..., k=10._ While the k-means should 
  use the `sklearn` implementation, the measure of "goodness" should be written by you. 
  The inputs should be 1) a numpy array and list of values for _k_, with the output being the 
  list of the "goodness" measures. 
* Plot the values of _k_ against your chosen measure of cluster "goodness" as a line plot with 
  each point marked clearly. 
* Examining your plot, find the value of _k_ that is closest to the "elbow"; that is where the 
  plot changes directions most sharply. This point should look like the elbow on a bent arm.   
  
The variable combinations are:
1. Gym time and average cups of coffee
2. Sleep and GPA
4. All numerical variables within `students_info.csv`

### Part B
Given your plots above, how many clusters do our students fall into. You must choose **one** 
number and justify your choice. 
