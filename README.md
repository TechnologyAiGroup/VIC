# VIC

---

VIC is a two-stage method to improve diagnosis results produced by mainstream commercial diagnosis tool.

It takes only a couple of seconds to train, and uses no additional information besides conventional logic diagnosis.

----

We choose `s13207` for the instance to display the process.

## Environment

`pip install -r requirements.txt`

## Data preprocessing

The raw data is placed in the `DiagData` folder and extracted as a CSV file by running the following command. 

`python raw2csv.py`


## Stage I of VIC

Cluster the data. The results will be saved in `cluster_data_` and `experiment/clusterInfo_`  

For bottom-up clustering, run the following command.    
`python vic_stage1.py s13207 hac 90`  
`python vic_stage1.py s13207 dtw 90` 

> `s13207` means the circuit name.  
> `hac` means the cluster method with Jaccard distance and hierarchical clustering. `dtw` means the cluster method with Dynamic Time Warping distance and hierarchical clustering.  
> The last number means the distance threshold in clustering process. 90 is equivalent to 90/100 = 0.9.

For top-down clustering, run the following command.  
​      `python vic_stage1.py s13207 f1 2`  
​      `python vic_stage1.py s13207 f2 2`  
​      `python vic_stage1.py s13207 f1 4`  
​      `python vic_stage1.py s13207 f2 4` 

> `s13207` means the circuit name.  
> `f1` and `f2` mean the encoding format 1 and format 2, with K-Means method.  
> The last number means the number of clusters. For ease of comparison, it is recommended to set this value as the clustering result of `hac` or `dtw`.


## Stage II of VIC

  Train with HMMs.

  - #### One-dimensional HMM

    The results will be saved to `experiment/batch_1d_`

    `python vic_stage2.py s13207 hac 4`

    > `s13207` means the circuit name.
    >
    > `hac` means the method and `4` is the number of clusters. The method can be replaced by `dtw`, `f1`or `f2`, and the number of clusters should also be replaced with the results of Stage I accordingly.

  - #### Three-dimensional HMM and Seven-dimensional HMM

    The results will be saved to `experiment/res_3d` or  `experiment/res_7d`

    `python vic_stage2_multi-observations.py.py s13207 hac 4`

    > The parameters are the same as above. In this python file, you can change the variable `n_dim` to 3 or 7 for different dimensions.



### Train without Stage I

  `python vic_stage1.py s13207 k1 1`

  `python vic_without_stage1.py s13207`

  > These parameters are fixed. 

### Random Compare Experiments

  `python random_cmp.py 100`

  `python random_cmp.py 10000`

  > The parameter means the number of random rearrangements. 

### Continuous Processing

  `python vic_stage2_continuous_processing.py s13207 hac 4`


## Statistic

Statistic the experimental results. The results will be saved in the CSV files to `experiment/`. Before running the following commands, please modify these paths in `statistic.py`. For example, 

  ```python
root_experiment = './experiment/batch_1d_'
root_clusterInfo = './experiment/clusterInfo_'
output_csv_path = f'./experiment/1d.csv'
  ```

  `python statistic.py`

