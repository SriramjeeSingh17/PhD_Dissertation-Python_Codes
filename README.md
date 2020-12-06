# Applying ML on weather data to predict prices


This repository contains codes for my paper "Estimating the Impact of Weather on CBOT Corn Futures Prices". In this paper, I use machine learning techniques on weather data to predict corn futures prices as well as their direction of movement.

### Please cite my paper if you use my codes. Thanks!

@article{ssingh_2020,
   title={Estimating the Impact of Weather on CBOT Corn Futures Prices},
   url={https://iastate.box.com/s/dsemvmcw8pn6occsyeb3ygjas0nf6r2l},
   author={Singh, Sriramjee},
   year={2020}
}


#### analysis1.py

This code analyzes the yield profile of the counties of the major corn-producing states. This is a dynamic program involving multiple nested loops, and files reading and writing. 

#### prelim_analysis1.py

This code finds the most important county as an early indicator to the market. This includes seaborn violin/box plots. It uses Haversine formula to spatially order the states and counties (the dynamic code automatically updates the benchmark).

#### preprocess2.py

This code appends all counties of a state for all years dynamically.

#### program1.py

This code manages the whole data sets. It includes weather data for all counties of all states, soil and other related variables, and futures prices. It does all required pre-processing to get the required final data.

#### program1_run.py

This code applies neural networks predictors like fully connected neural network, CNN, and CNN-LSTM on the data. The input data to be used in the neural networks is converted to 3D from 2D dynamically. It has other required preprocessing steps for modeling as well. It also has seaborn heatmap for finding correlations in the weather and soil data.

#### program1_run_class.py

This code applies classifiers like SVM, LR, Random Forest, XGBoost, CNN etc. on the data. It contains codes for K-means for clustering and PCA for dimension reduction as well.








