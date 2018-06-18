# Prediction of Twitter re-tweet dynamic using Linear Regression

This code predicts the number of retweets in a future period and it was used in *Kobayashi and Lambiotte, ICWSM, 2016, pp. 191-200*. The algorithm is a modification of Lineaer Regression method  *Szabo and Huberman, Communication of the ACM 53, 80 2010* and *Zhao et al., KDD, 2015 pp. 1513-1522*.

## Requirements

 - Python 3
 - Numpy >= 1.10.4
 - sklearn >=  0.19.1

## Getting started

The git repository can be cloned by simply using:

    git clone <TODO>

Once the repository is cloned, the folder should contain two different
sub-folders and this README file.

The **Data** folder contains some twitter data that can be used for training and testing.

The **Linear_Regression_model** folder contains all the core python code, example file and Data folder.

## Running some examples

There are two example code in the directory, i.e. example.py, example_cross_validation.py

 - *example.py* : This code estimates the model parameters (alpha, variance) from observed data and predicts the number of retweets from the observation time to the prediction time.
 - *example_cross_validation.py* : This code evaluates the error of the method based on 5-fold cross validation.

You can just run :

     python example.py
     python example_cross_validation.py

Without modifying anything. The example files are commented and should be
readable and understandable.

## Description of each module

- *estimate.py* :  functions for estimating the model parameters.
- *prediction.py* : function for predicting the number of retweets in a future period
- *cross_validation.py* : function for evaluating the accuracy of the model based on cross-validation.
 - *functions.py* :  a function for calculating the number of retweets and a function sorting the file names numerically


## Data source

The provided samples are extracted from the data set used by Zhao et al. in the
[SEISMIC](http://snap.stanford.edu/seismic/seismic.pdf) paper.
You can find more information about the data [here]
(http://snap.stanford.edu/seismic/#data).

For this work the data (used for training) was slightly aggregated to the
following format:
- one file peer tweet
- space separated
- first row: \<number of total re-tweets\> \<start time of tweet in days\>
- every other row: \<relative time of tweet/re-tweet in seconds\> \<number of followers\>
- only

## License

This project is licensed under the terms of the MIT license.
Please cite the paper (Kobayashi and Lambiotte, ICWSM 2016, pp.191-200, 2016) in you used the code.
Please contact Ryota Kobayashi if you want to use the code for commercial purposes.
