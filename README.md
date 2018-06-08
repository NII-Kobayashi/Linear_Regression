# Prediction of Twitter re-tweet dynamic using Linear Regression Model

This is the code illustrating the *Szabo and Huberman, Communication of the ACM 53, 80 2010;
Zhao et al., in KDD' 15 2015 pp. 1513-1522*.
It aims at providing a framework for experimenting with Linear Regression in
the context of twitter re-tweet prediction.

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

 - *example.py* : This code estimates all the model parameters (alpha, variance) from observation data
    and predicts future re-tweet activity.
 - *example_cross_validation.py* : This code evaluates the error of the model by using 5 cross validation
    and prints the average mean, media error and correlation

You can just run :

     python example.py
     python example_cross_validation.py

Without modifying anything. The example files are commented and should be
readable and understandable.

## Description of each module

 - *estimate.py* :  implements the basic mathematical expression
    from the linear regression equations used in the papers for estimating the parameters.
 - *prediction.py* : implements the basic mathematical expression
    from the linear regression equations used in the papers for predicting the parameters.
 - *cross_validation.py* : function for 5- fold cross validation to check the accuracy of the model.
 - *functions.py* :  implements the function for extracting the number
    of events from the data file and a function to sort the file name numerically


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

Please contact Ryota Kobayashi if you want to use the code for commercial purposes.