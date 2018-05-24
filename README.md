# Prediction of Twitter retweet dynamic using Linear Regression Model

This is the code illustrating the *Kobayashi & Lambiotte 2016* paper and aims at
providing a framework for experimenting with Linear Regression in
the context of twitter retweet prediction.

## Requirements

 - Python 3
 - glob
 - Numpy >= 1.10.4

## Getting started

The git repository can be cloned by simply using:

    git clone <TODO>

Once the repository is cloned, the folder should contain two different
sub-folders and this README file.

The **Data** folder contains some twitter data that can be used for training and testing.

The **Linear_Regression** folder contains all the core python code.

## Running some examples

You can just run :

    python3 <name of the example file>

Without modifying anything. The example files are commented and should be
readable and understandable.

## Description of each module

 - *main.py* : his is a front-end module that offers high-level functions to
   estimate the parameters of the model and compute prediction on a tweet
   sequence.
 - *estimate.py* :  implements the basic mathematical expression
    from the linear regression equations used in the paper for estimating the parameters.
 - *prediction.py* : implements the basic mathematical expression
    from the linear regression equations used in the paper for predicting the parameters.
 - *functions.py* :  implements the basic mathematical expression
    from the linear regression equations used in the paper for extracting the number
    of events from the data file and a function to sort the file name numerically


