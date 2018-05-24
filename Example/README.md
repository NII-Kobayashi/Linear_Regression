# Prediction of Twitter retweet dynamic using time dependant Hawkes Process

This is the code illustrating the *Kobayashi & Lambiotte 2016* paper and aims at
providing a framework for experimenting with Linear Regression in
the context of twitter retweet prediction.

## Requirements

 - Python 3
 - glob >= 0.19
 - Numpy >= 1.10.4

## Getting started

The git repository can be cloned by simply using:

    git clone <TODO>

Once the repository is cloned, the folder should contain three different
subfolders and this README file.

The **data** folder contains some twitter data that can be used for testing.

The **Linear_Regression** folder contains all the core python code.

## Running some examples

You can just run :

    python3 <name of the example file>

Without modifying anything. The example files are commented and should be
readable and understandable.

## Description of each module

 - *main.py* : this is a front-end module that offers high-level functions to
   estimate the parameters of the model and compute prediction on a tweet
   sequence.
 - *estimate.py* : this is the file that implements the estimation algorithm, to
   estimate the value of the infectious rate function according to a given tweet
   sequence at different points in time.
 - *fit.py* : given estimated values for the infectious rate function, this
   module permit to find the parameters for a given model that fit the
   estimation best. By default, the infectious rate model described in the
   article is used.
 - *prediction.py* : this module implements the prediction algorithm using the
   self-consistent integral equation described in the article, but with
   arbitrary kernel and infectious rate functions. By default, the ones in the
   paper are used.
 - *training.py* : implements the training for best parameters of the infectious
   rate function, with given tweet sequences as input. Will find the parameters
   for the infectious rate function described in the paper that minimize the
   global prediction error for all the input data.
 - *simulate.py* : implements a simulation of a random tweet sequence according
   to the model described in the paper. Very useful to generate artificial
   twitter data.
 - *functions.py* : implements the basic mathematical expression of the
   different functions used, mainly the kernel memory function and the
   infectious rate function as they are described in the paper.

