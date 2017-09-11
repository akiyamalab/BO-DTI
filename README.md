# Scripts for "Efficient Hyperparameter Optimization by Using Bayesian Optimization for Drug-Target Interaction Prediction"

A Bayesian optimization technique enables a short search time for a complex prediction model that includes many hyperparameters while maintaining the accuracy of the prediction model. Here, we apply a Bayesian optimization technique to the drug-target interaction (DTI) prediction problem as a method for computational drug discovery. We target neighborhood regularized logistic matrix factorization (NRLMF) (Liu _et al_., 2016), which is a state-of-the-art DTI prediction method, and accelerated parameter searches with the Gaussian process mutual information (GP-MI). Experimental results with four general benchmark datasets show that our GP-MI-based method obtained an 8.94-fold decrease in the computational time on average and almost the same predicted area under the curve (AUC) for all datasets compared to those of a grid parameter search, which was generally used in DTI predictions. Moreover, if a slight accuracy reduction (approximately 0.002 for AUC) is allowed, an increase in the calculation speed of 18 times or more can be obtained. Our results show for the first time that Bayesian optimization works effectively for the DTI prediction problem. By accelerating the time-consuming parameter search, the most advanced model can be used even if the number of drug candidates and target proteins to be predicted increase.


Requirements
------------
- python 2.x
- 
- 

Installation
------------
1. Download the archive of xx from this repository.
2. Extract the archive and cd into the extracted directory.
3. Run make command.

Commands:

    $ 
    $ 
    $ 

    
Usage
-----


Example
-------

    $ 
    $ 
    

References
----------
Tomohiro Ban, Masahito Ohue, Yutaka Akiyama: Efficient Hyperparameter Optimization by Using Bayesian Optimization for Drug-Target Interaction Prediction, In _Proceedings of the 7th IEEE International Conference on Computational Advances in Bio and Medical Sciences (ICCABS 2017)_, Orlando, FL, USA, October 19-21, 2017. (accepted) 

(Conference Website) http://www.iccabs.org/


Copyright © 2017 Akiyama_Laboratory, Tokyo Institute of Technology, All Rights Reserved.  
