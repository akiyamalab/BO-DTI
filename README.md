# Scripts for "Efficient Hyperparameter Optimization by Using Bayesian Optimization for Drug-Target Interaction Prediction"

A Bayesian optimization technique enables a short search time for a complex prediction model that includes many hyperparameters while maintaining the accuracy of the prediction model. Here, we apply a Bayesian optimization technique to the drug-target interaction (DTI) prediction problem as a method for computational drug discovery. We target neighborhood regularized logistic matrix factorization (NRLMF) (Liu _et al_., 2016), which is a state-of-the-art DTI prediction method, and accelerated parameter searches with the Gaussian process mutual information (GP-MI). Experimental results with four general benchmark datasets show that our GP-MI-based method obtained an 8.94-fold decrease in the computational time on average and almost the same predicted area under the curve (AUC) for all datasets compared to those of a grid parameter search, which was generally used in DTI predictions. Moreover, if a slight accuracy reduction (approximately 0.002 for AUC) is allowed, an increase in the calculation speed of 18 times or more can be obtained. Our results show for the first time that Bayesian optimization works effectively for the DTI prediction problem. By accelerating the time-consuming parameter search, the most advanced model can be used even if the number of drug candidates and target proteins to be predicted increase.


Requirements
------------

### Python
This script was created using Python 3.5.2 (Anaconda 2.4.0). For Python 3.5.2 please refer to the following URL.<br>
https://www.python.org/downloads/release/python-352/<br>

### Python packages
In addition, we use Numpy, scikit-learn (ver. 0.18.1 and above), scipy, pymatbridge (required only when using KBMF 2K) as Python package. For each package please refer to the following URL.<br>
−　Numpy: http://www.numpy.org/<br>
−　scikit-learn: http://scikit-learn.org/stable/<br>
−　scipy: http://www.scipy.org/<br>
−　pymatbridge: http://arokem.github.io/python-matlab-bridge/<br>

### Datasets
In order to execute the script, the Drug-Target Interaction data set created by Yamanishi et al. Is necessary. The data set can be downloaded from the following URL.<br>
http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/<br>
−　nr_admat_dgc.txt, nr_simmat_dc.txt, nr_simmat_dg.txt<br>
−　gpcr_admat_dgc.txt, gpcr_simmat_dc.txt, gpcr_simmat_dg.txt<br>
−　ic_admat_dgc.txt, ic_simmat_dc.txt, ic_simmat_dg.txt<br>
−　e_admat_dgc.txt, e_simmat_dc.txt, e_simmat_dg.txt<br>

Installation
------------
1. Download the archive of BO-DTI-master from this repository.
2. Extract the archive and cd into the extracted directory.
3. Run make command.

Commands:

    $ cd BO-DTI-master
    $ mkdir dataset
    $ cp ~/Downloads/*_admat_dgc.txt dataset
    $ cp ~/Downloads/*_simmat_dc.txt dataset
    $ cp ~/Downloads/*_simmat_dg.txt dataset

    
Usage
-----
You can specify the following options<br>
- gpmi ... GPMI algorithm can be used instead of grid search<br>
    - delta ... Adjust the balance between exploration and usage: delta > 0<br>
    - max_iter ... Specify the maximum value of iteration (number of combinations of parameters): max_iter > 0<br>
    - n_init ... Specify the initial number of samples: n_init > 0<br>
- seed ... Fix the division of cross validation
- job-id ... Specify the job id
- workdir ... ログファイルを出力するデSpecify the directory to output log filesィレクトリを指定する

For other, please refer to PyDTI


Example
-------
1. Command to execute grid search
```shell
$ python PyDTI.py --method="nrlmf" --dataset="nr" --cvs=1 --specify-arg=0 --predict-num=0 --seed="1" --job-id="1" --workdir="."
```

2. Command to execute GPMI algorithm
```shell
$ python PyDTI.py --method="nrlmf" --dataset="nr" --cvs=1 --specify-arg=0 --predict-num=0 --gpmi="delta=1e-100 max_iter=2688 n_init=1" --seed="1" --job-id="1" --workdir="."
```

Acknowledgement
---------------
This script was created based on PyDTI developed by Liu et al. PyDTI can be accessed from the following URL.<br>
https://github.com/stephenliu0423/PyDTI.git<br>

Contact
-------
This scripts was implemented by Tomohiro Ban.<br>
E-mail: ban@bi.c.titech.ac.jp

Department of Computer Science, Graduate School of Information Science and Engineering, Tokyo Institute of Technology, Japan<br>
http://www.bi.cs.titech.ac.jp/web/

If you have any questions, please feel free to contact the author.

References
----------
Tomohiro Ban, Masahito Ohue, Yutaka Akiyama: Efficient Hyperparameter Optimization by Using Bayesian Optimization for Drug-Target Interaction Prediction, In _Proceedings of the 7th IEEE International Conference on Computational Advances in Bio and Medical Sciences (ICCABS 2017)_, Orlando, FL, USA, October 19-21, 2017. (accepted) 

(Conference Website) http://www.iccabs.org/


Copyright © 2017 Akiyama_Laboratory, Tokyo Institute of Technology, All Rights Reserved.  
