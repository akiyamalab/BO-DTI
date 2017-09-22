# Scripts for "Efficient Hyperparameter Optimization by Using Bayesian Optimization for Drug-Target Interaction Prediction"

A Bayesian optimization technique enables a short search time for a complex prediction model that includes many hyperparameters while maintaining the accuracy of the prediction model. Here, we apply a Bayesian optimization technique to the drug-target interaction (DTI) prediction problem as a method for computational drug discovery. We target neighborhood regularized logistic matrix factorization (NRLMF) (Liu _et al_., 2016), which is a state-of-the-art DTI prediction method, and accelerated parameter searches with the Gaussian process mutual information (GP-MI). Experimental results with four general benchmark datasets show that our GP-MI-based method obtained an 8.94-fold decrease in the computational time on average and almost the same predicted area under the curve (AUC) for all datasets compared to those of a grid parameter search, which was generally used in DTI predictions. Moreover, if a slight accuracy reduction (approximately 0.002 for AUC) is allowed, an increase in the calculation speed of 18 times or more can be obtained. Our results show for the first time that Bayesian optimization works effectively for the DTI prediction problem. By accelerating the time-consuming parameter search, the most advanced model can be used even if the number of drug candidates and target proteins to be predicted increase.


Requirements
------------
このスクリプトは、Liuらが開発したPyDTIに基づいて作られた。PyDTIへは次のURLからアクセスすることができる。<br>
https://github.com/stephenliu0423/PyDTI.git

このスクリプトは、Python 3.5.2 (Anaconda 2.4.0)を使用して作られた。Python3.5.2に関しては次のURLを参照してください。<br>
- Python 3.5.2: https://www.python.org/downloads/release/python-352/

また、Pythonパッケージとして、Numpy、scikit-learn (ver. 0.18.1 以上)、scipy、pymatbridge (KBMF2Kを使用する場合のみ必要)を用いている。それぞれのパッケージについては次のURLを参照してください。
- Numpy: http://www.numpy.org/
- scikit-learn: http://scikit-learn.org/stable/
- scipy: http://www.scipy.org/
- pymatbridge: http://arokem.github.io/python-matlab-bridge/

スクリプトを実行するためには、Yamanishiらが作成したDrug-Target Interactionデータセットが必要である。データセットは次のURLからダウンロードすることができる。<br>
http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/
- Nuclear receptor  : nr_admat_dgc.txt, nr_simmat_dc.txt, nr_simmat_dg.txt
- GPCR              : gpcr_admat_dgc.txt, gpcr_simmat_dc.txt, gpcr_simmat_dg.txt
- Ion channel       : ic_admat_dgc.txt, ic_simmat_dc.txt, ic_simmat_dg.txt
- Enzyme            :e_admat_dgc.txt, e_simmat_dc.txt, e_simmat_dg.txt

Installation
------------
1. Download the archive of xx from this repository.
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


Example
-------

    $ python PyDTI.py --method="nrlmf" --dataset="nr" --cvs=1 --specify-arg=0 --predict-num=0 --gpmi="delta=1e-100 max_iter=2688 n_init=1" --seed="1" --job-id="1"
    $ 
    

References
----------
Tomohiro Ban, Masahito Ohue, Yutaka Akiyama: Efficient Hyperparameter Optimization by Using Bayesian Optimization for Drug-Target Interaction Prediction, In _Proceedings of the 7th IEEE International Conference on Computational Advances in Bio and Medical Sciences (ICCABS 2017)_, Orlando, FL, USA, October 19-21, 2017. (accepted) 

(Conference Website) http://www.iccabs.org/


Copyright © 2017 Akiyama_Laboratory, Tokyo Institute of Technology, All Rights Reserved.  
