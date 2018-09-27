
import time
import logging
from functions import *
from nrlmf import NRLMF
from netlaprls import NetLapRLS
from blm import BLMNII
from wnngip import WNNGIP
#from kbmf import KBMF
from cmf import CMF
from sklearn.gaussian_process import GaussianProcessRegressor

def nrlmf_cv_eval(method,dataset,cv_data,intMat,Kd,Kt,cvs,para,logger,scoring='auc',gpmi=None):

    # GP-MI (Bayesian optimization)
    if gpmi is not None:

        # Generate parameters
        params_grid, x_grid = list(), list()
        for r in [50, 100]:
            for x in np.arange(-5, 2):
                for y in np.arange(-5, 3):
                    for z in np.arange(-5, 1):
                        for t in np.arange(-3, 1):
                            params_grid.append({'cfix':para['c'],'K1':para['K1'],'K2':para['K2'],'num_factors':r,'lambda_d':2.**(x),'lambda_t':2.**(x),'alpha':2.**(y),'beta':2.**(z),'theta':2.**(t),'max_iter':100})
                            x_grid.append([para['c'],para['K1'],para['K2'],r,2.**(x),2.**(x),2.**(y),2.**(z),2.**(t),100])
                            
        # Initialization
        start = time.time()                            
        n_init = int(gpmi['n_init']) if gpmi['n_init'] > 0 else 1
        best_score = 0
        count = 1
        if n_init > 0:
            np.random.seed(list(cv_data.keys())[0])
            i_init = np.random.permutation(range(len(params_grid)))[0:n_init]
            X = np.array([x_grid[i] for i in i_init])
            y = np.array(list())
            for i in i_init:
                tic = time.clock()
                params_next = params_grid[i]
                model = NRLMF(**params_next)
                aupr_vec, auc_vec = train(model, cv_data, intMat, Kd, Kt)
                aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                y_next = auc_avg if scoring=='auc' else aupr_avg
                y = np.concatenate((y,[y_next]),axis=0)
                logger.info("%s %s cvs=%s (sample= %s) %.6f[sec]" % (params_grid[i],scoring,str(cvs),str(count),time.clock()-tic))
                logger.info(str(y_next))
                if i == 0:
                    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                    best_params, best_score = params_grid[i], y_next
                    auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
                if best_score < y_next:
                    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                    best_params, best_score = params_grid[i], y_next
                    auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
                count += 1

        # GP-MI algorithm
        alpha = np.log(2/gpmi['delta'])
        gamma = 0
        for i in range(int(gpmi['max_iter'])):
            tic = time.clock()
            gp = GaussianProcessRegressor()
            gp.fit(X,y)
            mean, sig = gp.predict(x_grid,return_std=True)
            phi = np.sqrt(alpha) * (np.sqrt(sig**2+gamma)-np.sqrt(gamma))
            idx = np.argmax(mean+phi)
            params_next = params_grid[idx]
            x_next = x_grid[idx]
            gamma = gamma + sig[idx]**2
            model = NRLMF(**params_next)
            aupr_vec, auc_vec = train(model, cv_data, intMat, Kd, Kt)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)            
            y_next = auc_avg if scoring=='auc' else aupr_avg
            logger.info("%s %s cvs=%s (sample= %s) %.6f[sec]" % (params_grid[i],scoring,str(cvs),str(i+n_init+1),time.clock()-tic))
            logger.info(str(y_next))
            if best_score < y_next:
                cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                best_params, best_score = params_next, y_next
                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
            if np.array_equal(x_next,X[-1]): break
            X = np.concatenate((X,[x_next]),axis=0)
            y = np.concatenate((y,[y_next]),axis=0)

        end = time.time()
        cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
        cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, time:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4], end-start)

    # Grid search
    else:
        start = time.clock()
        max_auc, auc_opt = 0, []
        for r in [50, 100]:
            for x in np.arange(-5, 2):
                for y in np.arange(-5, 3):
                    for z in np.arange(-5, 1):
                        for t in np.arange(-3, 1):
                            tic = time.clock()
                            model = NRLMF(cfix=para['c'], K1=para['K1'], K2=para['K2'], num_factors=r, lambda_d=2.**(x), lambda_t=2.**(x), alpha=2.**(y), beta=2.**(z), theta=2.**(t), max_iter=100)
                            cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                            logger.info(cmd)
                            aupr_vec, auc_vec = train(model, cv_data, intMat, Kd, Kt)
                            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                            logger.info("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
                            if auc_avg > max_auc:
                                max_auc = auc_avg
                                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]

        end = time.clock()
        cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
        cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, time:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4], end-start)

    logger.info('')        
    logger.info(cmd)


def netlaprls_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for x in np.arange(-6, 3):  # [-6, 2]
        for y in np.arange(-6, 3):  # [-6, 2]
            tic = time.clock()
            model = NetLapRLS(gamma_d=10**(x), gamma_t=10**(x), beta_d=10**(y), beta_t=10**(y))
            cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
            print(cmd)
            aupr_vec, auc_vec = train(model, cv_data, X, D, T)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
            if auc_avg > max_auc:
                max_auc = auc_avg
                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def blmnii_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for x in np.arange(0, 1.1, 0.1):
        tic = time.clock()
        model = BLMNII(alpha=x, avg=False)
        cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
        print(cmd)
        aupr_vec, auc_vec = train(model, cv_data, X, D, T)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
        if auc_avg > max_auc:
            max_auc = auc_avg
            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def wnngip_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for x in np.arange(0.1, 1.1, 0.1):
        for y in np.arange(0.0, 1.1, 0.1):
            tic = time.clock()
            model = WNNGIP(T=x, sigma=1, alpha=y)
            cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
            print(cmd)
            aupr_vec, auc_vec = train(model, cv_data, X, D, T)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
            if auc_avg > max_auc:
                max_auc = auc_avg
                auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def kbmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_auc, auc_opt = 0, []
    for d in [50, 100]:
        tic = time.clock()
        model = KBMF(num_factors=d)
        cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
        print(cmd)
        aupr_vec, auc_vec = train(model, cv_data, X, D, T)
        aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
        auc_avg, auc_conf = mean_confidence_interval(auc_vec)
        print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
        if auc_avg > max_auc:
            max_auc = auc_avg
            auc_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % auc_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (auc_opt[1], auc_opt[2], auc_opt[3], auc_opt[4])
    print(cmd)


def cmf_cv_eval(method, dataset, cv_data, X, D, T, cvs, para):
    max_aupr, aupr_opt = 0, []
    for d in [50, 100]:
        for x in np.arange(-2, -1):
            for y in np.arange(-3, -2):
                for z in np.arange(-3, -2):
                    tic = time.clock()
                    model = CMF(K=d, lambda_l=2.**(x), lambda_d=2.**(y), lambda_t=2.**(z), max_iter=30)
                    cmd = "Dataset:"+dataset+" CVS: "+str(cvs)+"\n"+str(model)
                    print(cmd)
                    aupr_vec, auc_vec = train(model, cv_data, X, D, T)
                    aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
                    auc_avg, auc_conf = mean_confidence_interval(auc_vec)
                    print("auc:%.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f, Time:%.6f\n" % (auc_avg, aupr_avg, auc_conf, aupr_conf, time.clock()-tic))
                    if aupr_avg > max_aupr:
                        max_aupr = aupr_avg
                        aupr_opt = [cmd, auc_avg, aupr_avg, auc_conf, aupr_conf]
    cmd = "Optimal parameter setting:\n%s\n" % aupr_opt[0]
    cmd += "auc: %.6f, aupr: %.6f, auc_conf:%.6f, aupr_conf:%.6f\n" % (aupr_opt[1], aupr_opt[2], aupr_opt[3], aupr_opt[4])
    print(cmd)
