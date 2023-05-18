import optuna
import sympy,time
import numpy as np
import pandas as pd

class StopWhenAllSolutionAreFound:
    def __init__(self):
        self.used_step = None
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            #print(f'在尋解第{study.trials[-1].number}次時找到所有解',end='')
            self.used_step = study.trials[-1].number
            study.stop()

class Solution:
    def __init__(self,func=None, func_coef=None):
        self.x = sympy.symbols('x')
        if func==None:
            assert func_coef != None, '"func_coef" can not be None!'
            self.func = self.get_function(func_coef=func_coef, x=self.x)
        if func_coef==None:
            assert func != None, '"coef" can not be None!'
            self.func = func
        self.originfunc = self.func
        self.sol = []
    def get_function(self, func_coef,x):
        f = 0
        for power, coef in enumerate(func_coef, start=0):
            f += coef * x ** power
        return f
    def update_function(self, s):
        g = self.x-s
        self.sol.append(s)
        Q, R = sympy.div(self.func, g, domain='ZZ')
        self.func=Q
    def initfunc(self):
        self.func = self.originfunc
        self.sol = []

def get_random_function(solution_num=5, domain_max=10,  domain_min=-10):
    x = sympy.symbols('x')
    coef = np.random.randint(domain_min, domain_max, size=(solution_num, ))
    f = 1
    for c in coef:
        f = f*(x-c)
    return f.expand()

def objective(trial,s, domain_max=10,  domain_min=-10):
    x = trial.suggest_int('x', domain_min, domain_max)
    func = s.func
    if func==1:
        raise optuna.TrialPruned
    sol = func.evalf(subs={'x': x})
    if round(sol,10)==0:
        s.update_function(x)
    return abs(sol)

if __name__=='__main__':
    # s = Solution(func_coef = [12, 17, -1, -5, 1])
    dataframe = pd.DataFrame(
        columns=['solution_num','domain_max','domain_max','找到幾個解','用了幾個step找到所有解','n_trials','used_time (seconds)'])
    optuna.logging.set_verbosity(optuna.logging.FATAL)
    study_stop = StopWhenAllSolutionAreFound()
    # 'solution_num','domain_max','domain_max','found solution num','used step'
    for solution_num in [1,10,20,30,40,50,60,70,80,90,100]:
        for domain in [10,20,30,40,50,60,70,80,90,100]:
            print(f' sol:{solution_num}, domain:{domain}')
            # solutiwon_num = 20
            domain_max = domain
            domain_min = -domain
            n_trials = 10000
            func = get_random_function(solution_num=solution_num,domain_max=domain_max,domain_min=domain_min)
            s = Solution(func = func)

            for i in range(10):
                print(f' sol:{solution_num}, domain:{domain}, i:{i}')
                study = optuna.create_study(direction='minimize')
                t1 = time.time()
                study.optimize(lambda trial:objective(trial,s,domain_max=domain_max,domain_min=domain_min),
                               n_trials=n_trials,callbacks=[study_stop]) #50 find 7sol
                t2 = time.time()
                used_time = round(t2-t1,5)#;print(f' sol:{solution_num}, used time: {used_time} sec.')
                dataframe.loc[len(dataframe.index)] = [solution_num, domain_max, domain_min, len(s.sol), study_stop.used_step,n_trials,used_time]
                s.initfunc()
                del study

    dataframe.to_csv('./data.csv')
    #定義域大小 解的個數 n_trails的數