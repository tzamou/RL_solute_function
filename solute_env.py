import optuna
import sympy
import numpy as np

class Create_csv:
    def __init__(self):
        import pandas as pd
        self.dataframe = pd.DataFrame(columns=['solution_num','domain_max','domain_max','found solution num','used step'])
class StopWhenAllSolutionAreFound:
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            print(f'在尋解第{study.trials[-1].number}次時找到所有解')
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

# s = Solution(func_coef = [12, 17, -1, -5, 1])
lensol=[]
optuna.logging.set_verbosity(optuna.logging.FATAL)
study_stop = StopWhenAllSolutionAreFound()

domain_max = 30
domain_min = -30
func = get_random_function(solution_num=70,domain_max=domain_max,domain_min=domain_min)
s = Solution(func = func)

for i in range(1):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial:objective(trial,s,domain_max=domain_max,domain_min=domain_min),
                   n_trials=1000,callbacks=[study_stop]) #50 find 7sol

    print(sorted(s.sol))
    print(len(s.sol))
    lensol.append(len(s.sol))
    s.initfunc()

print(lensol)
#定義域大小 解的個數 n_trails的數
