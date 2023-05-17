# import optuna
#
# def objective(trial):
#     x1 = trial.suggest_int('x1', -10, 10)
#     # x2 = trial.suggest_int('x2', -100, 100)
#     # x3 = trial.suggest_int('x3', -100, 100)
#
#     return abs((x1**3)-(3*x1**2)-(33*x1)+35)
#
# study = optuna.create_study(direction='minimize')
# study.optimize(objective,n_trials=50)
#
# # print(study.trials)
# # print(study.best_trial.params)
# # print(study.best_value)
# sol=[]
# for i in study.best_trials:
#     sol.append(i.params['x1'])
#
# print(set(sol))


from sympy import *
x = symbols('x')
f = x**5+x**4-x**2-1
g = x-1
Q,R = div(f,g,domain='ZZ')
print(Q) #除以多項式
print(f.evalf(subs={'x':1}),type(float(f.evalf(subs={'x':1})))) #帶入求解


a=[1,0,2,3] #常數 一次項係數 二次項係數
f=0
for power,coef in enumerate(a,start=0):
    f+=coef*x**power
print(f)

plot(f,ylim=(-10,10),xlim=(-10,10))

# import optuna
# def objective(trial):
#     x1 = trial.suggest_float('x1', -10, 10)
#     ans = f.evalf(subs={'x':x1})
#     return abs(ans)
#
# study = optuna.create_study(direction='minimize')
# study.optimize(objective,n_trials=50)
#
# sol=[]
# for i in study.best_trials:
#     sol.append(i.params['x1'])
#
# print(set(sol))
