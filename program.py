import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

dane=pd.read_excel('datas.xlsx')
test=pd.read_excel('testdatas.xlsx')

dane=np.array(dane)
test=np.array(test)

X,Y=np.split(dane,[4],axis=1)
# print(X)
# print(Y)
X_test,Y_test=np.split(test,[4],axis=1)
# print(X_test)
# print(Y_test)

#skalowanie danych

scaler1=StandardScaler()
scaler2=StandardScaler()
scaler1.fit(X)
scaler2.fit(Y)
X_sc=scaler1.transform(X)
Y_sc=scaler2.transform(Y)
X_sc,Y_sc=shuffle(X_sc,Y_sc)
X_test_sc=scaler1.transform(X_test)
Y_test_sc=scaler2.transform(Y_test)

# param_grid1={'n_estimators':[10,100,1000,10000], 'max_depth':[5,10,100,500,1000]}
# param_grid2={'activation':['logistic','tanh','relu'],'solver':['adam','sgd'],
#               'alpha':[10,1,0.1,0.01,0.001,0.0001,0.00001],'max_iter':[2000,5000,10000]}
# grid_search1=GridSearchCV(RandomForestRegressor(max_features="log2"), param_grid1,cv=10)
# grid_search2=GridSearchCV(MLPRegressor(hidden_layer_sizes=(50,50,50,50)), param_grid2,cv=10)

# print('RFR')
# grid_search1.fit(X_sc,Y_sc)
# print("Wynik w zestawie testowym: {:.3f}".format(grid_search1.score(X_test_sc,Y_test_sc)))
# print("Najlepsze parametry: {}".format(grid_search1.best_params_))
# print("Najlepszy wynik walidacji krzyzowej: {:.3f}".format(grid_search1.best_score_))

# print('\nMLPR')
# grid_search2.fit(X_sc,Y_sc)
# print("Wynik w zestawie testowym: {:.3f}".format(grid_search2.score(X_test_sc,Y_test_sc)))
# print("Najlepsze parametry: {}".format(grid_search2.best_params_))
# print("Najlepszy wynik walidacji krzyzowej: {:.3f}".format(grid_search2.best_score_))


#Uczenie lasami losowymi

RFR=RandomForestRegressor(n_estimators=1000, max_features="log2",max_depth=100) 
#Najlepsze parametry to n_estimators=10000 max_depth=1000 ale brak widocznego wzrostu dokladnosci ze znacznie wiekszym czasem obliczen
RFR.fit(X_sc,Y_sc)
print("Dokladnosc RFR w zestawie uczacym:   {:.3f}".format(RFR.score(X_sc,Y_sc)))
print("Dokladnosc RFR w zestawie testowym:  {:.3f}".format(RFR.score(X_test_sc,Y_test_sc)))
Y1=RFR.predict(X_test_sc)

#sieci neuronowe

mlpr=MLPRegressor(hidden_layer_sizes=(50,50,50,50),solver="adam",max_iter=10000,
                  activation="tanh"
                  ,random_state=0,alpha=0.001)
mlpr.fit(X_sc,Y_sc)
print("Dokladnosc MLPR w zestawie uczacym:  {:.3f}".format(mlpr.score(X_sc,Y_sc)))
print("Dokladnosc MLPR w zestawie testowym: {:.3f}".format(mlpr.score(X_test_sc,Y_test_sc)))
Y3=mlpr.predict(X_test_sc)

#wykres dla wybranego swiatlowodu testowego

plt.figure(figsize=(8,6),dpi=100)
plt.plot(scaler1.inverse_transform(X_test_sc)[66:77,3]
          ,scaler2.inverse_transform(Y_test_sc)[66:77,0]
          ,label='test')
plt.plot(scaler1.inverse_transform(X_test_sc)[66:77,3]
          ,scaler2.inverse_transform(Y1)[66:77,0]
          ,label='predict RFR')
plt.plot(scaler1.inverse_transform(X_test_sc)[66:77,3]
          ,scaler2.inverse_transform(Y3)[66:77,0]
          ,label='predict MLPR')
# plt.yscale("log")
plt.legend(fontsize=15)
plt.xlabel(r'Wavelength $[\mathrm{\mu m}]$',fontsize=15)
plt.ylabel(r'$n_\mathrm{eff}$',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()

plt.figure(figsize=(8,6),dpi=100)
plt.plot(scaler1.inverse_transform(X_test_sc)[66:77,3]
          ,scaler2.inverse_transform(Y_test_sc)[66:77,1]
          ,label='test')
plt.plot(scaler1.inverse_transform(X_test_sc)[66:77,3]
          ,scaler2.inverse_transform(Y1)[66:77,1]
          ,label='predict RFR')
plt.plot(scaler1.inverse_transform(X_test_sc)[66:77,3]
          ,scaler2.inverse_transform(Y3)[66:77,1]
          ,label='predict MLPR')
plt.legend(fontsize=15)
plt.xlabel(r'Wavelength $[\mathrm{\mu m}]$',fontsize=15)
plt.ylabel(r'$A_\mathrm{eff} [\mathrm{\mu m}^2]$',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()

print('\nParamtery swiatlowodu:\n')
print('Liczba pierscieni:\t      {:.0f}'.format(X_test[66,0]))
print('Stala sieci:              {:.3f} [um]'.format(X_test[66,1]))
print('Wspolczynnik wypelnienia: {:.3f}'.format(X_test[66,2]))

#Zestawienia wartosci przewidywanych z faktycznymi

plt.figure(figsize=(8,6),dpi=100)
plt.scatter(scaler2.inverse_transform(Y_test_sc)[:,0],
         scaler2.inverse_transform(Y1)[:,0],5,'C1',label='predict RFR')
plt.scatter(scaler2.inverse_transform(Y_test_sc)[:,0],
         scaler2.inverse_transform(Y3)[:,0],5,'C2',label='predict MLPR')
plt.plot([1.2,1.45],[1.2,1.45],'k-',linewidth=1)
plt.xlim((1.2,1.45))
plt.ylim((1.2,1.45))
plt.legend(fontsize=15)
plt.xlabel(r'actual $n_\mathrm{eff}$',fontsize=15)
plt.ylabel(r'predicted $n_\mathrm{eff}$',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()


plt.figure(figsize=(8,6),dpi=100)
plt.scatter(scaler2.inverse_transform(Y_test_sc)[:,1],
         scaler2.inverse_transform(Y1)[:,1],5,'C1',label='predict RFR')
plt.scatter(scaler2.inverse_transform(Y_test_sc)[:,1],
         scaler2.inverse_transform(Y3)[:,1],5,'C2',label='predict MLPR')
plt.plot([0.7,6.1],[0.7,6.1],'k-',linewidth=1)
plt.legend(fontsize=15)
plt.xlim((0.7,6.1))
plt.ylim((0.7,6.1))
plt.xlabel(r'actual $A_\mathrm{eff} [\mathrm{\mu m}^2]$',fontsize=15)
plt.ylabel(r'predicted $A_\mathrm{eff} [\mathrm{\mu m}^2]$',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()

#roznice miedzy wartosciami przewidywanymi a faktycznymi


plt.subplots(2,1,sharex=True,figsize=(8,6),dpi=100)
plt.subplot(2,1,1)
plt.plot(scaler1.inverse_transform(X_test_sc)[:,3]
          ,np.abs(scaler2.inverse_transform(Y1)[:,0]-scaler2.inverse_transform(Y_test_sc)[:,0])
          ,'C1.',label='predict RFR')
plt.legend(fontsize=15)
plt.ylabel(r'$\Delta n_\mathrm{eff}$',fontsize=15)
plt.yticks(fontsize=13)
plt.subplot(2,1,2)
plt.plot(scaler1.inverse_transform(X_test_sc)[:,3]
          ,np.abs(scaler2.inverse_transform(Y3)[:,0]-scaler2.inverse_transform(Y_test_sc)[:,0])
          ,'C2.',label='predict MLPR')
plt.legend(fontsize=15)
plt.xlabel(r'Wavelength $[\mathrm{\mu m}]$',fontsize=15)
plt.ylabel(r'$\Delta n_\mathrm{eff}$',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()

plt.subplots(2,1,sharex=True,figsize=(8,6),dpi=100)
plt.subplot(2,1,1)
plt.plot(scaler1.inverse_transform(X_test_sc)[:,3]
          ,np.abs(scaler2.inverse_transform(Y1)[:,1]-scaler2.inverse_transform(Y_test_sc)[:,1])
          ,'C1.',label='predict RFR')
plt.legend(fontsize=15)
plt.ylabel(r'$\Delta A_\mathrm{eff} [\mathrm{\mu m}^2]$',fontsize=15)
plt.yticks(fontsize=13)
plt.subplot(2,1,2)
plt.plot(scaler1.inverse_transform(X_test_sc)[:,3]
          ,np.abs(scaler2.inverse_transform(Y3)[:,1]-scaler2.inverse_transform(Y_test_sc)[:,1])
          ,'C2.',label='predict MLPR')
# plt.yscale("log")
plt.legend(fontsize=15)
plt.xlabel(r'Wavelength $[\mathrm{\mu m}]$',fontsize=15)
plt.ylabel(r'$\Delta A_\mathrm{eff} [\mathrm{\mu m}^2]$',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()

#prosty model
Xn=np.array([X_test[66:77,3]]).T
Yn=Y_test[66:77,0:2]
Xn_test=np.array([[1.05],[1.15],[1.25],[1.35],[1.45],[1.55],[1.65],[1.75],[1.85],[1.95]])

scaler3=StandardScaler()
scaler4=StandardScaler()
scaler3.fit(Xn)
scaler4.fit(Yn)
Xn_sc=scaler3.transform(Xn)
Yn_sc=scaler4.transform(Yn)
Xn_test_sc=scaler3.transform(Xn_test)

RFR2=RandomForestRegressor(n_estimators=1000, max_features="log2", max_depth=100)
RFR2.fit(Xn_sc,Yn_sc)
Yn1=RFR2.predict(Xn_test_sc)

mlpr2=MLPRegressor(hidden_layer_sizes=(50,50,50,50),solver="adam",max_iter=10000,
                  activation="relu"
                  ,random_state=0,alpha=0.001)
mlpr2.fit(Xn_sc,Yn_sc)
Yn3=mlpr2.predict(Xn_test_sc)

plt.figure(figsize=(8,6),dpi=100)
plt.plot(scaler3.inverse_transform(Xn_sc)
          ,scaler4.inverse_transform(Yn_sc)[:,0]
          ,label='test')
plt.plot(scaler3.inverse_transform(Xn_test_sc)
          ,scaler4.inverse_transform(Yn1)[:,0]
          ,label='predict RFR')
plt.plot(scaler3.inverse_transform(Xn_test_sc)
          ,scaler4.inverse_transform(Yn3)[:,0]
          ,label='predict MLPR')
plt.legend(fontsize=15)
plt.xlabel(r'Wavelength $[\mathrm{\mu m}]$',fontsize=15)
plt.ylabel(r'$n_\mathrm{eff}$',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()

plt.figure(figsize=(8,6),dpi=100)
plt.plot(scaler3.inverse_transform(Xn_sc)
          ,scaler4.inverse_transform(Yn_sc)[:,1]
          ,label='test')
plt.plot(scaler3.inverse_transform(Xn_test_sc)
          ,scaler4.inverse_transform(Yn1)[:,1]
          ,label='predict RFR')
plt.plot(scaler3.inverse_transform(Xn_test_sc)
          ,scaler4.inverse_transform(Yn3)[:,1]
          ,label='predict MLPR')
plt.legend(fontsize=15)
plt.xlabel(r'Wavelength $[\mathrm{\mu m}]$',fontsize=15)
plt.ylabel(r'predicted $A_\mathrm{eff} [\mathrm{\mu m}^2]$',fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()
