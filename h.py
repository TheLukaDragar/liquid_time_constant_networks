import pandas as pd


#open 2 csv files

d1 = pd.read_csv('/d/hpc/home/ld8435/liquid_time_constant_networks/predictionsss.csv', parse_dates=True, index_col=0)
d2 = pd.read_csv('/d/hpc/home/ld8435/liquid_time_constant_networks/pppp.csv', parse_dates=True, index_col=0)


print(d1.shape)
print(d1.head())

print(d2.shape)
print(d2.head())

#check how many colums are the same

print(d1.columns.intersection(d2.columns))
print(d1["PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE"])
print(d2["PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE"])

print("difference", d1["PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE"] - d2["PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE"])

d2.update(d1)

print(d2["PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE"])

print(d2.shape)
print(d2.head())

#save to csv

d2.to_csv('/d/hpc/home/ld8435/liquid_time_constant_networks/merged.csv')



