from Data_Generate import inputs
from F_ANN_CPU import  predict_inductor
import pandas as pd

samples = inputs(n=10)

results = []
for x in samples:
    L, Pw, Pc = predict_inductor(x)
    results.append((*x, L, Pw, Pc))

cols = ['dc1', 'dc2', 'ht', 'lg1', 'Nx', 'Ny', 'c', 'f', 'i', 'L(uH)', 'Pw(W)', 'Pc(W)']
pd.DataFrame(results, columns=cols).to_csv("inductor_predictions_100k.csv", index=False)
print("✅ 已保存：inductor_predictions_100k.csv")

