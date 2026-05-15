#调试1：L, Pw, Pc = predict_inductor(inputs)的使用

from F_ANN_CPU import predict_inductor

#inputs:dc1(mm), dc2(mm), ht(mm), lg1(mm), nx, ny, c(mm), f(kHz), i(A)

inputs=(15, 5, 5, 0.6, 3, 2, 3, 400, 9 )

L, Pw, Pc = predict_inductor(inputs)

print(f"L = {L:.3f} uH, Pw = {Pw:.3f} W, Pc = {Pc:.3f} W")