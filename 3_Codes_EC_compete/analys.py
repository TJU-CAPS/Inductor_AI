import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 ['SimHei'] 若用中文
matplotlib.rcParams['axes.unicode_minus'] = False

dc2= np.arange(1, 4, 0.1)
dc1= 8
ht = 3.14
lg1 =np.arange(0.7,4,0.1)
Nx=2
Ny=5
c=1
f=100
i=15

L=10

V=(dc1+4*0.7+2*dc2+2*Nx*c)*(dc1+2*0.7+2*Nx*c)*(2*ht+2*0.7+Nx*c) #单位mm3

lg=8*np.pi* 4*1e-7*Nx*Ny*dc1*dc1/(L*(8+np.pi*dc1*dc1/dc2))

Pc=