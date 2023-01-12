import pandas as pd
import pprint
import math
import matplotlib.pyplot as plt

tau = 240
b = 0.2
loss = 1
tolerance = 10 ** (-4)
learn_rate = 0.0001
iteration = 100

path = 'code/result/drying_ratio.xlsx'

table = pd.read_excel(path,header=[0],dtype=float).dropna()
def f0(time, tau, b):
    return (1-b)*math.exp(-time/tau) + b
def dif_f0_tau(time, tau, b):
    return (1-b)*(math.exp(-time/tau))*(-time/(tau**2))
def dif_f0_b(time, tau, b):
    return (-1)*math.exp(time/tau) + 1


def calc_loss(data, time, tau, b):
    loss = 0
    dif_tau = 0
    dif_b = 0
    for d,t in zip(data,time):
        estimate = f0(t, tau,b)
        dif = estimate - d
        dif_tau += 2*dif*dif_f0_tau(t,tau,b)*learn_rate
        dif_b += 2*dif*dif_f0_b(t,tau,b)*learn_rate
        loss += dif**2
        # print(f"data:{d}:time{t}--tau:{dif_tau},b:{dif_b}--estimate{estimate}:loss{loss}")
    return loss, tau - dif_tau/len(data), b - dif_b/len(data)

i = 0 
while loss > tolerance:
    i+=1
    loss, tau, b = calc_loss(table['1_6'],table['time'],tau,b)
    print(f"tau:{tau}, b:{b}, loss{loss}")
    if i > iteration:
        break
    # loss = 0


# x = [i*10 for i in range(100)]
# y0 = [f0(xi,tau,b) for xi in x]
# y1 = [dif_f0_tau(xi,tau,b) for xi in x]
# y2 = [dif_f0_b(xi,tau,b) for xi in x]

# fig, ax = plt.subplots(1, 1)
# ax.set_title('all in one')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.plot(x,y0)
# ax.plot(x,y1)
# ax.plot(x,y2)
# plt.show()