import numpy as np
import scipy.stats as ss

d_before=[]
d_after=[]
with open('ivaniv.dat','r') as f:
  for s in f:
    val=s.split()
    d_before.append(float(val[0]))
    d_after.append(float(val[1]))

d_before=np.array(d_before)
d_after=np.array(d_after)
N=len(d_before)

a,b,c,d=0,0,0,0
for i in range(N):
  sup_before=d_before[i]>=50
  sup_after=d_after[i]>=50

  if sup_before and sup_after:
    a+=1
  elif not sup_before and sup_after:
    b+=1
  elif sup_before and not sup_after:
    c+=1
  else:
    d+=1

numerator=b-c
denominator=np.sqrt((b+c)-(b-c)**2 / N)

z=numerator/denominator

pv = 2 * (1 - ss.norm.cdf(abs(z)))

print(f"Изменили мнение: 0->1: {b}, 1->0: {c}")
print(f"Z-статистика: {z:.4f}")
print(f"PV: {pv:.4f}")

# Уровень поддержки не изменился, так как pv >0.05

diff=d_after-d_before
xm=diff.mean()
m0=0
s0=diff.std()
T=(xm-m0)/(s0/np.sqrt(N))

dist=ss.t(df=N-1)
pv_t=(1-dist.cdf(abs(T)))*2

print(f"Средняя разница рейтингов: {xm}")
print(f"T-статистика: {T}, PV: {pv_t}")

# Средний рейтинг значимо снизился, так как pv<0.05 и T-критерий<0