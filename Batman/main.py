import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

f=open('buildings.dat.txt') # dz/Batman/buildings.dat.txt
f.readline()
d=[]

for s in f:
    d.append([int(x) for x in s.split()])

d=np.array(d)
clf=DBSCAN(eps=300,min_samples=2)
c=clf.fit_predict(d)

for i in range(len(c)):
  if c[i]==-1:
    print(f"Здание Во: X={d[i][0]}, Y={d[i][1]}, H={d[i][2]}")

plt.scatter(d[:,0],d[:,1],c=c)
plt.show()