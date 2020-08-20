import numpy as np

x=np.loadtxt('objvals')
tmp=np.unique(x[:,0])
x1=tmp.astype(int)
x2=np.zeros(x1.size)
for i in range(x1.size):
    ind=np.nonzero(x[:,0]==x1[i])
    x2[i]=x[ind,1].min()
np.savetxt('maxvals.dat',x2)
print max(x2)

    
