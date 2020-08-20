import numpy as np
x=np.loadtxt("tmp")
y=np.gradient(x[:,1],x[:,0])
z=np.transpose([x[:,0],x[:,2],y])
np.savetxt("tmp",z)
