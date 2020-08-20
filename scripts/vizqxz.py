import numpy as np
import h5py as hp
import sys

filename=sys.argv[1]
Nx=int(sys.argv[2])
Nz=int(sys.argv[3])
outputfile=sys.argv[4]
epsout=int(sys.argv[5])

E=np.loadtxt(filename)

F=np.zeros((Nx,Nz),dtype=np.complex64)

for iz in range(Nz):
    for ix in range(Nx):
        i=ix + Nx*iz 
        F[ix,iz]=E[2*i+0] + 1j*E[2*i+1]
        

fid=hp.File(outputfile,'w')
if epsout==1:
    fid.create_dataset('eps',data=np.real(F))
else:
    fid.create_dataset('F.real',data=np.real(F))
    fid.create_dataset('F.imag',data=np.imag(F))
fid.close()
