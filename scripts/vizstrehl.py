import numpy as np
import h5py as hp
import sys
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

filename=sys.argv[1]
outputfilename=sys.argv[2]
freq=float(sys.argv[3])
meepflux=float(sys.argv[4])
Nx=5000
Nxfar=5000
dx=0.05
dxfar=0.05
flen=80
NA=np.sin(np.arctan(Nx*dx/(2*flen)))
FWHM=0.88/(2*freq*NA)

xfar=np.linspace(-Nxfar*dxfar/2,Nxfar*dxfar/2,Nxfar)

            
fid=hp.File(filename,"r")
Effx=np.array(fid["ex.r"])+1j*np.array(fid["ex.i"])
Effy=np.array(fid["ey.r"])+1j*np.array(fid["ey.i"])
Effz=np.array(fid["ez.r"])+1j*np.array(fid["ez.i"])
Hffx=np.array(fid["hx.r"])+1j*np.array(fid["hx.i"])
Hffy=np.array(fid["hy.r"])+1j*np.array(fid["hy.i"])
Hffz=np.array(fid["hz.r"])+1j*np.array(fid["hz.i"])
fid.close()

Sy = np.real(np.multiply(Effz,np.conj(Hffx))) - np.real(np.multiply(Effx,np.conj(Hffz)))
Symax = np.max(Sy)
Sysum = np.sum(Sy)*dx
Sy=Sy/Symax
tmpSy=Sy-0.5

Syinterp=interp1d(xfar,tmpSy)
x0=fsolve(Syinterp,-FWHM/2)[0]
x1=fsolve(Syinterp, FWHM/2)[0]

print "\n"

print "Designed NA: "+str(NA)
print "Designed FWHM: "+str(FWHM)
print "Measured FWHM: "+str(x1-x0)
print "Sy at focal spot: "+str(Symax)
print "flux at focal plane: " + str(Sysum)

print "\n"

airymax = 0.885893*Sysum/FWHM
print "fPz adjusted Strehl: " + str(Symax/airymax)

airymax2 = 0.885893*meepflux/FWHM
print "Meep adjusted Strehl: " + str(Symax/airymax2)

airymax3 = 0.885893*Sysum/(x1-x0)
print "measured-FWHM adjusted Strehl: " + str(Symax/airymax3)

airymax4 = 0.885893*meepflux/(x1-x0)
print "Meep and measured-FWHM adjusted Strehl: " + str(Symax/airymax4)

print "\n"

np.savetxt(outputfilename+'.dat',Sy)

idealairy=np.zeros(Nxfar)
i=0
for xx in xfar:
    xxx=xx/np.pi
    idealairy[i]=np.power(np.sinc((2.783114756503021/(x1-x0))*xxx),2)
    i=i+1
    
np.savetxt('idealairy.dat',idealairy)
