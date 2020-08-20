#include "farfield.h"

PetscScalar hk1(PetscInt n, PetscReal x){

  return jn(n,x) + PETSC_i * yn(n,x); 

}

void GEH(PetscReal  x, PetscReal  z,
	 PetscReal x0, PetscReal z0, 
	 PetscReal omega, PetscReal mu, PetscReal eps,
	 PetscScalar *fEe, PetscScalar *fEh,
	 PetscScalar *fHe, PetscScalar *fHh)
{

  PetscReal k  = omega * sqrt(mu*eps);
  
  PetscReal Rx = x - x0;
  PetscReal Rz = z - z0;
  PetscReal Rr = sqrt(Rx*Rx + Rz*Rz);

  //PetscScalar Ik = PETSC_i*k;
  PetscScalar Iw = PETSC_i*omega;
  PetscReal k2 = pow(k,2);
  PetscReal Rr2 = pow(Rr,2);
  PetscReal Rr3 = pow(Rr,3);
  PetscScalar i4 = PETSC_i/4;
  
  //PetscScalar g  = cexp(PETSC_i * k *Rr)/(4*M_PI*Rr);
  //PetscScalar gz = Rz * (-1 + PETSC_i * k * Rr)*cexp(PETSC_i * k * Rr)/(4*M_PI*Rr3);
  //PetscScalar gxx = ( 3*pow(Rx,2)/Rr3 - 3*Ik*pow(Rx,2)/Rr2 - (1+pow(k*Rx,2))/Rr + Ik ) * cexp(PETSC_i * k * Rr)/(4*M_PI*Rr2);

  PetscScalar g   = i4 * hk1(0,k*Rr);
  PetscScalar gz  = i4 * (-k*Rz/Rr) * hk1(1,k*Rr);
  PetscScalar gxx = i4 * ( -k2*Rx*Rx*hk1(0,k*Rr)/Rr2 + k*(Rx-Rz)*(Rx+Rz)*hk1(1,k*Rr)/Rr3 );
  
  *fEe =  mu * (-1/eps) * gz ;
  *fEh =  mu * Iw * g ; 
  *fHe = eps * Iw * g + eps * (Iw/k2) * gxx ;
  *fHh = eps * (-1/mu)  * gz ;
    
}

