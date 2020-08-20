#include "varh.h"

static PetscReal stepfunc(PetscReal dz, PetscReal eta, PetscReal beta){

  PetscReal b = (beta>1e-2) ? beta : 1e-2;
  
  PetscReal r1 = tanh(b*eta) + tanh(b*(dz -eta));
  PetscReal r2 = tanh(b*eta) + tanh(b*(1.0-eta));

  return 1.0-(r1/r2);

}

static PetscReal stepgrad(PetscReal dz, PetscReal eta, PetscReal beta){

  PetscReal b = (beta>1e-2) ? beta : 1e-2;
  
  PetscReal csch = 1.0/sinh(b);
  PetscReal sech = 1.0/cosh(b*(dz-eta));

  return b*csch*sech*sech*sinh(b*dz)*sinh(b-b*dz);

}

static PetscReal piecewise(PetscReal rho, PetscReal beta){

  PetscReal eta=0.5;
  PetscReal rout;

  if(beta<1e-3){

    rout=rho;

  }else{

    if(rho>=0 && rho<=eta)
      rout=eta * ( exp(-beta*(1-rho/eta)) - (1-rho/eta)*exp(-beta) );
    else if(rho>eta && rho<=1.0)
      rout=(1-eta) * ( 1 - exp(-beta*(rho-eta)/(1-eta)) + (rho-eta)*exp(-beta)/(1-eta) ) + eta;
    else if(rho<0)
      rout=0;
    else
      rout=1;

  }

  return rout;

}

static PetscReal piecewisegrad(PetscReal rho, PetscReal beta){

  PetscReal eta=0.5;
  PetscReal rg;

  if(beta<1e-3){

    rg =1.0;

  }else{

    if(rho>=0 && rho<=eta)
      rg =eta * ( (beta/eta)*exp(-beta*(1-rho/eta)) + exp(-beta)/eta );
    else if(rho>eta && rho<=1.0)
      rg =(1-eta) * ( beta/(1-eta) * exp(-beta*(rho-eta)/(1-eta)) + exp(-beta)/(1-eta) );
    else if(rho<0)
      rg =0;
    else
      rg =0;

  }

  return rg;

}

// the output indexing as ix,iz,ilayer from fastest to slowest index
// the input indexing as jx,jlayer 

#undef __FUNCT__
#define __FUNCT__ "varh_expand"
void varh_expand(PetscScalar *hdof, PetscScalar *edof, PetscInt nx, PetscInt ncells, PetscInt nlayers, PetscInt *mz, PetscReal beta, PetscInt zfixed)
{

  PetscInt Nx=nx*ncells;
  PetscInt id=0;
  for (PetscInt il=0;il<nlayers;il++){
    for (PetscInt iz=0;iz<mz[il];iz++){
      for (PetscInt ix=0;ix<Nx;ix++){

	if(zfixed==0){
	  PetscReal dz=(PetscReal)iz/(PetscReal)mz[il];
	  edof[id]=stepfunc(dz,creal(hdof[ix+Nx*il]),beta) + PETSC_i*0.0;
	}else{
	  edof[id]=piecewise(creal(hdof[ix+Nx*il]),beta) + PETSC_i*0.0;
	}
	
	id++;
      }
    }
  }
    

}


#undef __FUNCT__
#define __FUNCT__ "varh_contract"
void varh_contract(PetscScalar *egrad, PetscScalar *hgrad, PetscScalar *hdof, PetscInt nx, PetscInt ncells, PetscInt nlayers, PetscInt *mz, PetscReal beta, PetscInt zfixed)
{

  PetscInt Nx=nx*ncells;
  PetscInt id=0;
  for (PetscInt il=0;il<nlayers;il++){
    for (PetscInt iz=0;iz<mz[il];iz++){
      for (PetscInt ix=0;ix<Nx;ix++){

	if(zfixed==0){
	  PetscReal dz=(PetscReal)iz/(PetscReal)mz[il];
	  if(iz==0)
	    hgrad[ix+Nx*il]  = stepgrad(dz,creal(hdof[ix+Nx*il]),beta)*egrad[id];
	  else
	    hgrad[ix+Nx*il] += stepgrad(dz,creal(hdof[ix+Nx*il]),beta)*egrad[id];
	}else{
	  if(iz==0)
	    hgrad[ix+Nx*il]  = piecewisegrad(creal(hdof[ix+Nx*il]),beta)*egrad[id];
	  else
	    hgrad[ix+Nx*il] += piecewisegrad(creal(hdof[ix+Nx*il]),beta)*egrad[id];
	}
	
	id++;
      }
    }
  }
    

}

