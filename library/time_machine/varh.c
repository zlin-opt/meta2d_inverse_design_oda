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
	  edof[id]=hdof[ix+Nx*il];
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
	    hgrad[ix+Nx*il]  = egrad[id];
	  else
	    hgrad[ix+Nx*il] += egrad[id];
	}
	
	id++;
      }
    }
  }
    

}

#undef __FUNCT__
#define __FUNCT__ "density_filter"
void density_filter(MPI_Comm comm, Mat *Qout, PetscInt nx, PetscInt ncells, PetscInt nlayers, PetscReal filt_radius, PetscReal sigma, PetscInt normalized)
{

  PetscPrintf(comm,"Creating the density filter. NOTE: the radius filt_radius must be greater than 0. filt_radius <= 1 means no filter.\n");

  PetscInt Nx=nx*ncells;
  PetscInt ncols=Nx*nlayers;
  PetscInt nxows=ncols;

  Mat Q;

  PetscInt box_size=2*ceil(filt_radius)-1;

  MatCreate(comm,&Q);
  MatSetType(Q,MATRIX_TYPE);
  MatSetSizes(Q,PETSC_DECIDE,PETSC_DECIDE, nxows,ncols);
  MatMPIAIJSetPreallocation(Q, box_size, PETSC_NULL, box_size, PETSC_NULL);

  PetscInt ns,ne;
  MatGetOwnershipRange(Q, &ns, &ne);

  PetscInt *cols = (PetscInt *)malloc(box_size*sizeof(PetscInt));
  PetscScalar *weights = (PetscScalar *)malloc(box_size*sizeof(PetscScalar));

  for(PetscInt i=ns;i<ne;i++){

    PetscInt k;
    PetscInt ix=(k=i)%Nx;
    PetscInt il=(k/=Nx)%nlayers;

    PetscInt jx_min=ix-ceil(filt_radius)+1;
    PetscInt jx_max=ix+ceil(filt_radius);

    PetscInt ind=0;
    PetscScalar norm=0.0+PETSC_i*0.0;
    
    for(PetscInt jx=jx_min;jx<jx_max;jx++){

      PetscInt jjx,jjl;
      
      if(jx < 0)
	jjx = jx + Nx; //-jx; 
      else if(jx >= Nx)
	jjx = jx - Nx; //2*Nx - jx - 2;
      else
	jjx = jx;

      jjl=il;
      
      PetscInt j=jjx+Nx*jjl;
      PetscReal dist2=pow(ix-jx,2);
      PetscReal sigma2=pow(sigma,2);
      cols[ind]=j;
      weights[ind]=exp(-dist2/sigma2)+PETSC_i*0.0;
      norm += weights[ind];
      ind++;

    }

    if(normalized==1)
      for(PetscInt j=0;j<ind;j++) weights[j]/=norm;

    MatSetValues(Q, 1, &i, ind, cols, weights, ADD_VALUES);
    
  }

  MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY);

  free(cols);
  free(weights);
  
  *Qout = Q;

}
