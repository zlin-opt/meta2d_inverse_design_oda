#include "vec.h"

#undef __FUNCT__
#define __FUNCT__ "setlayer_eps"
void setlayer_eps(Vec eps,
		  PetscInt Nx, PetscInt Nz,
		  PetscInt num_layers, PetscInt *zstarts, PetscInt *thickness,
		  PetscScalar *epsilon)
{

  PetscScalar *_eps = (PetscScalar *)malloc(Nx*Nz*sizeof(PetscScalar));
  
  for(PetscInt iz=0;iz<Nz;iz++){
    for(PetscInt ix=0;ix<Nx;ix++){
      PetscInt i = ix+Nx*iz;
      _eps[i]=0;
      for(PetscInt j=0;j<num_layers;j++){
	if(iz>=zstarts[j] && iz<zstarts[j]+thickness[j])
	  _eps[i]=epsilon[j];
      }
    }
  }

  array2mpi(_eps,SCAL,eps);

  free(_eps);
  
}

#undef __FUNCT__
#define __FUNCT__ "vecfill_zslice"
void vecfill_zslice(Vec v, PetscScalar *_v, PetscInt Nx, PetscInt Nz, PetscInt iz)
{

  PetscInt ns,ne;
  VecGetOwnershipRange(v,&ns,&ne);
  for( PetscInt j=ns;j<ne;j++)
    {
      PetscInt k;
      PetscInt jx=(k=j)%Nx;
      PetscInt jz=(k/=Nx)%Nz;
      if(jz==iz)
	VecSetValue(v,j, _v[jx], INSERT_VALUES);
      
    }
  VecAssemblyBegin(v);
  VecAssemblyEnd(v);

}
