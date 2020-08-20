#include "misc.h"

#undef __FUNCT__
#define __FUNCT__ "matmult_arrays"
void matmult_arrays(Mat W, void *_u, TYPE typein, void *_v, TYPE typeout, PetscInt transposeW)
{

  Vec u,v;
  if(transposeW==0)
    MatCreateVecs(W,&u,&v);
  else
    MatCreateVecs(W,&v,&u);
  array2mpi(_u,typein,u);
  PetscInt n;
  VecGetSize(v,&n);
  if(transposeW==0)
    MatMult(W,u,v);
  else
    MatMultTranspose(W,u,v);
  mpi2array(v,_v,typeout,n);
  VecDestroy(&u);
  VecDestroy(&v);

}
#undef __FUNCT__
#define __FUNCT__ "integer_sum"
PetscInt integer_sum(PetscInt *x, PetscInt i0, PetscInt i1)
{

  PetscInt result=0;
  for(PetscInt i=i0;i<i1;i++)
    result+=x[i];

  return result;

}


#undef __FUNCT__
#define __FUNCT__ "real_sum"
PetscReal real_sum(PetscReal *x, PetscInt i0, PetscInt i1)
{

  PetscReal result=0;
  for(PetscInt i=i0;i<i1;i++)
    result+=x[i];

  return result;

}

#undef __FUNCT__
#define __FUNCT__ "find_max"
PetscReal find_max(PetscReal *data, PetscInt n)
{

  PetscReal max = data[0];

  for (PetscInt i=0;i<n;i++)
    if (data[i] > max)
      max = data[i];

  return max;

}
