#include "array2vec.h"

#undef __FUNCT__
#define __FUNCT__ "array2mpi"
PetscErrorCode array2mpi(void *pt, TYPE typeinput, Vec v)
{
  PetscErrorCode ierr;
  PetscInt j, ns, ne;

  PetscReal *ptreal=PETSC_NULL;
  PetscScalar *ptscal=PETSC_NULL;
  if(typeinput==REAL)
    ptreal=(PetscReal *)pt;
  if(typeinput==SCAL)
    ptscal=(PetscScalar *)pt;
    
  ierr = VecGetOwnershipRange(v,&ns,&ne);
  for(j=ns;j<ne;j++){

    if(typeinput==REAL){
      if(ptreal!=PETSC_NULL)
	ierr=VecSetValue(v,j,ptreal[j]+PETSC_i*0.0,INSERT_VALUES);
      else
	SETERRQ(PETSC_COMM_WORLD,1,"ptreal is NULL in array2mpi function.");
    }
      
    if(typeinput==SCAL){
      if(ptscal!=PETSC_NULL)
	ierr=VecSetValue(v,j,ptscal[j],INSERT_VALUES);
      else
	SETERRQ(PETSC_COMM_WORLD,1,"ptscal is NULL in array2mpi function.");
    }
    
    CHKERRQ(ierr);
  }

  ierr = VecAssemblyBegin(v); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(v);  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "mpi2array"
PetscErrorCode mpi2array(Vec v, void *pt, TYPE typeoutput, PetscInt n)
{
  PetscErrorCode ierr;
  PetscInt i;
  PetscScalar *_a;
  Vec V_SEQ;
  VecScatter ctx;

  PetscReal *ptreal=PETSC_NULL;
  PetscScalar *ptscal=PETSC_NULL;
  if(typeoutput==REAL)
    ptreal=(PetscReal *)pt;
  if(typeoutput==SCAL)
    ptscal=(PetscScalar *)pt;
  
  ierr = VecScatterCreateToAll(v,&ctx,&V_SEQ);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,v,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,v,V_SEQ,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGetArray(V_SEQ,&_a);CHKERRQ(ierr);
  for (i = 0; i < n; i++){

    if(typeoutput==REAL){
      if(ptreal!=PETSC_NULL)
	ptreal[i] = creal(_a[i]);
      else
	SETERRQ(PETSC_COMM_WORLD,1,"ptreal is NULL in mpi2array function.");
    }

    if(typeoutput==SCAL){
      if(ptscal!=PETSC_NULL)
	ptscal[i] = _a[i];
      else
	SETERRQ(PETSC_COMM_WORLD,1,"ptscal is NULL in mpi2array function.");
    }
    
  }    
  ierr = VecRestoreArray(V_SEQ,&_a);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&V_SEQ);CHKERRQ(ierr);

  PetscFunctionReturn(0);
  
}
