#include "output.h"

#undef __FUNCT__
#define __FUNCT__ "writetofile"
PetscErrorCode writetofile(MPI_Comm comm, char *filename, void *data, TYPE typeout, PetscInt n)
{

  PetscInt rank;
  MPI_Comm_rank(comm, &rank);

  PetscReal *ptreal=PETSC_NULL;
  PetscScalar *ptscal=PETSC_NULL;
  if(typeout==REAL)
    ptreal=(PetscReal *)data;
  if(typeout==SCAL)
    ptscal=(PetscScalar *)data;

  if(rank==0){
    FILE *ptf;
    ptf = fopen(filename,"w");
    for (PetscInt i=0;i<n;i++){

      if(typeout==REAL){
	if(ptreal!=PETSC_NULL)
	  fprintf(ptf,"%.16g \n",ptreal[i]);
	else
	  SETERRQ(PETSC_COMM_WORLD,1,"ptreal is NULL in writetofile function.");
      }
      
      if(typeout==SCAL){
	if(ptscal!=PETSC_NULL){
	  fprintf(ptf,"%.16g\n",creal(ptscal[i]));
	  fprintf(ptf,"%.16g\n",cimag(ptscal[i]));
	}else{
	  SETERRQ(PETSC_COMM_WORLD,1,"ptscal is NULL in writetofile function.");
	}
      }

    }
    fclose(ptf);
  }
  MPI_Barrier(comm);

  PetscFunctionReturn(0);
  
}
