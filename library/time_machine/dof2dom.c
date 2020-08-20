#include "dof2dom.h"

//create a matrix that transforms the design-region array to full-grid domain array for epsilon

//Note that the indexing in the design-region array is chosen as ix,iz,ilayer in the order of the fastest to slowest
//In contrast, the indexing in the full-grid domain is chosen as ix,iz in the order of the fastest to slowest

#undef __FUNCT__
#define __FUNCT__ "create_Ainterp"
void create_Ainterp(MPI_Comm comm, Mat *A_out,
		    PetscInt Nx, PetscInt Nz,
		    PetscInt Mxo, PetscInt Mx, PetscInt *Mzo, PetscInt *Mz, PetscInt Mzslab,
		    PetscInt nlayers)
{

  PetscInt comm_size;
  MPI_Comm_size(comm,&comm_size);
  
  PetscInt nrows = 1*Nx*Nz;
  PetscInt ncols = 0;
  PetscInt mxmz[nlayers];
  for( PetscInt i=0;i<nlayers;i++){
    mxmz[i]=ncols;
    ncols += (Mzslab==0) ? Mx*Mz[i] : Mx;
  }
    
  //PetscPrintf(comm,"Creating the interpolation matrix A to embed the design-region into the full-domain\n");

  Mat A;

  MatCreate(comm,&A);
  MatSetType(A,MATRIX_TYPE);
  MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  if(comm_size==1)
    MatSeqAIJSetPreallocation(A, 1, PETSC_NULL);
  else
    MatMPIAIJSetPreallocation(A, 1, PETSC_NULL, 1, PETSC_NULL);
  
  PetscInt ns,ne;
  MatGetOwnershipRange(A, &ns, &ne);

  for(PetscInt i=ns;i<ne;i++){
    
    PetscInt k;
    PetscInt ix=(k=i)%Nx;
    PetscInt iz=(k/=Nx)%Nz;

    if(ix>=Mxo && ix<Mxo+Mx){
      PetscInt px = ix-Mxo;
      for(PetscInt ilayer=0;ilayer<nlayers;ilayer++){
	if(iz>=Mzo[ilayer] && iz<Mzo[ilayer]+Mz[ilayer]){
	  PetscInt indp;
	  if(Mzslab==0)
	    indp = px + Mx*(iz-Mzo[ilayer]) + mxmz[ilayer];
	  else
	    indp = px + Mx*ilayer;
	  MatSetValue(A,i,indp,1.0+PETSC_i*0.0,INSERT_VALUES);
	}
      }
    }
  }
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  
  *A_out = A;

}
