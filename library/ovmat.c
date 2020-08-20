#include "ovmat.h"

// rows are indexed ix,iz,ilayer,icell from fastest to slowest index
// columns are "globally" indexed jjx,jz,jlayer where jjx spans all the cells jjx = jx + nx*jcell

#undef __FUNCT__
#define __FUNCT__ "create_ovmat"
void create_ovmat(MPI_Comm comm, Mat *Wout, PetscInt *mrows_per_cell, PetscInt *cell_start, PetscInt nx, PetscInt px, PetscInt ncells, PetscInt nlayers, PetscInt pmlx0, PetscInt pmlx1, PetscInt *mz, PetscInt mzslab)
{

  //PetscPrintf(comm,"Creating the overlap extension matrix.\n");

  PetscInt mx =px+nx+px;

  PetscInt mrows=0;
  for(PetscInt j=0;j<ncells;j++){
    mrows_per_cell[j] = 0;
    for( PetscInt i=0;i<nlayers;i++)
      mrows_per_cell[j] += (mzslab==0) ? mx *mz[i] : mx  ;
    cell_start[j]=mrows;
    mrows += mrows_per_cell[j];
  }

  PetscInt idset_ix[mrows],idset_iz[mrows],idset_il[mrows],idset_ic[mrows];
  PetscInt id=0;
  for(PetscInt ic=0;ic<ncells;ic++){
    for(PetscInt il=0;il<nlayers;il++){
      PetscInt tmpmz = (mzslab==0) ? mz[il] : 1;
      for(PetscInt iz=0;iz<tmpmz;iz++){
	for(PetscInt ix=0;ix<mx;ix++){
	  idset_ix[id]=ix;
	  idset_iz[id]=iz;
	  idset_il[id]=il;
	  idset_ic[id]=ic;
	  id++;
	}
      }
    }
  }
  
  PetscInt Nx=nx*ncells;
  PetscInt ncols = 0;
  PetscInt nxnz[nlayers];
  for(PetscInt i=0;i<nlayers;i++){
    nxnz[i]= ncols;
    ncols += (mzslab==0) ? Nx*mz[i] : Nx ;
  }

  Mat W;

  PetscInt ns,ne;
  PetscInt i,j;
  PetscInt ix,iz,il,ic;
  PetscInt jx,jz,jl,jc,jjx;
  PetscScalar val;

  MatCreate(comm,&W);
  MatSetType(W,MATMPIAIJ);
  MatSetSizes(W,PETSC_DECIDE,PETSC_DECIDE, mrows,ncols);
  MatMPIAIJSetPreallocation(W, 1, PETSC_NULL, 1, PETSC_NULL);

  MatGetOwnershipRange(W, &ns, &ne);

  for(i=ns;i<ne;i++){

    val=1.0;

    ix=idset_ix[i];
    iz=idset_iz[i];
    il=idset_il[i];
    ic=idset_ic[i];
    
    if(ix < px){

      jc = ic-1;
      jx = nx - (px-ix); //jx = (nx - 1) - (px - 1 -ix);

    }else if(ix >= px+nx){

      jc = ic+1;
      jx = ix - (px+nx);

    }else{

      jc = ic;
      jx = ix - px;

    }

    if(jc>=ncells){

      jc  = jc - 1;
      val = 0.0;

    }

    if(jc<0){

      jc  = jc + 1;
      val = 0.0;

    }
    
    if( ix<pmlx0 || ix>=mx-pmlx1 )
      val=0.0;

    jl=il;
    jz=iz;
    jjx=jx+nx*jc;
    j=jjx+Nx*jz+nxnz[jl];

    MatSetValue(W,i,j,val,ADD_VALUES);

  }

  MatAssemblyBegin(W, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(W, MAT_FINAL_ASSEMBLY);

  *Wout = W;

}
