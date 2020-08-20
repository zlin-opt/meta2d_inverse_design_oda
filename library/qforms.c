#include "qforms.h"

// P = (-dx/(I omega)) Omega Sh Dze 
// dagger = 0, 1 or 2: no action, complex transpose or transpose only
// ix_lstart means local start ix (i.e. the actual starting index for the given array)
// ix_gstart means global start ix (i.e. the global start index of the given segment which is part of a much bigger array)

#undef __FUNCT__
#define __FUNCT__ "create_P"
void create_P(MPI_Comm comm, Mat *P_out, PetscInt dagger,
	      PetscInt nx, PetscInt nz,
	      PetscInt npmlx0, PetscInt npmlx1, PetscInt npmlz0, PetscInt npmlz1,
	      PetscReal dx, PetscReal dz,
	      PetscReal omega,
	      PetscInt ix_lstart, PetscInt nx_mtr, PetscInt iz_mtr)
{

  Mat Dze;
  create_Dze(comm, &Dze,
	     nx, nz,
	     npmlx0, npmlx1, npmlz0, npmlz1,
	     dx,dz,
	     omega);
  Mat Sh;
  syncHx(comm, &Sh, nx, nz);

  PetscScalar *tmp = (PetscScalar *)malloc(nx*nz*sizeof(PetscScalar));
  for(PetscInt iz=0;iz<nz;iz++){
    for(PetscInt ix=0;ix<nx;ix++){
      PetscInt i = ix + nx * iz;
      if( ix>=ix_lstart && ix<ix_lstart+nx_mtr && iz==iz_mtr )
	tmp[i] = -dx/(PETSC_i * omega);
      else
	tmp[i] = 0;
    }
  }
  Vec _tmp;
  MatCreateVecs(Sh,PETSC_NULL,&_tmp);
  array2mpi(tmp,SCAL, _tmp);
  
  Mat Omega;
  MatDuplicate(Sh, MAT_DO_NOT_COPY_VALUES, &Omega);
  MatDiagonalSet(Omega, _tmp, INSERT_VALUES);

  Mat SD, P;
  MatMatMult(Sh,Dze,   MAT_INITIAL_MATRIX, 5.0/(2.0+2.0), &SD);
  MatMatMult(Omega,SD, MAT_INITIAL_MATRIX, 5.0/(2.0+2.0),  &P);

  Mat Pt,Pdg;
  if(dagger==1){
    MatHermitianTranspose(P, MAT_INITIAL_MATRIX, &Pdg);
    *P_out = Pdg;
  }if(dagger==2){
    MatTranspose(P, MAT_INITIAL_MATRIX, &Pt);
    *P_out = Pt;
  }else{
    *P_out = P;
  }
  
  MatDestroy(&Dze);
  MatDestroy(&Sh);
  free(tmp);
  VecDestroy(&_tmp);
  MatDestroy(&Omega);
  MatDestroy(&SD);

}

#undef __FUNCT__
#define __FUNCT__ "gforms"
void gforms(MPI_Comm comm, Vec ge, Vec gh, 
	    PetscInt nx, PetscInt nz,
	    PetscInt npmlx0, PetscInt npmlx1, PetscInt npmlz0, PetscInt npmlz1,
	    PetscReal dx, PetscReal dz,
	    PetscReal omega,
	    PetscInt ix_lstart, PetscInt nx_mtr, PetscInt ix_gstart, PetscInt iz_mtr,
	    PetscReal xfar, PetscReal zfar,
	    PetscReal origx, PetscReal origz)
{

  Mat P;
  create_P(comm, &P, 0,
	   nx,nz,
	   npmlx0,npmlx1,npmlz0,npmlz1,
	   dx,dz,
	   omega,
	   ix_lstart,nx_mtr,iz_mtr);

  Vec GEe, GEh, GHe, GHh;
  VecDuplicate(ge,&GEe);
  VecDuplicate(ge,&GEh);
  VecDuplicate(ge,&GHe);
  VecDuplicate(ge,&GHh);

  PetscScalar *_GEe,*_GEh,*_GHe,*_GHh;
  VecGetArray(GEe,&_GEe);
  VecGetArray(GEh,&_GEh);
  VecGetArray(GHe,&_GHe);
  VecGetArray(GHh,&_GHh);

  PetscInt ns,ne;
  VecGetOwnershipRange(GEe,&ns,&ne);

  for(PetscInt i=ns;i<ne;i++){

    PetscInt k;
    PetscInt ix = (k=i)%nx;
    PetscInt iz = (k/=nx)%nz;

    PetscInt i_local=i-ns;
    if( ix>=ix_lstart && ix<ix_lstart+nx_mtr && iz==iz_mtr ){

      PetscReal xnear = (ix_gstart + ix - ix_lstart)*dx - origx;
      PetscReal znear = 0 - origz;
      PetscScalar fEe,fEh,fHe,fHh;
      GEH(xfar, zfar,
	  xnear, znear,
	  omega, 1,1,
	  &fEe,&fEh,
	  &fHe,&fHh);
      _GEe[i_local] = fEe*dx;
      _GEh[i_local] = fEh;
      _GHe[i_local] = fHe*dx;
      _GHh[i_local] = fHh;

    
    }else{

      _GEe[i_local]=0;
      _GEh[i_local]=0;
      _GHe[i_local]=0;
      _GHh[i_local]=0;

    } 
    
  }
  
  VecRestoreArray(GEe,&_GEe);
  VecRestoreArray(GEh,&_GEh);
  VecRestoreArray(GHe,&_GHe);
  VecRestoreArray(GHh,&_GHh);
    
  MatMultTranspose(P,GEh,ge);
  MatMultTranspose(P,GHh,gh);

  VecAYPX(ge, -1.0, GEe);
  VecAYPX(gh, -1.0, GHe);

  VecDestroy(&GEe);
  VecDestroy(&GEh);
  VecDestroy(&GHe);
  VecDestroy(&GHh);
  MatDestroy(&P);
  
}
