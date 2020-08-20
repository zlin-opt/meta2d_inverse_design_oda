#include "adjoint.h"

void strehlpieces(MPI_Comm subcomm, PetscInt Nx, PetscInt Nz, PetscScalar *dof, Mat A,
		  Mat DDe, Vec epsDiff, Vec epsBkg, PetscReal omega,
		  KSP ksp, int *its, int maxit,
		  PetscScalar *Jsrc, PetscInt iz_src,
		  Vec ge, Vec gh, Mat Pdg,
		  PetscScalar *fEy, PetscScalar *fHx, PetscScalar *nPz, 
		  PetscScalar *fEy_grad, PetscScalar *fHx_grad, PetscScalar *nPz_grad)
{

  Vec _dof,eps;
  MatCreateVecs(A,&_dof,&eps);
  array2mpi(dof,SCAL,_dof);
  MatMult(A,_dof,eps);
  VecPointwiseMult(eps,eps,epsDiff);
  VecAXPY(eps,1.0,epsBkg);

  Mat M;
  MatDuplicate(DDe,MAT_COPY_VALUES,&M);
  VecScale(eps,-omega*omega);
  MatDiagonalSet(M,eps,ADD_VALUES);

  Vec b;
  VecDuplicate(eps,&b);
  VecSet(b,0);
  vecfill_zslice(b,Jsrc,Nx,Nz,iz_src);
  VecScale(b,PETSC_i*omega);

  Vec x;
  VecDuplicate(eps,&x);
  SolveMatrixDirect(subcomm,ksp,M,b,x,its,maxit);

  Vec u,tmp,grad;
  VecDuplicate(eps,&u);
  VecDuplicate(eps,&tmp);
  VecDuplicate(eps,&grad);
  PetscInt ndof;
  VecGetSize(_dof,&ndof);

  //Compute fEy = ge . x and its grad
  VecTDot(ge,x,fEy);

  KSPSolveTranspose(ksp,ge,u);
  VecPointwiseMult(grad,u,x);
  VecScale(grad,omega*omega);
  VecPointwiseMult(grad,grad,epsDiff);
  MatMultTranspose(A,grad,_dof);
  mpi2array(_dof,fEy_grad,SCAL,ndof);  

  //Compute fHx = gh . x and its grad
  VecTDot(gh,x,fHx);

  KSPSolveTranspose(ksp,gh,u);
  VecPointwiseMult(grad,u,x);
  VecScale(grad,omega*omega);
  VecPointwiseMult(grad,grad,epsDiff);
  MatMultTranspose(A,grad,_dof);
  mpi2array(_dof,fHx_grad,SCAL,ndof);  

  //Compute nPz = x* . Pdg . x and its grad 
  Vec xconj;
  VecDuplicate(eps,&xconj);
  VecCopy(x,xconj);
  VecConjugate(xconj);
  MatMult(Pdg,x,tmp);
  VecTDot(xconj,tmp,nPz);
  
  Mat conjPdg;
  MatDuplicate(Pdg,MAT_COPY_VALUES,&conjPdg);
  MatConjugate(conjPdg);

  MatMult(conjPdg,xconj,tmp);
  MatMultTranspose(Pdg,xconj,grad);
  VecAXPY(tmp,1.0,grad);

  KSPSolveTranspose(ksp,tmp,u);
  VecPointwiseMult(grad,u,x);
  VecScale(grad,omega*omega);
  VecPointwiseMult(grad,grad,epsDiff);
  MatMultTranspose(A,grad,_dof);
  mpi2array(_dof,nPz_grad,SCAL,ndof);  

  VecDestroy(&_dof);
  VecDestroy(&eps);
  VecDestroy(&b);
  VecDestroy(&x);
  VecDestroy(&xconj);
  VecDestroy(&u);
  VecDestroy(&tmp);
  VecDestroy(&grad);
  MatDestroy(&M);
  MatDestroy(&conjPdg);
  
}

