#include "adjoint.h"

void strehlpieces(MPI_Comm subcomm, PetscInt Nx, PetscInt Nz, PetscScalar *dof, Mat A,
		  Mat DDe, Vec epsDiff, Vec epsBkg, PetscReal omega,
		  KSP ksp, int *its, int maxit,
		  PetscScalar *Jsrc, PetscInt iz_src,
		  Vec ge, Vec gh, Mat Pdg,
		  PetscScalar *fEy, PetscScalar *fHx, PetscScalar *nPz, 
		  PetscScalar *fEy_grad, PetscScalar *fHx_grad, PetscScalar *nPz_grad)
{

  PetscScalar tmpdebug;
  
  Vec _dof,eps;
  MatCreateVecs(A,&_dof,&eps);
  array2mpi(dof,SCAL,_dof);
  MatMult(A,_dof,eps);
  VecPointwiseMult(eps,eps,epsDiff);
  VecAXPY(eps,1.0,epsBkg);

  Mat M;
  MatDuplicate(DDe,MAT_COPY_VALUES,&M);
  VecScale(eps,-omega*omega);

  VecSum(eps,&tmpdebug);
  PetscPrintf(PETSC_COMM_WORLD,"debug Job1: sum_eps %g %g \n",creal(tmpdebug),cimag(tmpdebug));
  
  MatDiagonalSet(M,eps,ADD_VALUES);

  Vec b;
  VecDuplicate(eps,&b);
  VecSet(b,0);
  vecfill_zslice(b,Jsrc,Nx,Nz,iz_src);
  VecScale(b,PETSC_i*omega);

  VecSum(b,&tmpdebug);
  PetscPrintf(PETSC_COMM_WORLD,"debug Job1: sum_b %g %g \n",creal(tmpdebug),cimag(tmpdebug));
  
  Vec x;
  VecDuplicate(eps,&x);
  SolveMatrixDirect(subcomm,ksp,M,b,x,its,maxit);

  VecSum(x,&tmpdebug);
  PetscPrintf(PETSC_COMM_WORLD,"debug Job1: sum_Efield %g %g \n",creal(tmpdebug),cimag(tmpdebug));
  
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

  PetscPrintf(PETSC_COMM_WORLD,"debug Job1: fEy %g %g \n",creal(*fEy),cimag(*fEy));
  PetscPrintf(PETSC_COMM_WORLD,"debug Job1: fHx %g %g \n",creal(*fHx),cimag(*fHx));
  PetscPrintf(PETSC_COMM_WORLD,"debug Job1: nPz %g \n",creal(*nPz));

  PetscReal fSz = creal( - conj(*fHx) * *fEy );
  PetscPrintf(PETSC_COMM_WORLD,"debug Job1: fSz %g \n",fSz);
  PetscPrintf(PETSC_COMM_WORLD,"debug Job1: fSz/nPz %g \n",fSz/creal(*nPz));
  
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

PetscReal strehlval(PetscScalar fEy, PetscScalar fHx, PetscScalar nPz, PetscReal airyfactor,
		    PetscScalar *fEy_grad, PetscScalar *fHx_grad, PetscScalar *nPz_grad,
		    PetscReal *strehlgrad, PetscInt ndof)
{

  PetscReal fSz = creal( - conj(fHx) * fEy );
  PetscReal Pz = creal(nPz);
  PetscReal strehl = airyfactor * (fSz/Pz);

  PetscPrintf(PETSC_COMM_WORLD,"debug Job1: airyfactor %g \n",airyfactor);
  PetscPrintf(PETSC_COMM_WORLD,"debug Job1 v2: fSz %g \n",fSz);
  PetscPrintf(PETSC_COMM_WORLD,"debug Job1 v2: nPz %g \n",creal(nPz));
  PetscPrintf(PETSC_COMM_WORLD,"debug Job1: strehl %g \n",strehl);
  
  for(PetscInt i=0;i<ndof;i++){
    PetscReal fSz_grad = creal( - conj( fHx_grad[i] ) * fEy - conj(fHx) * fEy_grad[i] );
    PetscReal Pz_grad = creal( nPz_grad[i] );
    strehlgrad[i] = airyfactor * ( fSz_grad/Pz - (fSz/pow(Pz,2)) * Pz_grad );
  }
  
  return strehl;

}

PetscReal fSzval(PetscScalar fEy, PetscScalar fHx, 
		 PetscScalar *fEy_grad, PetscScalar *fHx_grad, 
		 PetscReal *fSzgrad, PetscInt ndof)
{

  PetscReal fSz = creal( - conj(fHx) * fEy );

  for(PetscInt i=0;i<ndof;i++)
    fSzgrad[i] = creal( - conj( fHx_grad[i] ) * fEy - conj(fHx) * fEy_grad[i] );
  
  return fSz;

}
