#include "solver.h"

#undef __FUNCT__
#define __FUNCT__ "setupKSPDirect"
PetscErrorCode setupKSPDirect(MPI_Comm comm, KSP *kspout, PetscInt maxit)
{
  PetscErrorCode ierr;
  KSP ksp;
  PC pc;

  ierr = KSPCreate(comm,&ksp);CHKERRQ(ierr);

  ierr = KSPSetType(ksp, KSPGMRES);CHKERRQ(ierr);
  //ierr = KSPSetType(ksp, KSPBCGS);CHKERRQ(ierr);

  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);

  ierr = KSPSetTolerances(ksp,1e-14,PETSC_DEFAULT,PETSC_DEFAULT,maxit);CHKERRQ(ierr);
  //ierr = KSPSetTolerances(ksp,1e-20,1e-20,PETSC_DEFAULT,maxit);CHKERRQ(ierr);

  ierr = PCSetFromOptions(pc);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  *kspout=ksp;

  PetscFunctionReturn(0);

}


#undef _FUNCT_
#define _FUNCT_ "SolveMatrixDirect"
PetscErrorCode SolveMatrixDirect(MPI_Comm comm, KSP ksp, Mat M, Vec b, Vec x, PetscInt *its, PetscInt maxit)
{
  /*-----------------KSP Solving------------------*/
  PetscErrorCode ierr;
  PetscLogDouble t1,t2,tpast;
  ierr = PetscTime(&t1);CHKERRQ(ierr);

  if (*its>(maxit-5)){
    ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);}
  else{
    ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);
    ierr = KSPSetReusePreconditioner(ksp,PETSC_TRUE);CHKERRQ(ierr);}

  ierr = PetscPrintf(comm,"==> initial-its is %d. maxit is %d.----\n ",*its,maxit);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(ksp,its);CHKERRQ(ierr);

  // if GMRES is stopped due to maxit, then redo it with sparse direct solve;
  if(*its>(maxit-2))
    {
      ierr = PetscPrintf(comm,"==> after-one-solve-its is %d. maxit is %d.----\n ",*its,maxit);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"==> Too Many Iterations. Re-solving with Sparse Direct Solver.\n");CHKERRQ(ierr);
      ierr = KSPSetOperators(ksp,M,M);CHKERRQ(ierr);
      ierr = KSPSetReusePreconditioner(ksp,PETSC_FALSE);CHKERRQ(ierr);
      ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(ksp,its);CHKERRQ(ierr);
    }

  //Print kspsolving information
  PetscReal norm;
  Vec xdiff;
  ierr=VecDuplicate(x,&xdiff);CHKERRQ(ierr);
  ierr = MatMult(M,x,xdiff);CHKERRQ(ierr);
  ierr = VecAXPY(xdiff,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(xdiff,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"==> Matrix solution: norm of error %g, Kryolv Iterations %d----\n ",norm,*its);CHKERRQ(ierr);

  ierr = PetscTime(&t2);CHKERRQ(ierr);
  tpast = t2 - t1;

  PetscPrintf(comm,"==> Matrix solution: the runing time is %f s \n",tpast);
  /*--------------Finish KSP Solving---------------*/

  VecDestroy(&xdiff);
  PetscFunctionReturn(0);
}
