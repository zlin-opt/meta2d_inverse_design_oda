#ifndef GUARD_obj_h
#define GUARD_obj_h

#include "petsc.h"

typedef struct{

  PetscInt colour;
  MPI_Comm subcomm;
  PetscInt nsims_per_comm;
  
  PetscInt nfreqs;
  PetscInt nangles;
  
  PetscInt nx;
  PetscInt ncells;
  PetscInt px;
  PetscInt Nz;
  PetscInt nlayers_active;
  PetscInt *mz;

  Mat W;
  Mat A;
  
  PetscReal beta;
  PetscInt zfixed;

  Mat *DDe;
  Vec *epsDiff;
  Vec *epsBkg;
  PetscReal *omega;

  KSP *ksp;
  PetscInt *its;
  PetscInt maxit;
  PetscInt reuse_ksp;
  
  PetscScalar **Jsrc;
  PetscInt jz_src;

  Mat *Pdag;
  Vec *ge;
  Vec *gh;
  PetscReal *airyfactor;

  PetscInt print_at;

} params_;

#endif
