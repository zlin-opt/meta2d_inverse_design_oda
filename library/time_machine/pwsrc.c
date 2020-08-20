#include "pwsrc.h"

void srcJy(PetscScalar *Jy,
	   PetscInt ix_start, PetscInt nx_segment, PetscReal dx, PetscReal dz,
	   PetscReal kwav, PetscReal alpha, 
	   PetscScalar Ay)
{

  
  for (PetscInt ix=0;ix<nx_segment;ix++){

    PetscReal x=(ix_start+ix)*dx;
    Jy[ix] = (Ay/dz) * cexp( PETSC_i * kwav * sin(alpha) * x );

  }

}

