#include "filters.h"

//NOTE that the cells are contiguous and the indexing is global using Nx = nx*ncells

#undef __FUNCT__
#define __FUNCT__ "density_filter"
void density_filter(MPI_Comm comm, Mat *Qout, PetscInt Nx, PetscInt nlayers, PetscReal filt_radius, PetscReal sigma, PetscInt normalized)
{

  PetscPrintf(comm,"Creating the density filter. NOTE: the radius filt_radius must be greater than 0. filt_radius <= 1 means no filter.\n");

  PetscInt ncols=Nx*nlayers;
  PetscInt nxows=ncols;

  Mat Q;

  PetscInt box_size=2*ceil(filt_radius)-1;

  MatCreate(comm,&Q);
  MatSetType(Q,MATRIX_TYPE);
  MatSetSizes(Q,PETSC_DECIDE,PETSC_DECIDE, nxows,ncols);
  MatMPIAIJSetPreallocation(Q, box_size, PETSC_NULL, box_size, PETSC_NULL);

  PetscInt ns,ne;
  MatGetOwnershipRange(Q, &ns, &ne);

  PetscInt *cols = (PetscInt *)malloc(box_size*sizeof(PetscInt));
  PetscScalar *weights = (PetscScalar *)malloc(box_size*sizeof(PetscScalar));

  for(PetscInt i=ns;i<ne;i++){

    PetscInt k;
    PetscInt ix=(k=i)%Nx;
    PetscInt il=(k/=Nx)%nlayers;

    PetscInt jx_min=ix-ceil(filt_radius)+1;
    PetscInt jx_max=ix+ceil(filt_radius);

    PetscInt ind=0;
    PetscScalar norm=0.0+PETSC_i*0.0;
    
    for(PetscInt jx=jx_min;jx<jx_max;jx++){

      PetscInt jjx,jjl;
      
      if(jx < 0)
	jjx = jx + Nx; //-jx; 
      else if(jx >= Nx)
	jjx = jx - Nx; //2*Nx - jx - 2;
      else
	jjx = jx;

      jjl=il;
      
      PetscInt j=jjx+Nx*jjl;
      PetscReal dist2=pow(ix-jx,2);
      PetscReal sigma2=pow(sigma,2);
      cols[ind]=j;
      weights[ind]=exp(-dist2/sigma2)+PETSC_i*0.0;
      norm += weights[ind];
      ind++;

    }

    if(normalized==1)
      for(PetscInt j=0;j<ind;j++) weights[j]/=norm;

    MatSetValues(Q, 1, &i, ind, cols, weights, ADD_VALUES);
    
  }

  MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY);

  free(cols);
  free(weights);
  
  *Qout = Q;

}


#undef __FUNCT__
#define __FUNCT__ "pixbunch"
void pixbunch(MPI_Comm comm, Mat *Bout, PetscInt Nx, PetscInt nlayers, PetscInt pix_rad)
{

  PetscPrintf(comm,"Creating the pixel multiplier. A pixel-bunch is %d . \n",pix_rad);
  if(pix_rad<1)
    PetscPrintf(comm,"ERROR: pix_rad can't be less than 1. \n");

  PetscInt nrows=Nx*nlayers;
  PetscInt Px=Nx/pix_rad;
  PetscInt ncols=Px*nlayers;

  Mat B;

  MatCreate(comm,&B);
  MatSetType(B,MATMPIAIJ);
  MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  MatMPIAIJSetPreallocation(B, 1, PETSC_NULL, 1, PETSC_NULL);

  PetscInt ns,ne;
  MatGetOwnershipRange(B, &ns, &ne);

  for(PetscInt i=ns;i<ne;i++){

    PetscInt k;
    PetscInt ix=(k=i)%Nx;
    PetscInt il=(k/=Nx)%nlayers;
    
    PetscInt jx=ix/pix_rad;

    PetscInt j=jx+Px*il;
    
    MatSetValue(B, i, j, 1.0, INSERT_VALUES);

  }

  MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);

  *Bout = B;

}


#undef __FUNCT__
#define __FUNCT__ "mirrormat"
void mirrormat(MPI_Comm comm, Mat *Qout, PetscInt Nx, PetscInt nlayers)
{

  PetscPrintf(comm,"Creating the mirror matrix. \n");

  PetscInt nrows=1*Nx*nlayers;
  PetscInt ncols=2*Nx*nlayers;

  Mat Q;

  MatCreate(comm,&Q);
  MatSetType(Q,MATMPIAIJ);
  MatSetSizes(Q,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  MatMPIAIJSetPreallocation(Q, 4, PETSC_NULL, 4, PETSC_NULL);

  PetscInt ns,ne;
  MatGetOwnershipRange(Q, &ns, &ne);

  for(PetscInt i=ns;i<ne;i++){

    PetscInt k;
    PetscInt ix=(k=i)%Nx;
    PetscInt il=(k/=Nx)%nlayers;
    
    PetscInt jx1=ix;
    PetscInt jx2=2*Nx-ix-1;
    PetscInt jl = il;

    PetscInt j1 = jx1 + 2*Nx*jl;
    PetscInt j2 = jx2 + 2*Nx*jl;

    PetscInt mcols=2;
    PetscInt cols[2]={j1,j2};
    PetscScalar vals[2]={1,1};

    MatSetValues(Q,1,&i,mcols,cols,vals,INSERT_VALUES);

  }

  MatAssemblyBegin(Q, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Q, MAT_FINAL_ASSEMBLY);

  Mat QT;
  MatTranspose(Q,MAT_INITIAL_MATRIX,&QT);

  *Qout = QT;
  MatDestroy(&Q);

}
