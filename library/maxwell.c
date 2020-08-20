#include "maxwell.h"

#define alpha 2
#define Refl cexp(-16)
#define n0bkg 1.0

#undef __FUNCT__
#define __FUNCT__ "pml_s"
PetscScalar pml_s(PetscInt N, PetscInt Npml0, PetscInt Npml1, PetscReal i, PetscReal delta, PetscReal omega)
{

  PetscReal p=i*delta;
  PetscReal ps=Npml0*delta;
  PetscReal pb=(N-Npml1)*delta;

  PetscReal d0=Npml0*delta;
  PetscReal d1=Npml1*delta;
  PetscReal l,Lpml;
  if (p<ps)
    l=ps-p,Lpml=d0; 
  else if (p>pb)
    l=p-pb,Lpml=d1;
  else
    l=0,Lpml=1;

  PetscReal lnR=log(Refl);
  PetscReal sigma=-(alpha+1)*lnR/(2*n0bkg*omega*Lpml);

  PetscScalar s = 1.0 + PETSC_i * sigma * pow(l/Lpml,alpha);

  return s;

}

//Note that the indexing chosen as ix,iz,ic in the order of the fastest to slowest

#undef __FUNCT__
#define __FUNCT__ "create_Dze"
void create_Dze(MPI_Comm comm, Mat *Dze_out,
		PetscInt Nx, PetscInt Nz,
		PetscInt Npmlx0, PetscInt Npmlx1, PetscInt Npmlz0, PetscInt Npmlz1,
		PetscReal dx, PetscReal dz, 
		PetscReal omega)
{

  PetscInt comm_size;
  MPI_Comm_size(comm,&comm_size);
  
  PetscInt nrows = Nx*Nz;
  PetscInt ncols = Nx*Nz;
  
  //PetscPrintf(comm,"Creating the matrix Dze (which operates on Ey).\n");

  Mat Dze;

  MatCreate(comm,&Dze);
  MatSetType(Dze,MATRIX_TYPE);
  MatSetSizes(Dze,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  if(comm_size==1)
    MatSeqAIJSetPreallocation(Dze, 2, PETSC_NULL);
  else
    MatMPIAIJSetPreallocation(Dze, 2, PETSC_NULL, 2, PETSC_NULL);
  
  PetscInt ns,ne;
  MatGetOwnershipRange(Dze, &ns, &ne);

  for(PetscInt i=ns;i<ne;i++){
    
    PetscInt k;
    PetscInt ix=(k=i)%Nx;
    PetscInt iz=(k/=Nx)%Nz;

    //-dEy/dz
    PetscScalar sz=pml_s(Nz, Npmlz0,Npmlz1, iz+0.5, dz, omega);
    PetscInt jx0=ix;
    PetscInt jz0=(iz<Nz-1) ? iz+1 : 0;
    PetscInt jx1=ix;
    PetscInt jz1=iz;

    PetscScalar val0 = -1.0/(sz*dz);
    PetscScalar val1 = +1.0/(sz*dz);
      
    PetscInt j0=jx0 + Nx*jz0;
    PetscInt j1=jx1 + Nx*jz1;
    PetscInt mcols=2;
    PetscInt jcols[2]={j0,j1};
    PetscScalar vals[2]={val0,val1};
    
    MatSetValues(Dze, 1, &i, mcols, jcols, vals, ADD_VALUES);

  }
    
  MatAssemblyBegin(Dze, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Dze, MAT_FINAL_ASSEMBLY);
  
  *Dze_out = Dze;

}

#undef __FUNCT__
#define __FUNCT__ "create_Dxe"
void create_Dxe(MPI_Comm comm, Mat *Dxe_out,
		PetscInt Nx, PetscInt Nz,
		PetscInt Npmlx0, PetscInt Npmlx1, PetscInt Npmlz0, PetscInt Npmlz1,
		PetscReal dx, PetscReal dz, 
		PetscReal omega)
{

  PetscInt comm_size;
  MPI_Comm_size(comm,&comm_size);
  
  PetscInt nrows = Nx*Nz;
  PetscInt ncols = Nx*Nz;
  
  //PetscPrintf(comm,"Creating the matrix Dxe (which operates on Ey).\n");

  Mat Dxe;

  MatCreate(comm,&Dxe);
  MatSetType(Dxe,MATRIX_TYPE);
  MatSetSizes(Dxe,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  if(comm_size==1)
    MatSeqAIJSetPreallocation(Dxe, 2, PETSC_NULL);
  else
    MatMPIAIJSetPreallocation(Dxe, 2, PETSC_NULL, 2, PETSC_NULL);
  
  PetscInt ns,ne;
  MatGetOwnershipRange(Dxe, &ns, &ne);

  for(PetscInt i=ns;i<ne;i++){
    
    PetscInt k;
    PetscInt ix=(k=i)%Nx;
    PetscInt iz=(k/=Nx)%Nz;

    //dEy/dx
    PetscScalar sx =pml_s(Nx, Npmlx0,Npmlx1, ix+0.5, dx, omega);
    PetscInt jx0=(ix<Nx-1) ? ix+1 : 0;
    PetscInt jz0=iz;
    PetscInt jx1=ix;
    PetscInt jz1=iz;
    
    PetscScalar val0 = +1.0/(sx*dx);
    PetscScalar val1 = -1.0/(sx*dx);
      
    PetscInt j0=jx0 + Nx*jz0;
    PetscInt j1=jx1 + Nx*jz1;
    PetscInt mcols=2;
    PetscInt jcols[2]={j0,j1};
    PetscScalar vals[2]={val0,val1};
      
    MatSetValues(Dxe, 1, &i, mcols, jcols, vals, ADD_VALUES);

  }
    
  MatAssemblyBegin(Dxe, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Dxe, MAT_FINAL_ASSEMBLY);
  
  *Dxe_out = Dxe;

}
  
#undef __FUNCT__
#define __FUNCT__ "create_Dzh"
void create_Dzh(MPI_Comm comm, Mat *Dzh_out,
		PetscInt Nx, PetscInt Nz,
		PetscInt Npmlx0, PetscInt Npmlx1, PetscInt Npmlz0, PetscInt Npmlz1,
		PetscReal dx, PetscReal dz,
		PetscReal omega)
{

  PetscInt comm_size;
  MPI_Comm_size(comm,&comm_size);
  
  PetscInt nrows = Nx*Nz;
  PetscInt ncols = Nx*Nz;

  //PetscPrintf(comm,"Creating the matrix d/dz (which operates on Hx).\n");

  Mat Dzh;

  MatCreate(comm,&Dzh);
  MatSetType(Dzh,MATRIX_TYPE);
  MatSetSizes(Dzh,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  if(comm_size==1)
    MatSeqAIJSetPreallocation(Dzh, 2, PETSC_NULL);
  else
    MatMPIAIJSetPreallocation(Dzh, 2, PETSC_NULL, 2, PETSC_NULL);
  
  PetscInt ns,ne;
  MatGetOwnershipRange(Dzh, &ns, &ne);

  for(PetscInt i=ns;i<ne;i++){

    PetscInt k;
    PetscInt ix=(k=i)%Nx;
    PetscInt iz=(k/=Nx)%Nz;

    //dHx/dz
    PetscScalar sz=pml_s(Nz, Npmlz0,Npmlz1, iz+0.0, dz, omega);
    PetscInt jx0=ix;
    PetscInt jz0=iz;
    PetscInt jx1=ix;
    PetscInt jz1=(iz>0) ? iz-1 : Nz-1;

    PetscScalar val0 = +1.0/(sz*dz);
    PetscScalar val1 = -1.0/(sz*dz);

    PetscInt j0=jx0 + Nx*jz0;
    PetscInt j1=jx1 + Nx*jz1;

    PetscInt mcols=2;
    PetscInt jcols[2]={j0,j1};
    PetscScalar vals[2]={val0,val1};

    MatSetValues(Dzh, 1, &i, mcols, jcols, vals, ADD_VALUES);

  }

  MatAssemblyBegin(Dzh, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Dzh, MAT_FINAL_ASSEMBLY);

  *Dzh_out = Dzh;

}

//**********************************************
#undef __FUNCT__
#define __FUNCT__ "create_Dxh"
void create_Dxh(MPI_Comm comm, Mat *Dxh_out,
		PetscInt Nx, PetscInt Nz,
		PetscInt Npmlx0, PetscInt Npmlx1, PetscInt Npmlz0, PetscInt Npmlz1,
		PetscReal dx, PetscReal dz,
		PetscReal omega)
{

  PetscInt comm_size;
  MPI_Comm_size(comm,&comm_size);
  
  PetscInt nrows = Nx*Nz;
  PetscInt ncols = Nx*Nz;

  //PetscPrintf(comm,"Creating the matrix -d/dx (which operates on Hz).\n");

  Mat Dxh;

  MatCreate(comm,&Dxh);
  MatSetType(Dxh,MATRIX_TYPE);
  MatSetSizes(Dxh,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  if(comm_size==1)
    MatSeqAIJSetPreallocation(Dxh, 2, PETSC_NULL);
  else
    MatMPIAIJSetPreallocation(Dxh, 2, PETSC_NULL, 2, PETSC_NULL);
  
  PetscInt ns,ne;
  MatGetOwnershipRange(Dxh, &ns, &ne);

  for(PetscInt i=ns;i<ne;i++){

    PetscInt k;
    PetscInt ix=(k=i)%Nx;
    PetscInt iz=(k/=Nx)%Nz;

    //-dHz/dx
    
    PetscScalar sx=pml_s(Nx, Npmlx0,Npmlx1, ix+0.0, dx, omega);
    PetscInt jx0=ix;
    PetscInt jz0=iz;
    PetscInt jx1=(ix>0) ? ix-1 : Nx-1;
    PetscInt jz1=iz;

    PetscScalar val0 = -1.0/(sx*dx);
    PetscScalar val1 = +1.0/(sx*dx);

    PetscInt j0=jx0 + Nx*jz0;
    PetscInt j1=jx1 + Nx*jz1;

    PetscInt mcols=2;
    PetscInt jcols[2]={j0,j1};
    PetscScalar vals[2]={val0,val1};

    MatSetValues(Dxh, 1, &i, mcols, jcols, vals, ADD_VALUES);

  }

  MatAssemblyBegin(Dxh, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Dxh, MAT_FINAL_ASSEMBLY);

  *Dxh_out = Dxh;

}


#undef __FUNCT__
#define __FUNCT__ "create_DDe"
void create_DDe(MPI_Comm comm, Mat *DDe_out,
	        PetscInt Nx, PetscInt Nz,
	        PetscInt Npmlx0, PetscInt Npmlx1, PetscInt Npmlz0, PetscInt Npmlz1,
	        PetscReal dx, PetscReal dz,
	        PetscReal omega)
{

  //PetscPrintf(comm,"Creating the matrix Dh.De (the double curl operator that acts on E fields). NOTE: mu = 1 is assumed.\n");  

  Mat Dze,Dzh,Dxe,Dxh;
  create_Dze(comm, &Dze, Nx,Nz, Npmlx0,Npmlx1,Npmlz0,Npmlz1, dx,dz, omega);
  create_Dzh(comm, &Dzh, Nx,Nz, Npmlx0,Npmlx1,Npmlz0,Npmlz1, dx,dz, omega);
  create_Dxe(comm, &Dxe, Nx,Nz, Npmlx0,Npmlx1,Npmlz0,Npmlz1, dx,dz, omega);
  create_Dxh(comm, &Dxh, Nx,Nz, Npmlx0,Npmlx1,Npmlz0,Npmlz1, dx,dz, omega);

  Mat D2z,D2x;
  MatMatMult(Dzh,Dze, MAT_INITIAL_MATRIX, 5.0/(2.0+2.0), &D2z);
  MatMatMult(Dxh,Dxe, MAT_INITIAL_MATRIX, 5.0/(2.0+2.0), &D2x);

  MatAXPY(D2z,1.0,D2x,DIFFERENT_NONZERO_PATTERN);
  
  *DDe_out = D2z;

  MatDestroy(&Dze);
  MatDestroy(&Dzh);
  MatDestroy(&Dxe);
  MatDestroy(&Dxh);
  MatDestroy(&D2x);
  

}

//create a matrix that syncs the Hx to the position of Ey[i,j] 
//Note that the indexing chosen as ix,iz in the order of the fastest to slowest

#undef __FUNCT__
#define __FUNCT__ "syncHx"
void syncHx(MPI_Comm comm, Mat *Ah_out, PetscInt Nx, PetscInt Nz)
{

  PetscInt comm_size;
  MPI_Comm_size(comm,&comm_size);
  
  PetscInt nrows = Nx*Nz;
  PetscInt ncols = Nx*Nz;

  //PetscPrintf(comm,"Creating the matrix syncHx to sync the Hx fields to the position of Ey.\n");

  Mat Ah;

  MatCreate(comm,&Ah);
  MatSetType(Ah,MATRIX_TYPE);
  MatSetSizes(Ah,PETSC_DECIDE,PETSC_DECIDE, nrows,ncols);
  if(comm_size==1)
    MatSeqAIJSetPreallocation(Ah, 2, PETSC_NULL);
  else
    MatMPIAIJSetPreallocation(Ah, 2, PETSC_NULL, 2, PETSC_NULL);
  
  PetscInt ns,ne;
  MatGetOwnershipRange(Ah, &ns, &ne);

  for(PetscInt i=ns;i<ne;i++){

    PetscInt k;
    PetscInt ix=(k=i)%Nx;
    PetscInt iz=(k/=Nx)%Nz;

    PetscInt jx0=ix; 
    PetscInt jz0=iz;
    PetscInt jx1=ix;
    PetscInt jz1=(iz>0) ? iz-1 : Nz-1;
    PetscScalar val0=0.5, val1=0.5;

    PetscInt j0=jx0 + Nx*jz0;
    PetscInt j1=jx1 + Nx*jz1;
    PetscInt mcols=2;
    PetscInt jcols[2]={j0,j1};
    PetscScalar vals[2]={val0,val1};
    
    MatSetValues(Ah, 1, &i, mcols, jcols, vals, ADD_VALUES);

  }

  MatAssemblyBegin(Ah, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Ah, MAT_FINAL_ASSEMBLY);

  *Ah_out = Ah;

}


