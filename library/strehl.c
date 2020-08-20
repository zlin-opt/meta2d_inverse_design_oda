#include "strehl.h"

extern PetscInt count;

void strehlobj(unsigned nspecs, PetscReal *result,
	       unsigned mdof, const PetscReal *dof,
	       PetscReal *grad,
	       void *data)
{

  params_ *ptdata = (params_*)data;

  PetscInt colour = ptdata->colour;
  MPI_Comm subcomm = ptdata->subcomm;
  PetscInt nsims_per_comm = ptdata->nsims_per_comm;
  
  PetscInt nfreqs = ptdata->nfreqs;
  PetscInt nangles = ptdata->nangles;

  PetscInt nx = ptdata->nx;
  PetscInt ncells = ptdata->ncells;
  PetscInt px = ptdata->px;
  PetscInt Nz = ptdata->Nz;
  PetscInt nlayers_active = ptdata->nlayers_active;
  PetscInt *mz = ptdata->mz;

  Mat W = ptdata->W;
  Mat A = ptdata->A;

  PetscReal beta = ptdata->beta;
  PetscInt zfixed = ptdata->zfixed;

  Mat *DDe = ptdata->DDe;
  Vec *epsDiff = ptdata->epsDiff;
  Vec *epsBkg = ptdata->epsBkg;
  PetscReal *omega = ptdata->omega;

  PetscInt maxit = ptdata->maxit;
  PetscInt reuse_ksp = ptdata->reuse_ksp;

  PetscScalar **Jsrc = ptdata->Jsrc;
  PetscInt jz_src = ptdata->jz_src;

  Mat *Pdag = ptdata->Pdag;
  Vec *ge = ptdata->ge;
  Vec *gh = ptdata->gh;
  PetscReal *airyfactor = ptdata->airyfactor;

  PetscPrintf(PETSC_COMM_WORLD,"Computing the Strehl ratios.\n");

  PetscInt ndof=mdof-1;
  PetscInt nx_odm = nx+2*px;
  PetscInt num_hdof = nx_odm*nlayers_active*ncells;
  PetscScalar *_hdof = (PetscScalar *)malloc(ndof*sizeof(PetscScalar));
  PetscScalar *hdof = (PetscScalar *)malloc(num_hdof*sizeof(PetscScalar));
  for(PetscInt i=0;i<ndof;i++)
    _hdof[i]=dof[i]+PETSC_i*0.0;
  matmult_arrays(W, _hdof,SCAL, hdof,SCAL, 0);

  PetscInt nsims = ncells*nfreqs*nangles;
  PetscScalar fEy_sims[nsims],fHx_sims[nsims],nPz_sims[nsims];
  PetscScalar *fEygrad_sims[nsims],*fHxgrad_sims[nsims],*nPzgrad_sims[nsims];
  for(PetscInt isim=0;isim<nsims;isim++){
    fEy_sims[isim]=0, fHx_sims[isim]=0, nPz_sims[isim]=0;
    fEygrad_sims[isim]=(PetscScalar *)malloc(num_hdof*sizeof(PetscScalar));
    fHxgrad_sims[isim]=(PetscScalar *)malloc(num_hdof*sizeof(PetscScalar));
    nPzgrad_sims[isim]=(PetscScalar *)malloc(num_hdof*sizeof(PetscScalar));
    for(PetscInt i=0;i<num_hdof;i++)
      fEygrad_sims[isim][i]=0, fHxgrad_sims[isim][i]=0, nPzgrad_sims[isim][i]=0;
  }
  
  for(PetscInt isim_per_comm=0;isim_per_comm<nsims_per_comm;isim_per_comm++){

    PetscInt isim=isim_per_comm + nsims_per_comm * colour;

    PetscInt itmp;
    PetscInt icell = (itmp=isim)%ncells;
    
    PetscInt num_edof = nx_odm*integer_sum(mz,0,nlayers_active);
    PetscInt num_hgrad = nx_odm*nlayers_active;
    PetscScalar *edof = (PetscScalar *)malloc(num_edof*sizeof(PetscScalar));
    PetscScalar *fEy_egrad = (PetscScalar *)malloc(num_edof*sizeof(PetscScalar));
    PetscScalar *fHx_egrad = (PetscScalar *)malloc(num_edof*sizeof(PetscScalar));
    PetscScalar *nPz_egrad = (PetscScalar *)malloc(num_edof*sizeof(PetscScalar));
    PetscScalar *fEy_hgrad = (PetscScalar *)malloc(num_hgrad*sizeof(PetscScalar));
    PetscScalar *fHx_hgrad = (PetscScalar *)malloc(num_hgrad*sizeof(PetscScalar));
    PetscScalar *nPz_hgrad = (PetscScalar *)malloc(num_hgrad*sizeof(PetscScalar));
    PetscInt cell_start=icell*nx_odm*nlayers_active;
    KSP ksp;
    PetscInt *its,tmpits=100;
    if(reuse_ksp==1){
      ksp=ptdata->ksp[isim_per_comm];
      its=&(ptdata->its[isim_per_comm]);
    }else{
      setupKSPDirect(subcomm, &ksp, maxit);
      its=&tmpits;
    }
    
    varh_expand(&(hdof[cell_start]), edof, nx_odm, 1, nlayers_active, mz, beta, zfixed);
    PetscScalar fEy,fHx,nPz;
    strehlpieces(subcomm, nx_odm,Nz, edof, A,
		 DDe[isim_per_comm], epsDiff[isim_per_comm], epsBkg[isim_per_comm], omega[isim_per_comm],
		 ksp, its, maxit,
		 Jsrc[isim_per_comm], jz_src,
		 ge[isim_per_comm], gh[isim_per_comm], Pdag[isim_per_comm],
		 &fEy, &fHx, &nPz,
		 fEy_egrad, fHx_egrad, nPz_egrad);
    varh_contract(fEy_egrad, fEy_hgrad, &(hdof[cell_start]), nx_odm,1,nlayers_active, mz, beta, zfixed);
    varh_contract(fHx_egrad, fHx_hgrad, &(hdof[cell_start]), nx_odm,1,nlayers_active, mz, beta, zfixed);
    varh_contract(nPz_egrad, nPz_hgrad, &(hdof[cell_start]), nx_odm,1,nlayers_active, mz, beta, zfixed);

    PetscInt rank;
    MPI_Comm_rank(subcomm,&rank);
    if(rank==0){
      fEy_sims[isim]=fEy;
      fHx_sims[isim]=fHx;
      nPz_sims[isim]=nPz;
      for(PetscInt i=0;i<num_hgrad;i++){
	fEygrad_sims[isim][cell_start+i]=fEy_hgrad[i];
	fHxgrad_sims[isim][cell_start+i]=fHx_hgrad[i];
	nPzgrad_sims[isim][cell_start+i]=nPz_hgrad[i];
      }
    }
    MPI_Barrier(PETSC_COMM_WORLD);

    free(edof);
    free(fEy_egrad);
    free(fHx_egrad);
    free(nPz_egrad);
    free(fEy_hgrad);
    free(fHx_hgrad);
    free(nPz_hgrad);
    if(reuse_ksp==0)
      KSPDestroy(&ksp);
    
  }

  //gather
  PetscScalar *fEy_all=(PetscScalar *)malloc(nsims*sizeof(PetscScalar));
  PetscScalar *fHx_all=(PetscScalar *)malloc(nsims*sizeof(PetscScalar));
  PetscScalar *nPz_all=(PetscScalar *)malloc(nsims*sizeof(PetscScalar));
  PetscScalar **fEygrad_all=(PetscScalar **)malloc(nsims*sizeof(PetscScalar*));
  PetscScalar **fHxgrad_all=(PetscScalar **)malloc(nsims*sizeof(PetscScalar*));
  PetscScalar **nPzgrad_all=(PetscScalar **)malloc(nsims*sizeof(PetscScalar*));
  for(PetscInt isim=0;isim<nsims;isim++){
    fEygrad_all[isim]=(PetscScalar *)malloc(num_hdof*sizeof(PetscScalar));
    fHxgrad_all[isim]=(PetscScalar *)malloc(num_hdof*sizeof(PetscScalar));
    nPzgrad_all[isim]=(PetscScalar *)malloc(num_hdof*sizeof(PetscScalar));
  }
  MPI_Allreduce(fEy_sims,fEy_all,nsims,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(fHx_sims,fHx_all,nsims,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  MPI_Allreduce(nPz_sims,nPz_all,nsims,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  for(PetscInt isim=0;isim<nsims;isim++){
    MPI_Allreduce(fEygrad_sims[isim],fEygrad_all[isim],num_hdof,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
    MPI_Allreduce(fHxgrad_sims[isim],fHxgrad_all[isim],num_hdof,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
    MPI_Allreduce(nPzgrad_sims[isim],nPzgrad_all[isim],num_hdof,MPIU_SCALAR,MPI_SUM,PETSC_COMM_WORLD);
  }
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"DEBUG Successfully gathered!\n");

  //Consolidate
  PetscScalar fEy_specs[nspecs], fHx_specs[nspecs], nPz_specs[nspecs];
  PetscScalar *fEygrad_specs[nspecs],*fHxgrad_specs[nspecs],*nPzgrad_specs[nspecs];
  for(PetscInt ispec=0;ispec<nspecs;ispec++){
    fEy_specs[ispec]=0, fHx_specs[ispec]=0, nPz_specs[ispec]=0;
    fEygrad_specs[ispec]=(PetscScalar *)malloc(num_hdof*sizeof(PetscScalar));
    fHxgrad_specs[ispec]=(PetscScalar *)malloc(num_hdof*sizeof(PetscScalar));
    nPzgrad_specs[ispec]=(PetscScalar *)malloc(num_hdof*sizeof(PetscScalar));
    for(PetscInt i=0;i<num_hdof;i++){
      fEygrad_specs[ispec][i]=0;
      fHxgrad_specs[ispec][i]=0;
      nPzgrad_specs[ispec][i]=0;
    }
  }
  for(PetscInt iangle=0;iangle<nangles;iangle++){
    for(PetscInt ifreq=0;ifreq<nfreqs;ifreq++){
      for(PetscInt icell=0;icell<ncells;icell++){

	PetscInt isim = icell + ncells*ifreq + ncells*nfreqs*iangle;
	PetscInt ispec = ifreq + nfreqs*iangle;

	fEy_specs[ispec] += fEy_all[isim];
	fHx_specs[ispec] += fHx_all[isim];
	nPz_specs[ispec] += nPz_all[isim];
	PetscInt block_size = nx_odm * nlayers_active;
	PetscInt cell_start = icell*nx_odm*nlayers_active;
	for(PetscInt i=0;i<block_size;i++){
	  fEygrad_specs[ispec][cell_start+i] += fEygrad_all[isim][cell_start+i];
	  fHxgrad_specs[ispec][cell_start+i] += fHxgrad_all[isim][cell_start+i];
	  nPzgrad_specs[ispec][cell_start+i] += nPzgrad_all[isim][cell_start+i];
	}
      }
    }
  }
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"DEBUG Successfully consolidated!\n");

  //back_prop through odm+filters
  PetscScalar *fEy_grad[nspecs],*fHx_grad[nspecs],*nPz_grad[nspecs];
  for(PetscInt ispec=0;ispec<nspecs;ispec++){
    fEy_grad[ispec]=(PetscScalar *)malloc(ndof*sizeof(PetscScalar));
    fHx_grad[ispec]=(PetscScalar *)malloc(ndof*sizeof(PetscScalar));
    nPz_grad[ispec]=(PetscScalar *)malloc(ndof*sizeof(PetscScalar));
    matmult_arrays(W, fEygrad_specs[ispec],SCAL, fEy_grad[ispec],SCAL, 1);
    matmult_arrays(W, fHxgrad_specs[ispec],SCAL, fHx_grad[ispec],SCAL, 1);
    matmult_arrays(W, nPzgrad_specs[ispec],SCAL, nPz_grad[ispec],SCAL, 1);
  }
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD,"DEBUG Successfully back-propped!\n");    

  //populate the real-valued result and grad;
  PetscReal epi_t = dof[ndof];
  for(PetscInt ispec=0;ispec<nspecs;ispec++){

    PetscReal fSz = creal( - conj(fHx_specs[ispec]) * fEy_specs[ispec] );
    PetscReal Pz = creal(nPz_specs[ispec]);
    PetscReal strehl = airyfactor[ispec] * (fSz/Pz);
    PetscPrintf(PETSC_COMM_WORLD,"Strehlobj for ispec %d at step %d is %0.16g \n",ispec,count,strehl);
    
    result[ispec] = epi_t - strehl;
    
    for(PetscInt i=0;i<ndof;i++){
      PetscReal fSz_grad = creal( - conj( fHx_grad[ispec][i] ) * fEy_specs[ispec] - conj(fHx_specs[ispec]) * fEy_grad[ispec][i] );
      PetscReal Pz_grad = creal( nPz_grad[ispec][i] );
      grad[i + mdof*ispec] = - airyfactor[ispec] * ( fSz_grad/Pz - (fSz/pow(Pz,2)) * Pz_grad );
    }
    grad[ndof + mdof*ispec] = 1.0;

  }

  free(fEy_all);
  free(fHx_all);
  free(nPz_all);
  for(PetscInt isim=0;isim<nsims;isim++){
    free(fEygrad_sims[isim]);
    free(fHxgrad_sims[isim]);
    free(nPzgrad_sims[isim]);
    free(fEygrad_all[isim]);
    free(fHxgrad_all[isim]);
    free(nPzgrad_all[isim]);
  }
  free(fEygrad_all);
  free(fHxgrad_all);
  free(nPzgrad_all);
  for(PetscInt ispec=0;ispec<nspecs;ispec++){
    free(fEygrad_specs[ispec]);
    free(fHxgrad_specs[ispec]);
    free(nPzgrad_specs[ispec]);
    free(fEy_grad[ispec]);
    free(fHx_grad[ispec]);
    free(nPz_grad[ispec]);
  }
  free(_hdof);
  free(hdof);
  PetscPrintf(PETSC_COMM_WORLD,"DEBUG Successful!\n");

}
