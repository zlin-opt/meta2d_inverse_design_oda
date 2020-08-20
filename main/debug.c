#include "petsc.h"
#include "petscsys.h"
#include "nlopt.h"
#include <assert.h>
#include "lib2d.h"

PetscInt count=0;
PetscInt mma_verbose;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{

  MPI_Init(NULL, NULL);
  PetscInitialize(&argc,&argv,NULL,NULL);

  PetscInt nget;

  //Get the specifications: frequencies, angles and polarizations
  PetscInt nfreqs,nangles;
  getint("-nfreqs",&nfreqs,3);
  getint("-nangles",&nangles,3);
  PetscReal freqs[nfreqs], angles[nangles];
  nget=nfreqs;
  getrealarray("-freqs",freqs,&nget,1.0);
  nget=nangles;
  getrealarray("-angles",angles,&nget,0.0);
  PetscInt nspecs = nfreqs*nangles;
  
  //Get nx, px, pmlx, dx, ncells;
  PetscInt nx, px, pmlx, ncells;
  getint("-nx",&nx,2000);
  getint("-px",&px,50);
  getint("-pmlx",&pmlx,25);
  getint("-ncells",&ncells,5);
  PetscReal dx;
  getreal("-dx",&dx,0.05);

  PetscInt Nx=nx*ncells;
  
  //Get vertical layers information
  PetscInt nlayers_total,nlayers_active;
  getint("-nlayers_total", &nlayers_total, 3);
  getint("-nlayers_active",&nlayers_active,1);
  PetscInt thickness[nlayers_total];
  PetscInt id_active_layers[nlayers_active];
  nget=nlayers_total;
  getintarray("-thickness",thickness,&nget,10);
  nget=nlayers_active;
  getintarray("-id_active_layers",id_active_layers,&nget,0);
  PetscInt Nz=integer_sum(thickness,0,nlayers_total);
  PetscPrintf(PETSC_COMM_WORLD,"NOTE: Nz = %d \n",Nz);
  PetscReal dz;
  PetscInt pmlz;
  getreal("-dz",&dz,dx);
  getint("-pmlz",&pmlz,pmlx);
  //Construct mzo and izo (starting z-coordinates for the active/all layers)
  PetscInt mz[nlayers_active],mzo[nlayers_active];
  for(PetscInt i=0;i<nlayers_active;i++){
    mz[i]=thickness[id_active_layers[i]];
    mzo[i]=integer_sum(thickness,0,id_active_layers[i]);
  }
  PetscInt izo[nlayers_total];
  for(PetscInt i=0;i<nlayers_total;i++)
    izo[i]=integer_sum(thickness,0,i);
  
  //Get the epsilon information
  PetscReal epsbkg[nfreqs][2*nlayers_total];
  PetscReal epsfeg[nfreqs][2*nlayers_active];
  PetscScalar bkg_eps[nfreqs][nlayers_total];
  PetscScalar feg_eps[nfreqs][nlayers_active];
  PetscScalar diff_eps[nfreqs][nlayers_active];
  for(PetscInt ifreq=0;ifreq<nfreqs;ifreq++){

    char flag[PETSC_MAX_PATH_LEN];
    sprintf(flag,"-ifreq%d_epsbkg",ifreq);
    nget=2*nlayers_total;
    getrealarray(flag,epsbkg[ifreq],&nget,1);
    sprintf(flag,"-ifreq%d_epsfeg",ifreq);
    nget=2*nlayers_active;
    getrealarray(flag,epsfeg[ifreq],&nget,1);

    for(PetscInt j=0;j<nlayers_total;j++)
      bkg_eps[ifreq] [j]=epsbkg[ifreq][j]+PETSC_i*epsbkg[ifreq][j+nlayers_total];
    for(PetscInt j=0;j<nlayers_active;j++){
      feg_eps[ifreq] [j]=epsfeg[ifreq][j]+PETSC_i*epsfeg[ifreq][j+nlayers_active];
      diff_eps[ifreq][j]=feg_eps[ifreq][j]-bkg_eps[ifreq][id_active_layers[j]];
    }
    
  }
  
  //Get filter info, focal length, print_at and current amplitude; also angle / spatial compress factors for imaging
  PetscReal filter_radius,filter_sigma,filter_beta;
  getreal("-filter_radius",&filter_radius,0.6);
  getreal("-filter_sigma",&filter_sigma,10);
  getreal("-filter_beta",&filter_beta,0);
  PetscInt filter_choice;
  getint("-filter_choice",&filter_choice,0);
  PetscInt zfixed;
  getint("-zfixed",&zfixed,1);
  PetscInt pixbunch_radius=(PetscInt)round(filter_radius);
  if(filter_choice==0)
    PetscPrintf(PETSC_COMM_WORLD,"NOTE filter choice is %d. Density filter is used with radius %g. [less than 1 means identity].\n", filter_choice, filter_radius);
  else
    PetscPrintf(PETSC_COMM_WORLD,"NOTE filter choice is %d. Pixel-bunching is used with bunch size %d [ must NOT be 0].\n",filter_choice, pixbunch_radius);
  
  PetscReal foclen;
  getreal("-focal_length",&foclen,100);
  PetscReal virtual_foclen;
  getreal("-virtual_focal_length",&virtual_foclen,foclen);
  PetscInt print_at;
  getint("-print_at",&print_at,1);
  PetscReal image_compress_factor;
  getreal("-image_compress_factor",&image_compress_factor,1.0);
  PetscInt jz_src,jz_mtr;
  getint("-jz_src",&jz_src,pmlz+5);
  getint("-jz_mtr",&jz_mtr,Nz-pmlz-5);
  PetscReal maxstrehl[nspecs];
  for(PetscInt ispec=0;ispec<nspecs;ispec++){

    char flag[PETSC_MAX_PATH_LEN];
    sprintf(flag,"-ispec%d_maxstrehl",ispec);
    getreal(flag,&(maxstrehl[ispec]),1.0);

  }
  PetscInt reuse_ksp;
  getint("-reuse_ksp",&reuse_ksp,1);
  
  //Choose the correct number of ndof based on the filter choice and build the filters
  PetscInt ndof0 = Nx*nlayers_active/2;
  PetscInt ndof1 = Nx*nlayers_active/(2*pixbunch_radius);
  PetscInt ndof = (filter_choice==0) ? ndof0 : ndof1;
  PetscReal *dof = (PetscReal *)malloc(ndof*sizeof(PetscReal));
  PetscBool flg;
  char strin[PETSC_MAX_PATH_LEN];
  PetscOptionsGetString(PETSC_NULL,"-initial_filename",strin,PETSC_MAX_PATH_LEN-1,&flg);
  if(flg){
    PetscPrintf(PETSC_COMM_WORLD,"--initial_filename is %s \n",strin);
    readfromfile(strin, dof,REAL, ndof);
  }else{
    PetscReal autoinit=0.5;
    PetscPrintf(PETSC_COMM_WORLD,"***NOTE: NO initial filename given. Auto-initialized to %g\n",autoinit);
    for(PetscInt i=0;i<ndof;i++)
      dof[i] = autoinit;
  }

  Mat Filt,Mirr,Q;
  if(filter_choice==0)
    density_filter(PETSC_COMM_WORLD, &Filt, Nx/2,nlayers_active, filter_radius,filter_sigma, 1);
  else
    pixbunch(PETSC_COMM_WORLD, &Filt, Nx/2,nlayers_active, pixbunch_radius);
  mirrormat(PETSC_COMM_WORLD, &Mirr, Nx/2,nlayers_active);
  MatMatMult(Mirr,Filt,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Q);
  MatDestroy(&Filt);
  MatDestroy(&Mirr);
  Mat OvM,W;
  PetscInt mrows_per_cell[ncells], cell_start[ncells];
  create_ovmat(PETSC_COMM_WORLD, &OvM, mrows_per_cell,cell_start, nx,px,ncells,nlayers_active, pmlx,pmlx, mz, 1);
  MatMatMult(OvM,Q,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&W);
  MatDestroy(&OvM);
  PetscInt Wrows,Wcols;
  MatGetSize(W,&Wrows,&Wcols);
  PetscPrintf(PETSC_COMM_WORLD,"Dimensions of the \"filter+OVDM\" matrix: %d x %d \n",Wrows,Wcols);
  PetscPrintf(PETSC_COMM_WORLD,"NOTE Wrows = (nx+2px) * ncells * nlayers ( %d = %d x %d x %d ) \n",(nx+2*px)*ncells*nlayers_active,nx+2*px,ncells,nlayers_active);
  PetscPrintf(PETSC_COMM_WORLD,"NOTE Wcols = nx * ncells * nlayers / (2 * [pixbunch_radius]) = ndof ( %d = %d x %d x %d / (2 x [%d]) = %d ) \n",
	      nx*ncells*nlayers_active/(2*((filter_choice==0)?1:pixbunch_radius)),
	      nx,ncells,nlayers_active,    (filter_choice==0)?1:pixbunch_radius,
	      ndof);

  
  //Fastest to slowest indexes isim -> icell, ifreq, iangle
  //Fastest to slowest indexes ispec -> ifreq, iangle
  PetscInt nsims = ncells*nfreqs*nangles;

  //Handle the MPI splitting
  PetscInt nprocs_total;
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs_total);
  PetscInt nprocs_per_sim;
  PetscInt nsims_per_comm;
  PetscInt ncomms;
  PetscInt nprocs_error=-10;
  for(nsims_per_comm=1;nsims_per_comm<=nsims;nsims_per_comm++){
    for(nprocs_per_sim=1;nprocs_per_sim<=nprocs_total;nprocs_per_sim++){
      nprocs_error = nprocs_per_sim * nsims - nsims_per_comm * nprocs_total;
      ncomms = nsims/nsims_per_comm;
      if(nprocs_error == 0)
	break;
    }
    if(nprocs_error == 0)
      break;
  }
  if(nprocs_error != 0)
    SETERRQ(PETSC_COMM_WORLD,1,"The total number of processors is not consistent with the given number of simulations.");
  else
    PetscPrintf(PETSC_COMM_WORLD,"***NOTE: nprocs_per_sim ( %d ) x nsims ( %d ) = nsims_per_comm ( %d ) x nprocs_total ( %d ), ncomms= %d \n",
		nprocs_per_sim, nsims,
		nsims_per_comm, nprocs_total,
		ncomms);
  PetscInt nprocs_per_comm=nprocs_per_sim;

  PetscInt rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm subcomm;
  PetscInt colour = rank/nprocs_per_comm;
  MPI_Comm_split(MPI_COMM_WORLD, colour, rank, &subcomm);

  if(rank==0)
    mma_verbose=1;
  
  params_ params;
  PetscInt nx_odm=nx+2*px;
  //Build arrays, vectors and matrices
  Mat A;
  create_Ainterp(subcomm, &A,
		 nx_odm, Nz,
		 0, nx_odm, mzo, mz, 0,
		 nlayers_active);
  Vec tmpVec;
  MatCreateVecs(A,PETSC_NULL,&tmpVec);

  Mat DDe[nsims_per_comm];
  Vec epsDiff[nsims_per_comm];
  Vec epsBkg[nsims_per_comm];
  KSP ksp[nsims_per_comm];
  Mat Pdag[nsims_per_comm];
  Vec ge[nsims_per_comm];
  Vec gh[nsims_per_comm];
  params.omega=(PetscReal *)malloc(nsims_per_comm*sizeof(PetscReal));
  params.Jsrc=(PetscScalar **)malloc(nsims_per_comm*sizeof(PetscScalar *));
  params.its=(PetscInt *)malloc(nsims_per_comm*sizeof(PetscInt));
  params.airyfactor=(PetscReal *)malloc(nspecs*sizeof(PetscReal));
  for(PetscInt isim_per_comm=0;isim_per_comm<nsims_per_comm;isim_per_comm++){

    PetscInt isim = isim_per_comm + nsims_per_comm * colour;
    PetscInt itmp;
    PetscInt icell = (itmp=isim)%ncells;
    PetscInt ifreq = (itmp/=ncells)%nfreqs;
    PetscInt iangle = (itmp/=nfreqs)%nangles;
    
    PetscReal omega = 2*M_PI*freqs[ifreq];
    create_DDe(subcomm, &(DDe[isim_per_comm]),
	       nx_odm, Nz,
	       pmlx, pmlx, pmlz, pmlz,
	       dx, dz,
	       omega);
    params.omega[isim_per_comm]=omega;

    VecDuplicate(tmpVec,&(epsBkg[isim_per_comm]));
    VecDuplicate(tmpVec,&(epsDiff[isim_per_comm]));
    setlayer_eps( epsBkg[isim_per_comm], nx_odm,Nz, nlayers_total, izo, thickness,  bkg_eps[ifreq]);
    setlayer_eps(epsDiff[isim_per_comm], nx_odm,Nz,nlayers_active, mzo,        mz, diff_eps[ifreq]);

    PetscReal nsub = sqrt(creal(bkg_eps[ifreq][0]));
    PetscReal kwav = nsub*omega;
    PetscReal alpha = angles[iangle] * M_PI/180;
    params.Jsrc[isim_per_comm]=(PetscScalar *)malloc(nx_odm*sizeof(PetscScalar));
    srcJy(params.Jsrc[isim_per_comm], icell*nx-px,nx_odm,dx,dz, kwav,alpha,1);

    params.maxit=15;
    setupKSPDirect(subcomm, &ksp[isim_per_comm], params.maxit);
    params.its[isim_per_comm]=100;

    create_P(subcomm, &Pdag[isim_per_comm], 1,
	     nx_odm, Nz,
	     pmlx,pmlx,pmlz,pmlz,
	     dx,dz,
	     omega,
	     px,nx,jz_mtr);

    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);

    PetscPrintf(subcomm,"Before creating q forms %d \n",isim);
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);
   
    PetscReal nsub0=sqrt(creal(bkg_eps[0][0]));
    PetscReal alpha2=asin(nsub0*sin(alpha));
    PetscReal fspotx=virtual_foclen*tan(alpha2) * image_compress_factor;
    PetscReal fspotz=foclen;
    PetscReal origx=Nx*dx/2.0;
    PetscReal origz=0;
    VecDuplicate(tmpVec,&ge[isim_per_comm]);
    VecDuplicate(tmpVec,&gh[isim_per_comm]);
    
    gforms(subcomm, ge[isim_per_comm], gh[isim_per_comm],
	   nx_odm,Nz,
	   pmlx,pmlx,pmlz,pmlz,
	   dx,dz,
	   omega,
	   px,nx,icell*nx,jz_mtr,
	   fspotx,fspotz,
	   origx,origz);

    PetscScalar tmp;
    VecSum(ge[isim_per_comm],&tmp);
    PetscPrintf(PETSC_COMM_WORLD,"debug params: sum_ge %g %g \n",creal(tmp),cimag(tmp));
    VecSum(gh[isim_per_comm],&tmp);
    PetscPrintf(PETSC_COMM_WORLD,"debug params: sum_gh %g %g \n",creal(tmp),cimag(tmp));
    
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);

    PetscPrintf(subcomm,"The coordinates (x,z) of the focal spot: (%g, %g) for ifreq %d and iangle %d in comm %d. NOTE the origin at the lens center.\n",
		fspotx, fspotz,
		ifreq,iangle,colour);
    
    MPI_Barrier(subcomm);
    MPI_Barrier(PETSC_COMM_WORLD);
    
  }  
  VecDestroy(&tmpVec);

  params.subcomm=subcomm;
  params.colour=colour;
  params.nsims_per_comm=nsims_per_comm;
  
  params.nfreqs=nfreqs;
  params.nangles=nangles;
  
  params.nx=nx;
  params.ncells=ncells;
  params.px=px;
  params.nlayers_active=nlayers_active;
  params.Nz=Nz;
  params.mz=mz;
  
  params.W=W;
  params.A=A;

  params.beta=filter_beta;
  params.zfixed=zfixed;

  params.DDe=DDe;
  params.epsDiff=epsDiff;
  params.epsBkg=epsBkg;

  params.ksp=ksp;
  params.reuse_ksp=reuse_ksp;
  
  params.jz_src=jz_src;

  params.Pdag=Pdag;
  params.ge=ge;
  params.gh=gh;

  for(PetscInt iangle=0;iangle<nangles;iangle++){
    for(PetscInt ifreq=0;ifreq<nfreqs;ifreq++){

      PetscReal alpha = angles[iangle] * M_PI/180;
      PetscReal nsub0=sqrt(creal(bkg_eps[0][0]));
      PetscReal alpha2=asin(nsub0*sin(alpha));
      
      PetscInt ispec = ifreq + nfreqs*iangle;
      PetscReal NA = sin(atan(Nx*dx/(2*foclen)));
      PetscReal FWHM = (0.88/(2*freqs[ifreq]*NA)) * (1/cos(alpha2));
      
      params.airyfactor[ispec] = (FWHM/0.885893) * (1/maxstrehl[ispec]);    

    }
  }

  params.print_at=print_at;

  PetscInt Job;
  getint("-Job",&Job,1);
  
  if(Job==-1){

    PetscInt idof;
    getint("-idof",&idof,ndof/2);
    PetscReal ss[3];
    nget=3;
    getrealarray("-s0,s1,ds",ss,&nget,0);
    PetscReal s0=ss[0], s1=ss[1], ds=ss[2];

    PetscInt mdof=ndof+1;
    PetscReal *result=(PetscReal *)malloc(nspecs*sizeof(PetscReal));
    PetscReal *_dof=(PetscReal *)malloc(mdof*sizeof(PetscReal));
    PetscReal *grad=(PetscReal *)malloc(nspecs*mdof*sizeof(PetscReal));
    for(PetscInt i=0;i<ndof;i++)
      _dof[i]=dof[i];
    _dof[ndof]=0;

    for(PetscReal s=s0;s<s1;s+=ds){

      _dof[idof]=s;
      strehlobj((unsigned) nspecs, result,
		(unsigned) mdof, _dof,
		grad,
		&params);
      for(PetscInt ispec=0;ispec<nspecs;ispec++){
	PetscReal objval = result[ispec];
	PetscReal gradval = grad[idof + mdof*ispec];
	PetscPrintf(PETSC_COMM_WORLD,"objval%d: %g %0.16g %0.16g \n", ispec, s, objval, gradval);
      }
      
    }

    free(result);
    free(_dof);
    free(grad);
  }

  if(Job==1){

    PetscInt mdof = ndof+1;

    PetscInt algouter, alginner, algmaxeval;
    getint("-algouter",&algouter,24);
    getint("-alginner",&alginner,24);
    getint("-algmaxeval",&algmaxeval,500);

    PetscReal epi_t;
    getreal("-epi_t",&epi_t,0);

    PetscReal *lb=(PetscReal *)malloc(mdof*sizeof(PetscReal));
    PetscReal *ub=(PetscReal *)malloc(mdof*sizeof(PetscReal));
    PetscReal *_dof=(PetscReal *)malloc(mdof*sizeof(PetscReal));
    for(PetscInt i=0;i<ndof;i++){
      lb[i]=0.0;
      ub[i]=1.0;
      _dof[i]=dof[i];
    }
    lb[ndof]=0.0;
    ub[ndof]=1.0/0.0;
    _dof[ndof]=epi_t;
    MPI_Barrier(PETSC_COMM_WORLD);

    nlopt_opt opt;
    nlopt_opt local_opt;
    opt = nlopt_create((nlopt_algorithm)algouter, mdof);
    nlopt_set_lower_bounds(opt,lb);
    nlopt_set_upper_bounds(opt,ub);
    nlopt_set_maxeval(opt,algmaxeval);
    nlopt_set_maxtime(opt,100000000);

    if(alginner){
      local_opt=nlopt_create(alginner, mdof);
      nlopt_set_ftol_rel(local_opt, 1e-6);
      nlopt_set_maxeval(local_opt,10000);
      nlopt_set_local_optimizer(opt,local_opt);
    }

    PetscReal *mctol = (PetscReal *)malloc(nspecs*sizeof(PetscReal));
    for(PetscInt i=0;i<nspecs;i++)
      mctol[i]=1e-8;
    nlopt_add_inequality_mconstraint(opt, (unsigned) nspecs, (nlopt_mfunc) strehlobj, &params, mctol);

    nlopt_set_max_objective(opt,(nlopt_func)dummy_obj,&print_at);

    PetscReal maxf;
    nlopt_result result=nlopt_optimize(opt,_dof,&maxf);
    PetscPrintf(PETSC_COMM_WORLD,"IMPORTANT: nlopt_result: %d \n",result);

    nlopt_destroy(opt);
    if(alginner) nlopt_destroy(local_opt);

    free(lb);
    free(ub);
    free(_dof);
    free(mctol);

  }

  if(Job==0){

    PetscScalar tmp;
    
    PetscInt ispec;
    getint("-ispec",&ispec,0);
    PetscInt itmp;
    PetscInt ifreq=(itmp=ispec)%nfreqs;
    PetscInt iangle=(itmp/=nfreqs)%nangles;

    Mat FiltFull,MirrFull,Qfull;
    if(filter_choice==0)
      density_filter(PETSC_COMM_WORLD, &FiltFull, Nx/2,nlayers_active, filter_radius,filter_sigma, 1);
    else
      pixbunch(PETSC_COMM_WORLD, &FiltFull, Nx/2,nlayers_active, pixbunch_radius);
    mirrormat(PETSC_COMM_WORLD, &MirrFull, Nx/2,nlayers_active);
    MatMatMult(MirrFull,FiltFull,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Qfull);
    MatDestroy(&FiltFull);
    MatDestroy(&MirrFull);
    Mat OvMfull,Wfull;
    PetscInt mrowscell[ncells], cellstart[ncells];
    create_ovmat(PETSC_COMM_WORLD, &OvMfull, mrowscell,cellstart, Nx,px,1,nlayers_active, pmlx,pmlx, mz, 1);
    MatMatMult(OvMfull,Qfull,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Wfull);
    MatDestroy(&OvMfull);
    MatDestroy(&Qfull);
    
    PetscInt odmNx=Nx+2*px;
    
    PetscScalar *hdof=(PetscScalar *)malloc(odmNx*nlayers_active*sizeof(PetscScalar));
    matmult_arrays(Wfull, dof,REAL, hdof,SCAL, 0);

    PetscInt mzsum = integer_sum(mz,0,nlayers_active);
    PetscScalar *edof=(PetscScalar *)malloc(odmNx*mzsum*sizeof(PetscScalar));
    varh_expand(hdof, edof, odmNx, 1, nlayers_active, mz, filter_beta, zfixed);

    Mat Afull;
    create_Ainterp(PETSC_COMM_WORLD, &Afull,
		   odmNx, Nz,
		   0, odmNx, mzo, mz, 0,
		   nlayers_active);
    Vec _edof,eps;
    MatCreateVecs(Afull,&_edof,&eps);
    array2mpi(edof,SCAL, _edof);
    MatMult(Afull,_edof,eps);

    Vec fullepsBkg,fullepsDiff;
    VecDuplicate(eps,&fullepsBkg);
    VecDuplicate(eps,&fullepsDiff);
    setlayer_eps( fullepsBkg, odmNx,Nz, nlayers_total, izo, thickness,  bkg_eps[ifreq]);
    setlayer_eps(fullepsDiff, odmNx,Nz,nlayers_active, mzo,        mz, diff_eps[ifreq]);

    VecPointwiseMult(eps,eps,fullepsDiff);
    VecAXPY(eps,1.0,fullepsBkg);

    PetscScalar *_eps=(PetscScalar *)malloc(odmNx*Nz*sizeof(PetscScalar));
    mpi2array(eps, _eps,SCAL, odmNx*Nz);
    writetofile(PETSC_COMM_WORLD, "epsilon.qxz", _eps,SCAL, odmNx*Nz);

    PetscReal omega0=2*M_PI*freqs[ifreq];
    Mat DDefull;
    create_DDe(PETSC_COMM_WORLD, &DDefull,
	       odmNx, Nz,
	       pmlx, pmlx, pmlz, pmlz,
	       dx, dz,
	       omega0);
    VecScale(eps,-omega0*omega0);

    VecSum(eps,&tmp);
    PetscPrintf(PETSC_COMM_WORLD,"debug Job0: sum_eps %g %g \n",creal(tmp),cimag(tmp));
    
    MatDiagonalSet(DDefull,eps,ADD_VALUES);

    PetscReal nsub0 = sqrt(creal(bkg_eps[ifreq][0]));
    PetscReal kwav0 = nsub0*omega0;
    PetscReal alpha0 = angles[iangle] * M_PI/180;
    PetscScalar *Jsrc=(PetscScalar *)malloc(odmNx*sizeof(PetscScalar));
    srcJy(Jsrc, -px,odmNx,dx,dz, kwav0,alpha0,1);

    Vec b;
    VecDuplicate(eps, &b);
    VecSet(b,0.0);
    vecfill_zslice(b,Jsrc,odmNx,Nz,jz_src);
    VecScale(b,PETSC_i*omega0);

    VecSum(b,&tmp);
    PetscPrintf(PETSC_COMM_WORLD,"debug Job0: sum_b %g %g \n",creal(tmp),cimag(tmp));

    KSP kspfull;
    PetscInt itsfull=100;
    PetscInt maxit=15;
    setupKSPDirect(PETSC_COMM_WORLD, &kspfull, maxit);
    
    Vec x;
    VecDuplicate(eps,&x);
    SolveMatrixDirect(PETSC_COMM_WORLD,kspfull,DDefull,b,x,&itsfull,maxit);

    VecSum(x,&tmp);
    PetscPrintf(PETSC_COMM_WORLD,"debug Job0: sum_Efield %g %g \n",creal(tmp),cimag(tmp));
    
    PetscScalar *_x=(PetscScalar *)malloc(odmNx*Nz*sizeof(PetscScalar));
    mpi2array(x, _x,SCAL, odmNx*Nz);
    writetofile(PETSC_COMM_WORLD, "Efield.qxz", _x,SCAL, odmNx*Nz);

    Mat Pdagfull;
    create_P(PETSC_COMM_WORLD, &Pdagfull, 1,
	     odmNx, Nz,
	     pmlx,pmlx,pmlz,pmlz,
	     dx,dz,
	     omega0,
	     px,Nx,jz_mtr);
    Vec Pdagfullx,xconj;
    VecDuplicate(x,&Pdagfullx);
    VecDuplicate(x,&xconj);
    MatMult(Pdagfull,x,Pdagfullx);
    VecCopy(x,xconj);
    VecConjugate(xconj);
    PetscScalar Pz_near;
    VecTDot(xconj,Pdagfullx,&Pz_near);

    PetscInt Nxfar,Nzfar;
    getint("-Nxfar",&Nxfar,1);
    getint("-Nzfar",&Nzfar,1);
    PetscReal dxfar,dzfar;
    getreal("-dxfar",&dxfar,dx);
    getreal("-dzfar",&dzfar,dz);
    PetscReal nsub00=sqrt(creal(bkg_eps[0][0]));
    PetscReal alpha2=asin(nsub00*sin(alpha0));
    PetscReal fspotx=virtual_foclen*tan(alpha2) * image_compress_factor;
    PetscReal fspotz=foclen;
    PetscReal origx=Nx*dx/2.0;
    PetscReal origz=0;
    
    PetscReal fSz[Nxfar*Nzfar];
    Vec gefull,ghfull;
    VecDuplicate(x,&gefull);
    VecDuplicate(x,&ghfull);
    for(PetscInt jz=0;jz<Nzfar;jz++){
      for(PetscInt jx=0;jx<Nxfar;jx++){
	PetscReal xfar = fspotx + (jx+0.5)*dxfar - Nxfar*dxfar/2.0;
	PetscReal zfar = fspotz + (jz+0.5)*dzfar - Nzfar*dzfar/2.0;
	PetscInt j = jx + Nxfar * jz;
	
	gforms(PETSC_COMM_WORLD, gefull, ghfull,
	       odmNx,Nz,
	       pmlx,pmlx,pmlz,pmlz,
	       dx,dz,
	       omega0,
	       px,Nx,0,jz_mtr,
	       xfar,zfar,
	       origx,origz);

	PetscScalar tmp;
	VecSum(gefull,&tmp);
	PetscPrintf(PETSC_COMM_WORLD,"debug Job0: sum_ge %g %g \n",creal(tmp),cimag(tmp));
	VecSum(ghfull,&tmp);
	PetscPrintf(PETSC_COMM_WORLD,"debug Job0: sum_gh %g %g \n",creal(tmp),cimag(tmp));
	
	PetscScalar fHx, fEy;
	VecTDot(ghfull,x,&fHx);
	VecTDot(gefull,x,&fEy);
	fSz[j] = creal( - conj(fHx) * fEy );

	PetscPrintf(PETSC_COMM_WORLD,"debug Job0: fEy %g %g \n",creal(fEy),cimag(fEy));
	PetscPrintf(PETSC_COMM_WORLD,"debug Job0: fHx %g %g \n",creal(fHx),cimag(fHx));
	PetscPrintf(PETSC_COMM_WORLD,"debug Job0: fSz %g \n",fSz[j]);
	PetscPrintf(PETSC_COMM_WORLD,"debug Job0: fSz/Pz %g \n",fSz[j]/creal(Pz_near));
	PetscPrintf(PETSC_COMM_WORLD,"debug Job0: airyfactor %g \n",params.airyfactor[ispec]);
	PetscPrintf(PETSC_COMM_WORLD,"debug Job0: strehl %g \n",params.airyfactor[ispec] * fSz[j]/creal(Pz_near));
	
	PetscPrintf(PETSC_COMM_WORLD,"Farfield calculations %g percent done.\n", (double)j*100.0/((double)(Nxfar*Nzfar)));
	
      }
    }
    writetofile(PETSC_COMM_WORLD, "farSz.dat", fSz,REAL, Nxfar*Nzfar);
    PetscPrintf(PETSC_COMM_WORLD, "max Sz at focal spot: %g \n",find_max(fSz,Nxfar*Nzfar));
    if(Nzfar==1)
      PetscPrintf(PETSC_COMM_WORLD, "Pz at focal plane: %g \n",real_sum(fSz,0,Nxfar)*dxfar);
    PetscPrintf(PETSC_COMM_WORLD, "Transmitted Pz above the lens: %g \n",creal(Pz_near));
    PetscPrintf(PETSC_COMM_WORLD, "Strehl ratio: %g \n", params.airyfactor[ispec] * find_max(fSz,Nxfar*Nzfar)/creal(Pz_near));
    
    
    VecDestroy(&fullepsBkg);
    VecDestroy(&fullepsDiff);
    VecDestroy(&_edof);
    VecDestroy(&eps);
    VecDestroy(&b);
    VecDestroy(&x);
    MatDestroy(&Afull);
    MatDestroy(&Qfull);
    MatDestroy(&DDefull);
    free(hdof);
    free(edof);
    free(_eps);
    free(Jsrc);
    free(_x);
    KSPDestroy(&kspfull);

  }
  
  MPI_Barrier(PETSC_COMM_WORLD);
  PetscFinalize();

  
}
