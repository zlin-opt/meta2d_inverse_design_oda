#include "input.h"

//functions to read from stdin which can take in default values 
#undef __FUNCT__
#define __FUNCT__ "getreal"
PetscErrorCode getreal(const char *flag, PetscReal *var, PetscReal autoval)
{
  PetscErrorCode ierr;
  PetscBool flg;
  ierr=PetscOptionsGetReal(PETSC_NULL,flag,var,&flg); CHKERRQ(ierr);
  if(!flg) *var=autoval;
  ierr=PetscPrintf(PETSC_COMM_WORLD,"--%s is %g \n",flag,*var); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "getint"
PetscErrorCode getint(const char *flag, PetscInt *var, PetscInt autoval)
{
  PetscErrorCode ierr;
  PetscBool flg;
  ierr=PetscOptionsGetInt(PETSC_NULL,flag,var,&flg); CHKERRQ(ierr);
  if(!flg) *var=autoval;
  ierr=PetscPrintf(PETSC_COMM_WORLD,"--%s is %d \n",flag,*var); CHKERRQ(ierr);

  PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "getstr"
PetscErrorCode getstr(const char *flag, char *strin, const char default_strin[])
{
  PetscErrorCode ierr;
  PetscBool flg;
  ierr=PetscOptionsGetString(PETSC_NULL,flag,strin,PETSC_MAX_PATH_LEN-1,&flg); CHKERRQ(ierr);
  if(!flg) strcpy(strin,default_strin);
  ierr=PetscPrintf(PETSC_COMM_WORLD,"--%s is %s \n",flag,strin);

  PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "getintarray"
PetscErrorCode getintarray(const char *flag, PetscInt *z, PetscInt *nz, PetscInt default_val)
{
  PetscErrorCode ierr;
  PetscBool flg;
  PetscInt i,nget=*nz;
  char buffer[PETSC_MAX_PATH_LEN], tmp[PETSC_MAX_PATH_LEN];
  ierr=PetscOptionsGetIntArray(PETSC_NULL,flag,z,&nget,&flg); CHKERRQ(ierr);
  if(nget!=*nz){
    PetscPrintf(PETSC_COMM_WORLD,"!!!!WARNING: %d values expected for %s but received %d.\n",*nz,flag,nget);
    *nz=nget;
  }
  if(!flg) {
    for(i=0;i<*nz;i++){
      z[i]=default_val;
    }
  }
  strcpy(buffer," ");
  for(i=0;i<*nz;i++){
    sprintf(tmp,"%d, ",z[i]);
    strcat(buffer,tmp);
  }
  PetscPrintf(PETSC_COMM_WORLD,"--%s is %s total: %d  \n",flag,buffer,*nz);

  PetscFunctionReturn(ierr);

}

#undef __FUNCT__
#define __FUNCT__ "getrealarray"
PetscErrorCode getrealarray(const char *flag, PetscReal *r, PetscInt *nr, PetscReal default_val)
{
  PetscErrorCode ierr;
  PetscBool flg;
  PetscInt i,nget=*nr;
  char buffer[PETSC_MAX_PATH_LEN], tmp[PETSC_MAX_PATH_LEN];
  ierr=PetscOptionsGetRealArray(PETSC_NULL,flag,r,&nget,&flg); CHKERRQ(ierr);
  if(nget!=*nr){
    PetscPrintf(PETSC_COMM_WORLD,"!!!!WARNING: %d values expected for %s but received %d.\n",*nr,flag,nget);
    *nr=nget;
  }
  if(!flg) {
    for(i=0;i<*nr;i++){
      r[i]=default_val;
    }
  }
  strcpy(buffer," ");
  for(i=0;i<*nr;i++){
    sprintf(tmp,"%g, ",r[i]);
    strcat(buffer,tmp);
  }
  PetscPrintf(PETSC_COMM_WORLD,"--%s is %s total: %d  \n",flag,buffer,*nr);

  PetscFunctionReturn(ierr);

}

#undef __FUNCT__
#define __FUNCT__ "readfromfile"
PetscErrorCode readfromfile(char *name, void *data, TYPE typein, PetscInt n)
{

  PetscReal *ptreal=PETSC_NULL;
  PetscScalar *ptscal=PETSC_NULL;
  if(typein==REAL)
    ptreal=(PetscReal *)data;
  if(typein==SCAL)
    ptscal=(PetscScalar *)data;
  
  FILE *ptf;
  PetscInt i;
  PetscReal tmp;
  ptf = fopen(name,"r");
  for (i=0;i<n;i++)
    {
      if(fscanf(ptf,"%lf",&tmp)==1){

	if(typein==REAL){
	  if(ptreal!=PETSC_NULL)
	    ptreal[i] = tmp;
	  else
	    SETERRQ(PETSC_COMM_WORLD,1,"ptreal is NULL in readfromfile function.");
	}
	  
	if(typein==SCAL){
	  if(ptscal!=PETSC_NULL)
	    ptscal[i] = tmp + PETSC_i * 0.0;
	  else
	    SETERRQ(PETSC_COMM_WORLD,1,"ptscal is NULL in readfromfile function.");
	}
	  
      }else{

	printf("ERROR Reading from file \n");

      }

    }
  fclose(ptf);

  PetscFunctionReturn(0);
  
}

