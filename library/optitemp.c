#include "petsc.h"
#include "nlopt.h"
#include "optitemp.h"

extern int count;

#undef __FUNCT__
#define __FUNCT__ "optimize_generic"
double optimize_generic(int DegFree, double *epsopt,
			double *lb, double *ub,
			nlopt_func obj, void *objdata,
			nlopt_func *constraint, void **constrdata, int nconstraints,
			alg_ alg,
			nlopt_result *result)
{

  nlopt_opt opt;
  nlopt_opt local_opt;
  int i;
  double maxf;

  opt = nlopt_create(alg.outer, DegFree);
  nlopt_set_lower_bounds(opt,lb);
  nlopt_set_upper_bounds(opt,ub);
  nlopt_set_maxeval(opt,alg.maxeval);
  nlopt_set_maxtime(opt,alg.maxtime);

  //if(alg.outer==11) nlopt_set_vector_storage(opt,4000);
  if(alg.inner){
    local_opt=nlopt_create(alg.inner, DegFree);
    nlopt_set_ftol_rel(local_opt, 1e-6); //1e-11);
    nlopt_set_maxeval(local_opt,10000);
    nlopt_set_local_optimizer(opt,local_opt);
  }

  if(nconstraints){
    for(i=0;i<nconstraints;i++){
      nlopt_add_inequality_constraint(opt,constraint[i],constrdata[i],1e-8);
    }
  }

  if(obj){
    if(alg.maxobj)
      nlopt_set_max_objective(opt,obj,objdata);
    else
      nlopt_set_min_objective(opt,obj,objdata);
    *result=nlopt_optimize(opt,epsopt,&maxf);
  }

  nlopt_destroy(opt);
  if(alg.inner) nlopt_destroy(local_opt);

  return maxf;
  
}

#undef __FUNCT__
#define __FUNCT__ "dummy_obj"
double dummy_obj(int ndofAll_with_dummy, double *dofAll_with_dummy, double *dofgradAll_with_dummy, void *data)
{

  if(dofgradAll_with_dummy){
    int i;
    for(i=0;i<ndofAll_with_dummy-1;i++){
      dofgradAll_with_dummy[i]=0;
    }
    dofgradAll_with_dummy[ndofAll_with_dummy-1]=1.0;
  }
  count++;
  PetscPrintf(PETSC_COMM_WORLD,"******dummy value at step %d is %g \n",count,dofAll_with_dummy[ndofAll_with_dummy-1]);

  int *print_at = (int *)data;
  if(*print_at>0 && (count%*print_at)==0){
    char output_filename[PETSC_MAX_PATH_LEN];
    sprintf(output_filename,"outputdof_step%d.txt",count);
    writetofile(PETSC_COMM_WORLD,output_filename,dofAll_with_dummy,REAL,ndofAll_with_dummy-1);
  }

  return dofAll_with_dummy[ndofAll_with_dummy-1];

}
