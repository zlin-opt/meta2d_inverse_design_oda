#ifndef GUARD_optitemp_h
#define GUARD_optitemp_h

#include "petsc.h"
#include "nlopt.h"
#include "output.h"

typedef struct{

  nlopt_algorithm outer;
  nlopt_algorithm inner;
  int maxeval;
  int maxtime;
  int maxobj;
  
} alg_;

double optimize_generic(int DegFree, double *epsopt,
			double *lb, double *ub,
			nlopt_func obj, void *objdata,
			nlopt_func *constraint, void **constrdata, int nconstraints,
			alg_ alg,
			nlopt_result *result);

double dummy_obj(int ndofAll_with_dummy, double *dofAll_with_dummy, double *dofgradAll_with_dummy, void *print_at);

#endif
