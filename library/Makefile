export PETSC_DIR=${HOME}/petsc-3.6.4
export PETSC_ARCH=arch-mumps-opt
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CLEANFILES = *.o

NLOPT_INC = -I /usr/local/include

CC=mpicc
CFLAGS   += -O3 -Wall -march=native -fcx-limited-range -fno-exceptions
INCFLAGS = -I. ${NLOPT_INC} ${PETSC_CC_INCLUDES}

LIBS4OBJS=maxwell.o input.o output.o solver.o array2vec.o misc.o dof2dom.o vec.o pwsrc.o varh.o farfield.o qforms.o adjoint.o ovmat.o filters.o optitemp.o strehl.o

lib: lib2d.a $(LIBS4OBJS)

lib2d.a: $(LIBS4OBJS)
	$(AR) crvs $@ *.o


