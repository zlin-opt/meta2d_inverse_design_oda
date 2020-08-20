export PETSC_DIR=${HOME}/petsc-3.6.4
export PETSC_ARCH=arch-mumps-opt
include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CLEANFILES = *.o *_exec

CC=mpicc

2d_INC = -I ../library/
2d_LIB = ../library/lib2d.a

NLOPT_INC = -I /usr/local/include
NLOPT_LIB = /usr/local/lib/libnlopt.a

CFLAGS   += -O3 -Wall -march=native -fcx-limited-range -fno-exceptions
INCFLAGS = -I. ${2d_INC} ${NLOPT_INC} ${PETSC_CC_INCLUDES} 
LIBS=$(2d_LIB) $(NLOPT_LIB) $(PETSC_LIB)

all: strehlopt_exec virtualstrehlopt_exec debug_exec

strehlopt.o: strehlopt.c
	$(CC) -c $(CFLAGS) $(INCFLAGS) $< -o $@
strehlopt_exec: strehlopt.o
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

virtualstrehlopt.o: virtualstrehlopt.c
	$(CC) -c $(CFLAGS) $(INCFLAGS) $< -o $@
virtualstrehlopt_exec: virtualstrehlopt.o
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)

debug.o: debug.c
	$(CC) -c $(CFLAGS) $(INCFLAGS) $< -o $@
debug_exec: debug.o
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
