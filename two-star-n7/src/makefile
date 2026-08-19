CC=gcc
CFLAGS=-Wall -Wextra -Wno-unused-result -std=gnu99 -O2

# GSL paths (Homebrew on macOS ARM)
GSL_PREFIX=$(shell brew --prefix gsl 2>/dev/null || echo /usr/local)
GSL_INC=-I$(GSL_PREFIX)/include
GSL_LIB=-L$(GSL_PREFIX)/lib -lgsl -lgslcblas

all: hmc

test_prior: test_prior.o prior.o hull.o
	$(CC) $(CFLAGS) test_prior.o prior.o hull.o -o test_prior -lm

test_prior.o: test_prior.c prior.h hull.h
	$(CC) $(CFLAGS) $(GSL_INC) -c test_prior.c -o test_prior.o

# Run the full verification battery: C self-contained tests + R
# cross-check (hull vs chull(), IS estimator vs pure-R Eq. 14,
# analytical gradient vs numDeriv::grad).
test: test_prior
	./test_prior
	Rscript test_prior.R

hmc: hmc.o hmc_driver.o graph_2410.o prior.o hull.o
	$(CC) $(CFLAGS) hmc.o graph_2410.o hmc_driver.o prior.o hull.o -o hmc -lm $(GSL_LIB)

graph_2410.o: graph_2410.c graph_2410.h
	$(CC) $(CFLAGS) $(GSL_INC) -c graph_2410.c -o graph_2410.o

hmc.o: hmc.c *.h
	$(CC) $(CFLAGS) $(GSL_INC) -c hmc.c -o hmc.o

hmc_driver.o: hmc_driver.c *.h
	$(CC) $(CFLAGS) $(GSL_INC) -c hmc_driver.c -o hmc_driver.o

prior.o: prior.c prior.h hull.h
	$(CC) $(CFLAGS) $(GSL_INC) -c prior.c -o prior.o

hull.o: hull.c hull.h
	$(CC) $(CFLAGS) $(GSL_INC) -c hull.c -o hull.o

clean:
	rm -f *.o
	rm -f hmc test_prior
