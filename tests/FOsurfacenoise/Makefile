# Isolated build for the freeze-out-surface noise comparison.
#
# Builds the UNMODIFIED sources from ../../src into ./bin, so this test never
# touches the top-level bin/ or any measurement data. Mirrors the flags of the
# project Makefile and of tests/decay_v2_v3.

ROOT      := ../..
CXX       := g++
CXX_FLAGS := -std=c++2b -O2 -DNDEBUG -Wall -Wno-sign-compare -fopenmp

INCLUDE      := -I$(ROOT)/include -I$(ROOT)/Cuba-4.2.2
DIRLIBRARIES := -L$(ROOT)/Cuba-4.2.2
LIBRARIES    := -lboost_program_options -lgsl -lgslcblas -lm -lcuba
DYNLINK      := -Wl,-rpath=$(PWD)/$(ROOT)/Cuba-4.2.2

BIN     := bin
TARGETS := $(BIN)/specV2 $(BIN)/decayV4

.PHONY: all run compare clean distclean

all: $(TARGETS)

$(BIN)/%: $(ROOT)/src/%.cpp | $(BIN)
	$(CXX) $(CXX_FLAGS) $(INCLUDE) -o $@ $< $(DIRLIBRARIES) $(LIBRARIES) $(DYNLINK)

$(BIN):
	mkdir -p $(BIN)

# build the two surfaces, then the primary and decay spectra for each
run: all
	$(ROOT)/venv/bin/python run.py

# figures + report from whatever is already in out/
compare:
	$(ROOT)/venv/bin/python compare.py

clean:
	rm -rf $(BIN)

# also removes the generated surfaces, runs and figures
distclean: clean
	rm -rf out data
