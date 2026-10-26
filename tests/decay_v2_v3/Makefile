# Isolated build for the decayV2-vs-decayV3 comparison.
#
# Builds the UNMODIFIED sources from ../../src into ./bin, so this test never
# touches the top-level bin/ or any measurement data. Mirrors the flags of the
# project Makefile.

ROOT      := ../..
CXX       := g++
CXX_FLAGS := -std=c++2b -O2 -DNDEBUG -Wall -Wno-sign-compare

INCLUDE      := -I$(ROOT)/include -I$(ROOT)/Cuba-4.2.2
DIRLIBRARIES := -L$(ROOT)/Cuba-4.2.2
LIBRARIES    := -lboost_program_options -lgsl -lm -lcuba
DYNLINK      := -Wl,-rpath=$(PWD)/$(ROOT)/Cuba-4.2.2

BIN     := bin
TARGETS := $(BIN)/decayV2 $(BIN)/decayV3 $(BIN)/decayV4

.PHONY: all clean distclean

all: $(TARGETS)

$(BIN)/%: $(ROOT)/src/%.cpp | $(BIN)
	$(CXX) $(CXX_FLAGS) $(INCLUDE) -o $@ $< $(DIRLIBRARIES) $(LIBRARIES) $(DYNLINK)

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)

# also removes generated runs and reports
distclean: clean
	rm -rf out
