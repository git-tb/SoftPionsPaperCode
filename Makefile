CXX       := g++
CXX_FLAGS := -std=c++2b
# CXX_FLAGS += -g
CXX_FLAGS += -O2
CXX_FLAGS += -DNDEBUG
CXX_FLAGS += -fopenmp
CXX_FLAGS += -Wall
CXX_FLAGS += -Wno-sign-compare

EXE 	:= NO_EXECUTBALE_SPECIFIED
BIN     := bin
SRC     := src

ARGS	:=

INCLUDE :=
DIRLIBRARIES := 
LIBRARIES :=

INCLUDE += -Iinclude
INCLUDE += -Iboost_1_82_0
# INCLUDE += -Icubature
INCLUDE += -ICuba-4.2.2

DIRLIBRARIES += -Lboost_1_82_0/lib
# DIRLIBRARIES += -Lcubature
DIRLIBRARIES += -LCuba-4.2.2

LIBRARIES += -lboost_program_options
LIBRARIES += -lgsl
LIBRARIES += -lm
LIBRARIES += -lcuba
# LIBRARIES += -lhcubature -lpcubature

DYNLINK :=

# DYNLINK += -Wl,-rpath=$(PWD)/boost_1_82_0/lib/
# DYNLINK += -Wl,-rpath=$(PWD)/cubature/
DYNLINK += -Wl,-rpath=$(PWD)/Cuba-4.2.2

run:
	./$(BIN)/$(EXE) $(ARGS)

build:
	$(CXX) $(CXX_FLAGS) $(INCLUDE) -o $(BIN)/$(EXE) $(SRC)/$(EXE).cpp $(DIRLIBRARIES) $(LIBRARIES) $(DYNLINK)

checklibs:
	ldd ./bin/$(EXE)

# EXAMPLE CALL IS make run EXE=spec ARGS="--initpath=data/init_20240725_145025/initialfields_piplus.csv"


