ROOTDIR=../../..
include $(ROOTDIR)/common.txt

INCLUDE= -I$(INCLUDEDIR) -I$(SCALARDIR) -I$(TENSORDIR) -I$(TENSORVIEWDIR) -I$(BACKENDADIR) -I$(BACKENDBDIR) 

TESTS=$(patsubst %.cpp,%,$(wildcard *.cpp))

DEPS=$(TENSORDIR)/*.hpp $(INCLUDEDIR)/*.hpp $(SCALARDIR)/*.hpp $(BACKENDADIR)/*.hpp $(BACKENDBDIR)/*.hpp 

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_EXTERNS+=$(INCLUDEDIR)/Cnine_base.cu 
endif 


$(TESTS): %: %.cpp
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o $@ $@.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o $@ $@.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) $(EXECS) $(OBJECTS) $(CUDA_OBJECTS) 

anew: clean all



