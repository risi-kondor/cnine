ROOTDIR=../../..
include $(ROOTDIR)/common.txt

INCLUDE= -I$(INCLUDEDIR) -I$(INCLUDEDIR)/cmaps -I$(SCALARDIR) -I$(TENSORDIR) -I$(TENSORDIR)/ops \
-I$(TENSORVIEWDIR) -I$(TENSORVIEWDIR)/ops  


TESTS= testRtensor_view testCtensorView_accumulate 

DEPS=$(TENSORDIR)/*.hpp $(INCLUDEDIR)/*.hpp $(SCALARDIR)/*.hpp  $(TENSORVIEWDIR)/*.hpp 

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
#CUDA_OBJECTS+=$(CETENSORDIR)/CtensorBpack.o 
CUDA_OBJECTS+=$(CUDADIR)/TensorView_accumulators.o 
CUDA_EXTERNS+=$(INCLUDEDIR)/Cnine_base.cu 
endif 




# -------------------------------------------------------------------------------------------------------------


testRtensor_view: testRtensor_view.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testRtensor_view testRtensor_view.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testRtensor_view testRtensor_view.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCtensorView_accumulate: testCtensorView_accumulate.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorView_accumulate testCtensorView_accumulate.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCtensorView_accumulate testCtensorView_accumulate.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif




objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) $(EXECS) $(OBJECTS) $(CUDA_OBJECTS) 

anew: clean all



