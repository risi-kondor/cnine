ROOTDIR=..
include $(ROOTDIR)/common.txt

INCLUDE= -I$(INCLUDEDIR)  -I$(SCALARDIR) -I$(TENSORDIR) -I$(TENSORVIEWDIR) -I$(TENSORARRAYDIR) -I$(TENSORARRAYDIR)/cell_ops -I$(TENSORARRAYDIR)/cell_maps  
INCLUDE+= -I$(BACKENDADIR) -I$(BACKENDBDIR) \
-I$(MANAGEDDIR) -I$(MSCALARDIR) -I$(MTENSORDIR) \
$(CENGINE_INCLUDES) 

TESTS=

DEPS= $(TENSORDIR)/*.hpp $(INCLUDEDIR)/*.hpp $(INCLUDEDIR)/cmaps/*.hpp $(SCALARDIR)/*.hpp $(MSCALARDIR)/*.hpp $(MTENSORDIR)/*.hpp

EXECS= 

OBJECTS=
ifdef EIGENDIR
OBJECTS+=EigenRoutines.o 
endif 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_OBJECTS+=Flock.o CtensorA_add.o CtensorA_add_plus.o CtensorA_copy.o CtensorA_plus.o
CUDA_OBJECTS+=TensorView_accumulators.o  
CUDA_OBJECTS+=BasicCtensorProducts.o  
CUDA_OBJECTS+=RtensorEinsumProducts.o  
CUDA_OBJECTS+=RtensorUtils.o  
CUDA_EXTERNS+=$(INCLUDEDIR)/Cnine_base.cu 
NVCCFLAGS+= -D_WITH_CUDA -D_WITH_CUBLAS
endif 

$(CUDA_OBJECTS): %.o: 
	$(NVCC) $(NVCCFLAGS) -c $*.cu $(CFLAGS) $(MACROS) $(INCLUDE) 


# -------------------------------------------------------------------------------------------------------------



#Flock.o: Flock.cu Makefile  
#	$(NVCC) $(NVCCFLAGS) -c Flock.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

#CtensorA_add.o: CtensorA_add.cu Makefile  
#	$(NVCC) $(NVCCFLAGS) -c CtensorA_add.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

#CtensorA_add_plus.o: CtensorA_add_plus.cu Makefile  
#	$(NVCC) $(NVCCFLAGS) -c CtensorA_add_plus.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

#CtensorA_copy.o: CtensorA_copy.cu Makefile  
#	$(NVCC) $(NVCCFLAGS) -c CtensorA_copy.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

#CtensorA_plus.o: CtensorA_plus.cu Makefile  
#	$(NVCC) $(NVCCFLAGS) -c CtensorA_plus.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

#TensorView_accumulators.o: TensorView_accumulators.cu Makefile  
#	$(NVCC) $(NVCCFLAGS) -c TensorView_accumulators.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

#EigenRoutines.o: EigenRoutines.cpp Makefile  
#	$(CC) -c EigenRoutines.cpp $(CFLAGS) $(INCLUDE) -I$(EIGENDIR) -D_WITH_EIGEN 

#BasicCtensorProducts.o: TensorView_accumulators.cu Makefile  
#	$(NVCC) $(NVCCFLAGS) -c BasicCtensorProducts.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

#RtensorEinsumProducts.o: TensorView_accumulators.cu Makefile  
#	$(NVCC) $(NVCCFLAGS) -c RtensorEinsumProducts.cu $(CFLAGS) $(MACROS) $(INCLUDE) 

#RtensorUtils.o: TensorView_accumulators.cu Makefile  
#	$(NVCC) $(NVCCFLAGS) -c RtensorUtils.cu $(CFLAGS) $(MACROS) $(INCLUDE) 



objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects  

clean: 
	rm -f $(CUDA_OBJECTS) 

anew: clean all


