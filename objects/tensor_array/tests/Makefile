ROOTDIR=../../..
include $(ROOTDIR)/common.txt

INCLUDE= -I$(INCLUDEDIR) -I$(INCLUDEDIR)/cmaps -I$(SCALARDIR) -I$(TENSORDIR) 
INCLUDE+= -I$(TENSORARRAYDIR) -I$(TENSORARRAYDIR)/cell_ops 
INCLUDE+= -I$(TENSORVIEWDIR) -I$(TENSORVIEWDIR)/ops -I$(TENSORVIEWDIR)/functions  

TESTS= testRtensorArray \
testCtensorArray testCtensorArray2 testCtensorArray_Cmaps testCtensorArray_Cmaps3 \
testCtensorArray_accumulator \
testCtensorArray_constructors testCtensorArray_broadcast testCtensorArray_devices testCtensorArray_reshape \
testCtensorArray_copy_kernel testCtensorArray_plus_kernel
TESTS+= test_CtensorB_array

DEPS=$(TENSORDIR)/*.hpp $(INCLUDEDIR)/*.hpp $(INCLUDEDIR)/cmaps/*.hpp $(SCALARDIR)/*.hpp $(TENSORARRAYDIR)/*.hpp \
$(TENSORARRAYDIR)/cell_ops/*.hpp  

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
CUDA_OBJECTS+=$(CNINE_CUDADIR)/CtensorA_add.o $(CNINE_CUDADIR)/CtensorA_add_plus.o \
$(CNINE_CUDADIR)/CtensorA_copy.o $(CNINE_CUDADIR)/CtensorA_plus.o 

CUDA_EXTERNS+=$(INCLUDEDIR)/Cnine_base.cu 
endif 




# -------------------------------------------------------------------------------------------------------------


testRtensorArray: testRtensorArray.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testRtensorArray testRtensorArray.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -ferror-limit=1 -o testRtensorArray testRtensorArray.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


# -------------------------------------------------------------------------------------------------------------


testCtensorArray: testCtensorArray.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorArray testCtensorArray.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -ferror-limit=1 -o testCtensorArray testCtensorArray.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCtensorArray2: testCtensorArray2.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorArray2 testCtensorArray2.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -ferror-limit=1 -o testCtensorArray2 testCtensorArray2.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCtensorArray_Cmaps: testCtensorArray_Cmaps.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorArray_Cmaps testCtensorArray_Cmaps.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -ferror-limit=1 -o testCtensorArray_Cmaps testCtensorArray_Cmaps.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCtensorArray_Cmaps2: testCtensorArray_Cmaps2.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorArray_Cmaps2 testCtensorArray_Cmaps2.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -ferror-limit=1 -o testCtensorArray_Cmaps2 testCtensorArray_Cmaps2.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCtensorArray_Cmaps3: testCtensorArray_Cmaps3.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorArray_Cmaps3 testCtensorArray_Cmaps3.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -ferror-limit=1 -o testCtensorArray_Cmaps3 testCtensorArray_Cmaps3.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCtensorArray_accumulator: testCtensorArray_accumulator.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorArray_accumulator testCtensorArray_accumulator.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -ferror-limit=1 -o testCtensorArray_accumulator testCtensorArray_accumulator.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCtensorArray_copy_kernel: testCtensorArray_copy_kernel.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorArray_copy_kernel testCtensorArray_copy_kernel.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -ferror-limit=1 -o testCtensorArray_copy_kernel testCtensorArray_copy_kernel.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCtensorArray_plus_kernel: testCtensorArray_plus_kernel.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorArray_plus_kernel testCtensorArray_plus_kernel.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -ferror-limit=1 -o testCtensorArray_plus_kernel testCtensorArray_plus_kernel.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


test_CtensorB_array: test_CtensorB_array.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o test_CtensorB_array test_CtensorB_array.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -ferror-limit=1 -o test_CtensorB_array test_CtensorB_array.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif




# -------------------------------------------------------------------------------------------------------------



testCtensorArray_constructors: testCtensorArray_constructors.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorArray_constructors testCtensorArray_constructors.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -ferror-limit=1 -o testCtensorArray_constructors testCtensorArray_constructors.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCtensorArray_broadcast: testCtensorArray_broadcast.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorArray_broadcast testCtensorArray_broadcast.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -ferror-limit=1 -o testCtensorArray_broadcast testCtensorArray_broadcast.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCtensorArray_devices: testCtensorArray_devices.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorArray_devices testCtensorArray_devices.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -ferror-limit=1 -o testCtensorArray_devices testCtensorArray_devices.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCtensorArray_reshape: testCtensorArray_reshape.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorArray_reshape testCtensorArray_reshape.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -ferror-limit=1 -o testCtensorArray_reshape testCtensorArray_reshape.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif




objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) $(EXECS) $(OBJECTS) $(CUDA_OBJECTS) 

anew: clean all





