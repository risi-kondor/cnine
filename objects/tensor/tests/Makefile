ROOTDIR=../../..
include $(ROOTDIR)/common.txt

INCLUDE= -I$(INCLUDEDIR) -I$(SCALARDIR) -I$(TENSORDIR) 

TESTS= testRtensorObj testRtensorObj2 testRtensorObj3 \
testCtensorObj testCtensorObj2 testCtensorObj3 testCtensorSlice testCtensorPackObj \
testCtensorB

DEPS=$(TENSORDIR)/*.hpp $(INCLUDEDIR)/*.hpp $(SCALARDIR)/*.hpp  

EXECS= 

CUDA_OBJECTS=  
CUDA_EXTERNS=

ifdef WITH_CUDA
INCLUDE+=-I$(CUDA_HOME)/include
#CUDA_OBJECTS+=$(CETENSORDIR)/CtensorBpack.o 
CUDA_EXTERNS+=$(INCLUDEDIR)/Cnine_base.cu 
endif 




# -------------------------------------------------------------------------------------------------------------


testRtensorObj: testCtensorObj.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testRtensorObj testRtensorObj.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testRtensorObj testRtensorObj.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif

testRtensorObj2: testRtensorObj2.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testRtensorObj2 testRtensorObj2.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testRtensorObj2 testRtensorObj2.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif

testRtensorObj3: testRtensorObj3.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testRtensorObj3 testRtensorObj3.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testRtensorObj3 testRtensorObj3.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


# -------------------------------------------------------------------------------------------------------------


testCtensorObj: testCtensorObj.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorObj testCtensorObj.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCtensorObj testCtensorObj.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCtensorObj2: testCtensorObj2.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorObj2 testCtensorObj2.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCtensorObj2 testCtensorObj2.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif

testCtensorObj3: testCtensorObj3.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorObj3 testCtensorObj3.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCtensorObj3 testCtensorObj3.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif

testCtensorSlice: testCtensorSlice.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorSlice testCtensorSlice.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCtensorSlice testCtensorSlice.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testCtensorPackObj: testCtensorPackObj.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorPackObj testCtensorPackObj.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCtensorPackObj testCtensorPackObj.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


# -------------------------------------------------------------------------------------------------------------


testCtensorB: testCtensorB.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testCtensorB testCtensorB.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testCtensorB testCtensorB.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif


testSet: testSet.cpp $(DEPS)
ifdef WITH_CUDA
	$(NVCC) $(NVCCFLAGS) -o testSet testSet.cpp $(CUDA_EXTERNS) $(CUDA_OBJECTS) $(CFLAGS) $(MACROS) $(INCLUDE) $(LIBS) 
else
	$(CC) -o testSet testSet.cpp $(CFLAGS) $(INCLUDE) $(LIBS) 
endif







objects: $(OBJECTS) $(CUDA_OBJECTS)
tests: $(TESTS)
execs: $(EXECS)
derived: $(DERIVED) 

all: objects execs derived tests 

clean: 
	rm -f $(TESTS) $(EXECS) $(OBJECTS) $(CUDA_OBJECTS) 

anew: clean all





