OPENCV_PATH = ../../opencv

TARGETS= ../bin/example
LIBS= -lstdc++ -lblas -llapack \
    -L $(OPENCV_PATH)/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc
SRCS= example.cpp

INCLUDE+=-I. \
    -I../spams/cppspams -I../spams/linalg/ -I../spams/decomp/ -I../spams/prox \
    -I../spams/dictLearn -I ../spams/image -I ../spams/cvspams \
    -I $(OPENCV_PATH)/include
CXXFLAGS = -std=c++11 -fopenmp -g   -DNDEBUG -DUSE_BLAS_LIB $(INCLUDE)

all: depend $(TARGETS)
	


clean:
	rm -f *.o *.a *.out

%.o : %.cpp 
	$(CXX) -c $(CXXFLAGS) $<

../bin/example: example.o 
	$(CXX) $(CXXFLAGS) -o $@  $< $(LIBS)

# Dependencies
.depend: $(HEADERS)
	 touch .depend
	 makedepend -f .depend -- $(CXXFLAGS) -- $(SRCS)

depend: .depend
	touch depend

include .depend

