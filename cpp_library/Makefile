TARGETS= bin/test_decomp
LIBS= -lstdc++ -lblas -llapack -L ../../opencv-2.4.13/bin/ -L ../../opencv-2.4.13/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc
SRCS= test_decomp.cpp

#HEADERS= spams.h cppspams.h ../linalg/misc.h ../linalg/cblas_template.h \
	../linalg/linalg.h \
       	../decomp/decomp.h 

INCLUDE+=-I. -I../linalg/ -I../decomp/ -I../prox -I../dictLearn -I ../image  -I /data/weyer/opencv-2.4.13/modules/core/include -I /data/weyer/opencv-2.4.13/modules/highgui/include -I /data/weyer/opencv-2.4.13/modules/imgproc/include
CXXFLAGS = -std=c++11 -fopenmp -g   -DNDEBUG -DUSE_BLAS_LIB $(INCLUDE)

all: depend $(TARGETS)
	


clean:
	rm -f *.o *.a *.out

%.o : %.cpp 
	$(CXX) -c $(CXXFLAGS) $<

bin/test_decomp: test_decomp.o 
	$(CXX) $(CXXFLAGS) -o $@  $< $(LIBS)

# Dependencies
.depend: $(HEADERS)
	 touch .depend
	 makedepend -f .depend -- $(CXXFLAGS) -- $(SRCS)

depend: .depend
	touch depend

include .depend

