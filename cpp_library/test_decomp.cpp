#include "cppspams.h"
#include "image.h"

#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>


using namespace cv;
using namespace std;


static struct timespec tstart, tend;

float delta_t(struct timespec &t1,struct timespec &t2) {
  float sec = (float)(t2.tv_sec - t1.tv_sec);
  float ms = (float)(t2.tv_nsec - t1.tv_nsec) / 1000000.; 
  float t = (sec * 1000. + ms) / 1000.;
  return t;
}

std::map<std::string, std::string> TEST_IMAGE_PATHS = {
    { "boat", "../data/boat.png" },
    { "grayscale_big", "../data/blackwhite.png" },
    { "grayscale_small", "../data/blackwhite_small.png" },
    { "grayscale_medium", "../data/blackwhite_medium.png" }
};

void test_omp() {
  std::cout << "OMP" << std::endl;
  int m(64), n(100000), p(200);
   Matrix<double> X(m,n);
   X.setAleat();
   double* prD = new double[m*p];
   Matrix<double> D(prD,m,p); 
   D.setAleat(); 
   D.normalize();
   const int L = 10;
   double eps = 1.0;
   double lambda = 0.;
   SpMatrix<double> spa;
   clock_gettime(CLOCK_REALTIME,&tstart);
   cppOMP(X,D,spa,&L,&eps,&lambda);
   clock_gettime(CLOCK_REALTIME,&tend);
   float nbs = X.n() / delta_t(tstart,tend);
   std::cout << nbs << " signals processed per second." << std::endl;
   delete[](prD);

}

void test_lasso() {
    int m = 3;
    int n = 4;
    int p = 1;
    
    //Static allocation for matrices is used, so that matrices can be
    //initialized easily. The mapping of the array is column major ie. the 
    //arrays have to be initialized 'transposed'!
    
    double prD[m*p] = { 2, 1.8, 0.5 };    
    Matrix<double> D(prD, m, p);

    double prX[m*n] = { 1, 1, 0
                      , 2, 2, 1
                      , 3, 3, 2
                      , 4, 4, 0
                      };
    Matrix<double> X(prX, m, n);

    //Create empty sparse matrix
    Matrix<double> *path;


    SpMatrix<double> *spa = cppLasso(X, D, &path, false, 10, 0.15);

    std::cout << "Lasso algorithm completed:" << std::endl << std::endl;
    D.print("Matrix D");
    std:cout  << std::endl;
    X.print("Matrix X");
    std::cout  << std::endl;
    spa->print("Result Matrix");
    std::cout << std::endl;
    
    delete spa;
}

template<typename I, typename O> Image<O> cv2spams(Mat_<I> image) {
    Image<O> spams_image(image.cols, image.rows, image.channels());
	
    //Manually copy data to spams image
    int l = image.cols * image.rows * image.channels();
    for(int i = 0; i < l; i++) {
    	(spams_image.rawX())[i] = (O) image(i);
    }
    
    return spams_image;
}

template<typename I, typename O> Mat_<O> spams2cv(Image<I> image) {
	Mat_<O> cv_image(image.height(), image.width());
	//Manually copy data to spams image
	int l = image.width() * image.height() * image.numChannels();
	for(int i = 0; i < l; i++) {
		cv_image(i) = (O) image[i];
	}
	
    return cv_image;
}

template<typename T> Image<T> readImage(string filepath) {
    Mat cv_input = imread(filepath, -1);

    if(cv_input.empty()) {
        throw "Could not open or find the image";
    }
    //TODO template function correctly or removed it
    cv_input.convertTo(cv_input, CV_64F);
    
	return cv2spams<T, T>(cv_input);
}

void test_scale() {
    Image<double> image = readImage<double>(TEST_IMAGE_PATHS.at("boat"));

    image.scal(0.75);

    Mat cv_output = spams2cv<double, unsigned char>(image);

    namedWindow("Display window", WINDOW_NORMAL);
    imshow("Display window", cv_output);
    waitKey(0);
}

void test_patches() {
	Image<double> spams_image = readImage<double>(
		// Use big image as it seems that the patch functions have
		// problems with very small images or patch sizes
		TEST_IMAGE_PATHS.at("boat")
	);

	Matrix<double> X;
	image.extractPatches(X, 2, 2);

	std:cout  << std::endl;
	X.print("Matrix X");
	std::cout  << std::endl;
}

struct progs {
  const char *name;
  void (*prog)();
} progs[] = {
    "omp", test_omp,
    "lasso", test_lasso,
    "scale", test_scale,
	"patches", test_patches,
};
int main(int argc, char** argv) {
    for(int i = 1;i < argc;i++) {
        bool found = false;
        for (int j = 0;j < (sizeof(progs) / sizeof(struct progs));j++) {
            if(strcmp(argv[i],progs[j].name) == 0) {
                found = true;
                (progs[j].prog)();
                break;
            }
        }
        if(! found) {
          std::cout << "!! " << argv[i] << " not found" << std::endl;
          return 1;
        }
    }
    return 0;
}
