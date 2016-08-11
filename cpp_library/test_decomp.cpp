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

static const string TEST_IMAGE_PATH = "../data/boat.png";


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

Image<double>* cv2spams(Mat image) {
    Image<double>* spams_image = new Image<double>(image.cols, image.rows, image.channels());
	
    //Manually copy data to spams image
    double* X = spams_image->rawX();
    int l = image.cols * image.rows * image.channels();
    for(int i = 0; i < l; i++) {
        X[i] = (double) image.data[i];
    }
    
    return spams_image;
}

Mat spams2cv(Image<double>* image) {
	Mat cv_image(image->height(), image->width(), image->numChannels());
	
    //Manually copy data to spams image
	int l = image->width() * image->height() * image->numChannels();
	for(int i = 0; i < l; i++) {
		cv_image.data[i] = (*image)[i];
	}
	
    return cv_image;
}

template<typename T> class SpamsImage : public Image<T> {
	public: 
		/// Constructor_ind
		SpamsImage(INTM w, INTM h, INTM numChannels = 1) : Image<T>(w, h, numChannels) {};
		
		/// Constructor with existing data 
		SpamsImage(T* X, INTM w, INTM h, INTM numChannels = 1) 
			: Image<T>(X, w,h, numChannels) {};
		/// Empty constructor
		SpamsImage();
	
		/// Destructor
		~SpamsImage();
};

SpamsImage<double> test() {
	return SpamsImage<double>(10, 10, 1);
}

Image<double>* readImage(string filepath) {
    Mat cv_input = imread(filepath, CV_LOAD_IMAGE_GRAYSCALE);

    if(! cv_input.data) {
        throw "Could not open or find the image";
    }
    cv_input.convertTo(cv_input, )
    
	return cv2spams(cv_input);
}

void test_image() {
    Image<double>* image = readImage(TEST_IMAGE_PATH);

    image->scal(0.5);

    Mat cv_output = spams2cv(image);
    cv_output.convertTo(cv_output, CV_8UC1);
    
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow("Display window", cv_output);

    waitKey(0);
}

struct progs {
  const char *name;
  void (*prog)();
} progs[] = {
    "omp", test_omp,
    "lasso", test_lasso,
    "scale", test_image,
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
