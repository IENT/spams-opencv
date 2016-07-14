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
    
    //TODO: Is there an easier way for initialization?

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

void test_image() {
    int channels = 1;
    
    // Copied from opencv doc and adapted
    Mat image = imread("../data/boat.png", CV_LOAD_IMAGE_GRAYSCALE);

    if(! image.data) {
        cout <<  "Could not open or find the image" << std::endl;
        return;
    }

    int l1 = image.cols * image.rows * image.channels();
    double p1[l1];
    for(int i = 0; i < l1; i++) {
        p1[i] = (double) image.data[i];
//        cout << image.data[i] << " -> " << p1[i] << std::endl;
    }

    Image<double> I(p1, image.cols, image.rows, 
                           image.channels());
    
    I.scal(0.5);
    
    int l2 = I.width() * I.height() * image.channels();
    unsigned char p2[l2];
    for(int i = 0; i < l2; i++) {
        p2[i] = (unsigned char) I.rawX()[i];
//        cout << I.rawX()[i] << " -> " << p2[i] << std::endl;
    }
    
    Mat output(I.width(), I.height(), CV_LOAD_IMAGE_GRAYSCALE, p2);
    
    cout << image.cols << "  " << I.width() << std::endl;
    
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow("Display window", output);                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
}

struct progs {
  const char *name;
  void (*prog)();
} progs[] = { 
    "omp", test_omp,
    "lasso", test_lasso,
    "image", test_image,
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
