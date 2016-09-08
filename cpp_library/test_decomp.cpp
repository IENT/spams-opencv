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


	int step = 20, patch_size = 20;

	Matrix<double> patches;
	spams_image.extractPatches(patches, patch_size, step);

	//Print patch matrices for small images
	if(spams_image.width() <= 20) {
		std:cout  << patches.m() << std::endl << patches.n() << std::endl;
		patches.print("Untransformed Patches");
		std::cout  << std::endl;
	}

	//Set every second patch to white
	for(int x = 0; x < patches.n(); x++) {
		if(x % 2 == 0) {
			for(int y = 0; y < patches.m(); y++) {
				patches(y, x) = 255;
			}
		}
	}

	//Print transformed patch matrices for small images
	if(spams_image.width() <= 20) {
		std::cout  << std::endl;
		patches.print("Transformed Patches");
		std::cout  << std::endl;
	}

	//Interpolate 1 to 1 between the patches and the original image
	//resulting in a chess board type effect
	spams_image.combinePatches(patches, 1, step, true);

	//Convert to cv image and render
    Mat cv_image = spams2cv<double, unsigned char>(spams_image);

    namedWindow("Display window", WINDOW_NORMAL);
    imshow("Display window", cv_image);
    waitKey(0);
}

struct progs {
  const char *name;
  void (*prog)();
} progs[] = {
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
