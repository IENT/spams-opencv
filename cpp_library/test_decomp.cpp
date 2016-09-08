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



Image<double> cv2spams(Mat image) {
	// Check if the provided cv image is a valid input
	if(image.channels() != 1) {
		throw "Cannot convert cv image with more than one channel to spams";
	}
	if(image.type() != CV_64F) {
		throw "Can only convert CV_64F ie. double images to spams";
	}

    Image<double> spams_image(image.cols, image.rows);
	
    //Manually copy data to spams image
    int l = image.cols * image.rows;
    for(int i = 0; i < l; i++) {
    	(spams_image.rawX())[i] = image.at<double>(i);
    }
    
    return spams_image;
}

Mat spams2cv(Image<double> image) {
	// Check if the provided spams image is a valid input
	if(image.numChannels() != 1) {
		throw "Cannot convert spams image with more than one channel to cv";
	}

	Mat cv_image(image.height(), image.width(), CV_64F);

	//Manually copy data to spams image
	int l = image.width() * image.height();
	for(int i = 0; i < l; i++) {
		cv_image.at<double>(i) = image[i];
	}
	
    return cv_image;
}

Image<double> readTestSpamsImage(string filepath) {
    Mat cv_image = imread(filepath, -1);

    if(cv_image.empty()) {
        throw "Could not open or find the cv image";
    }
    cv_image.convertTo(cv_image, CV_64F);
    
	return cv2spams(cv_image);
}

void displayTestSpamsImage(Image<double> spams_image) {
    Mat cv_image = spams2cv(spams_image);
    cv_image.convertTo(cv_image, CV_8U);

    namedWindow("Display window", WINDOW_NORMAL);
    imshow("Display window", cv_image);
    waitKey(0);
}

void test_scale() {
    Image<double> image = readTestSpamsImage(TEST_IMAGE_PATHS.at("boat"));

    image.scal(0.75);

    displayTestSpamsImage(image);
}

void test_patches() {
	Image<double> spams_image = readTestSpamsImage(
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
	displayTestSpamsImage(spams_image);
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
