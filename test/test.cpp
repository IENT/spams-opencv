#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <time.h>
#include <iostream>
#include <map>

#include "image.h"
#include "dicts.h"
#include "decomp.h"
#include "cppspams.h"
#include "linalg.h"


std::map<std::string, std::string> TEST_IMAGE_PATHS = {
    { "boat", "../data/boat.png" },
	{ "lena", "../data/lena.png"},
    { "grayscale_big", "../data/blackwhite.png" },
    { "grayscale_small", "../data/blackwhite_small.png" },
    { "grayscale_medium", "../data/blackwhite_medium.png" },
    { "chess_medium", "../data/chess_medium.png"}
};


Image<double>* cv2spams_image(cv::Mat cv_image) {
	// Check if the provided cv image is a valid input
	if(cv_image.channels() != 1) {
		throw "Cannot convert cv image with more than one channel to spams";
	}
	if(cv_image.type() != CV_64F) {
		throw "Can only convert CV_64F ie. double images to spams";
	}

	int height = cv_image.rows, width = cv_image.cols;

	Image<double>* spams_image = new Image<double>(height, width);

    //Manually copy data to spams image
	for(int col = 0; col < width; col++) {
		for(int row = 0; row < height; row++) {
			(spams_image->rawX())[height * col + row] = cv_image.at<double>(width * row + col);
    	}
    }

    return spams_image;
}

Matrix<double>* cv2spams_matrix(cv::Mat cv_matrix) {
	// Check if the provided cv image is a valid input
	if(cv_matrix.channels() != 1) {
		throw "Cannot convert cv image with more than one channel to spams";
	}
	if(cv_matrix.type() != CV_64F) {
		throw "Can only convert CV_64F ie. double images to spams";
	}

	int height = cv_matrix.rows, width = cv_matrix.cols;

	Matrix<double>* spams_matrix = new Matrix<double>(height, width);

    //Manually copy data to spams image
	for(int col = 0; col < width; col++) {
		for(int row = 0; row < height; row++) {
			(spams_matrix->rawX())[height * col + row] = cv_matrix.at<double>(width * row + col);
    	}
    }

    return spams_matrix;
}


cv::Mat spams2cv(Image<double>* spams_image) {
	// Check if the provided spams image is a valid input
	if(spams_image->numChannels() != 1) {
		throw "Cannot convert spams image with more than one channel to cv";
	}

	int height = spams_image->height(), width = spams_image->width();

	cv::Mat cv_image(height, width, CV_64F);

	//Manually copy data to spams image
	for(int col = 0; col < width; col++) {
		for(int row = 0; row < height; row++) {
    		cv_image.at<double>(width * row + col) = (*spams_image)[height * col + row];
    	}
    }

    return cv_image;
}

cv::Mat spams2cv(Matrix<double>* spams_matrix) {
	int height = spams_matrix->m(), width = spams_matrix->n();

	cv::Mat cv_image(height, width, CV_64F);

	//Manually copy data to spams image
	for(int col = 0; col < width; col++) {
		for(int row = 0; row < height; row++) {
    		cv_image.at<double>(width * row + col) = (*spams_matrix)[height * col + row];
    	}
    }

	return cv_image;
}

cv::Mat readCvImage(string filepath) {
	cv::Mat cv_image = cv::imread(filepath, -1);

    if(cv_image.empty()) {
        throw "Could not open or find the cv image";
    }
    cv_image.convertTo(cv_image, CV_64F);
    
    return cv_image;
}

Image<double>* readTestSpamsImage(string filepath) {
	return cv2spams_image(readCvImage(filepath));
}

void displayTestSpamsImage(Image<double>* spams_image) {
	cv::Mat cv_image = spams2cv(spams_image);
    cv_image.convertTo(cv_image, CV_8U);

    cv::namedWindow("Display window", cv::WINDOW_NORMAL);
    cv::imshow("Display window", cv_image);
    cv::waitKey(0);
}

void test_scale() {
    Image<double>* image = readTestSpamsImage(TEST_IMAGE_PATHS.at("boat"));

    image->scal(0.75);

    displayTestSpamsImage(image);
}

void test_patches() {
	Image<double>* spams_image = readTestSpamsImage(
		// Use big image as it seems that the combinePatches function
		// has problems with very small images
		TEST_IMAGE_PATHS.at("boat")
	);


	int patch_size = 20, step = patch_size;

	Matrix<double> patches;
	spams_image->extractPatches(patches, patch_size, step);

	//Print patch matrices for small images
	if(spams_image->width() <= 20) {
		std::cout  << patches.m() << std::endl << patches.n() << std::endl;
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
	if(spams_image->width() <= 20) {
		std::cout  << std::endl;
		patches.print("Transformed Patches");
		std::cout  << std::endl;
	}

	//Interpolate 1 to 1 between the patches and the original image
	//resulting in a chess board type effect
	spams_image->combinePatches(patches, 1, step, true);

	//Convert to cv image and render
	displayTestSpamsImage(spams_image);
}

void test_trainDL() {
	cv::Mat cv_image = readCvImage(TEST_IMAGE_PATHS.at("boat"));
    //Map from 0 - 255 to -128 - 127
    cv_image = cv_image - 128;

	//Extract non-overlapping patches from image
	int patch_size = 2, step = patch_size;

	Image<double>* spams_image = cv2spams_image(cv_image);
	Matrix<double> patches;

	spams_image->extractPatches(patches, patch_size, step);
	delete spams_image;

	//Learn dictionary with l0 norm constraint by one eg. only find only one
	//representation
	int dict_width = 2; // Denoted as k in spams
	Trainer<double> trainer(dict_width);
	ParamDictLearn<double> param;
	param.lambda = 1;
	param.mode = (constraint_type) 3;
	param.iter = 100;

	trainer.train(patches, param);

	//Print dictionary and patches
	Matrix<double> dictionary;
	trainer.getD(dictionary);

	patches.print("Patches - The columns should be either 0.5 or -0.5");
	dictionary.print("Dictionary - The columns should be either 0.5 or -0.5");
}

/// This is a cpp version of the coresponding matlab demo code and the results
/// can be compared with the latter
void test_trainDL_edges() {
	cv::Mat cv_image = readCvImage(TEST_IMAGE_PATHS.at("lena"));

    //Extract patches
	int patch_size = 8, step = 1;

	Image<double>* spams_image = cv2spams_image(cv_image);
	Matrix<double>* patches = new Matrix<double>();

	spams_image->extractPatches(*patches, patch_size, step);
	delete spams_image;

	//Substract mean from patches and normalize by l2-norm
	cv::Mat cv_patches = spams2cv(patches);

	cv_patches = cv_patches / 255;
	for(int i = 0; i < cv_patches.cols; i++) {
		cv_patches.col(i) = cv_patches.col(i) - cv::mean(cv_patches.col(i))[0];
		cv_patches.col(i) = cv_patches.col(i) / cv::norm(cv_patches.col(i));
	}

	//Override original patches with the ones transformed by opencv
	patches = cv2spams_matrix(cv_patches);

	//Learn dictionary
	int dict_width = 256;
	if (dict_width % 8 != 0) {
		throw "Only multiples of 8 can be used as dict width";
	}
	Trainer<double> trainer(dict_width, 400);
	ParamDictLearn<double> param;
	param.lambda = 0.15;
	param.iter = 1000;
	param.verbose = true;

	trainer.train(*patches, param);

	//Combine dictionary columns to patches and convert to cv image
	Matrix<double> dictionary;
	trainer.getD(dictionary);

	cv::Mat cv_dictionary = (spams2cv(&dictionary) + 1) * 128;
	Matrix<double>* dictionary_transformed = cv2spams_matrix(cv_dictionary);

	int patches_per_column = 16,
		cols = patches_per_column * patch_size + (patches_per_column - 1),
		rows = dict_width / 16 * patch_size + (patches_per_column - 1) ;
	Image<double>* spams_image_dictionary = new Image<double>(cols, rows);

	spams_image_dictionary->combinePatches(*dictionary_transformed, 0, patch_size + 1, false);

	cv::Mat cv_image_dictionary = spams2cv(spams_image_dictionary);
	delete spams_image_dictionary;

	//Display
	cv_image_dictionary.convertTo(cv_image_dictionary, CV_8U);

	cv::namedWindow("Display window", cv::WINDOW_NORMAL);
    cv::imshow("Display window", cv_image_dictionary);

    //Fit the patches with the dictionary resulting in the corresponding alphas
    Matrix<double> *path, alpha;
    SpMatrix<double> *sparse_alpha = cppLasso(*patches, dictionary, &path, false, 10, 0.15);
    sparse_alpha->toFull(alpha);
    delete sparse_alpha, patches;

    //Convert alphas and dictionary to cv and calculate residue between coding and
    //original image
	cv::Mat cv_alpha = spams2cv(&alpha);

	cv::Mat powers, sum_difference, sum_alpha;

	//Calculate square of difference between coding and original
	cv::pow(cv_patches - cv_dictionary * cv_alpha, 2, powers);
	//Calculate column sums
	cv::reduce(powers, sum_difference, 0, CV_REDUCE_SUM, CV_64F);
	cv::reduce(cv::abs(cv_alpha), sum_alpha, 0, CV_REDUCE_SUM, CV_64F);
	//Calculate residue as mean of the addition of the two sums
	double residue = cv::mean(0.5 * sum_difference + param.lambda * sum_alpha)[0];

	//Print residue
	cout << std::endl << std::endl
		 << "Residue of image and dictionary coding is: " << residue << std::endl;

	//Block for display window
    cv::waitKey(0);
}

struct progs {
  const char *name;
  void (*prog)();
} progs[] = {
    "scale", test_scale,
	"patches", test_patches,
	"dictionary", test_trainDL,
	"dictionary_edges", test_trainDL_edges,
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
