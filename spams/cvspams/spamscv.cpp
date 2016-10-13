#include "spamscv.h"

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

    for(int col = 0; col < width; col++) {
        for(int row = 0; row < height; row++) {
            int spams_index = height * col + row, cv_index = width * row + col;
            (spams_image->rawX())[spams_index] = cv_image.at<double>(cv_index);
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

    for(int col = 0; col < width; col++) {
        for(int row = 0; row < height; row++) {
            int spams_index = height * col + row, cv_index = width * row + col;
            (spams_image->rawX())[spams_index] = cv_image.at<double>(cv_index);
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

    for(int col = 0; col < width; col++) {
        for(int row = 0; row < height; row++) {
            int cv_index = width * row + col, spams_index = height * col + row;
            cv_image.at<double>(cv_index) = (*spams_image)[spams_index];
        }
    }

    return cv_image;
}

cv::Mat spams2cv(Matrix<double>* spams_matrix) {
    int height = spams_matrix->m(), width = spams_matrix->n();

    cv::Mat cv_image(height, width, CV_64F);

    for(int col = 0; col < width; col++) {
        for(int row = 0; row < height; row++) {
            int cv_index = width * row + col, spams_index = height * col + row;
            cv_image.at<double>(cv_index) = (*spams_image)[spams_index];
        }
    }

    return cv_image;
}
