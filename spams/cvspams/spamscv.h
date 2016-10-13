#ifndef SPAMSCV_H
#define SPAMSCV_H

#include <opencv2/core/core.hpp>

#include "image.h"
#include "linalg.h"


/*! \brief Convert a cv image to a spams image
 *  
 *  @param cv_image cv matrix with one channel and double
    @return double spams image pointer
 *  */
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

/*! \brief Convert a cv image to a spams matrix
 *  
 *  @param cv_image single channel, double cv image
    @return double spams matrix pointer
 *  */
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
            (spams_matrix->rawX())[spams_index] = cv_matrix.at<double>(cv_index);
        }
    }

    return spams_matrix;
}

/*! \brief Convert a spams image to cv matrix
 *  
 *  @param spams_image single channel, double spams image
 *  @return double cv matrix
 *  */
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

/*! \brief Convert a spams matrix to cv matrix
 *  
 *  @param spams_matrix double spams matrix
 *  @return double cv matrix
 *  */
cv::Mat spams2cv(Matrix<double>* spams_matrix) {
    int height = spams_matrix->m(), width = spams_matrix->n();

    cv::Mat cv_image(height, width, CV_64F);

    for(int col = 0; col < width; col++) {
        for(int row = 0; row < height; row++) {
            int cv_index = width * row + col, spams_index = height * col + row;
            cv_image.at<double>(cv_index) = (*spams_matrix)[spams_index];
        }
    }

    return cv_image;
}

#endif
