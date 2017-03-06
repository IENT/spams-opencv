# OpenCV SPAMS Interface

This repostory contains a small set of conversion functions between OpenCV and SPAMS as well as a few examples how to use spams from cpp with opencv. Of course you need opencv installed.

## Build Instructions

Open *example/Makefile* and change `OPENCV_PATH`, so it points to OpenCV on your machine.

```Makefile
OPENCV_PATH = /your/opencv/path
```

Build and start demo. The possible arguments to the script can be found at the end of the *example.cpp* file. Be sure to execute the example form the right folder, as it includes hardcoded image paths which will lead to execution failure otherwise.

```bash
make
cd ..
bin/example test_patches
```
## OpenCV SPAMS Conversion

To use SPAMS from cpp include all subdirectories of the *spams/* folder. The file *spams/cvspams/cvspams.h* contains the OpenCV SPAMS conversion functions. To conform to the style of SPAMS no seperate `*.cpp` files were created. It has to be noted, that the conversion is currently restricted to single channel, double images and matrices.

## SPAMS Interface

The folder *spams/cppspams/* contains some example interface functions from SPAMS. These examples were made by the SPAMS authors and are not complete, but a lot of functions from SPAMS do not need an additional interface anyway, eg. dictionary learning can be used directly as shown in the `test_trainDL` example in *example/example.cpp*.
