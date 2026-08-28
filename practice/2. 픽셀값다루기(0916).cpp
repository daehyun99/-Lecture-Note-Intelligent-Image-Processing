#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>

#define _CRT_SECURE_NO_WARNINGS

using namespace cv;

typedef struct {
	int r, g, b;
}int_rgb;


int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); // 0까지 포함된 갯수임

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}

#define imax(x, y) ((x)>(y) ? x : y)
#define imin(x, y) ((x)<(y) ? x : y)

int BilinearInterpolation(int** image, int width, int height, double x, double y)
{
	int x_int = (int)x;
	int y_int = (int)y;

	int A = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int B = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];
	int C = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int D = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];

	double dx = x - x_int;
	double dy = y - y_int;

	double value
		= (1.0 - dx) * (1.0 - dy) * A + dx * (1.0 - dy) * B
		+ (1.0 - dx) * dy * C + dx * dy * D;

	return((int)(value + 0.5));
}


void DrawHistogram(char* comments, int* Hist)
{
	int histSize = 256; /// Establish the number of bins
						// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 512;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	Mat r_hist(histSize, 1, CV_32FC1);
	for (int i = 0; i < histSize; i++)
		r_hist.at<float>(i, 0) = Hist[i];
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// Display
	namedWindow(comments, WINDOW_AUTOSIZE);
	imshow(comments, histImage);

	waitKey(0);

}


int EX0916_1()	// (실습2) 영상 이진화
{
	int height, width;
	int** img = ReadImage((char*)"barbara.png", &height, &width);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] >= 128) {
				img[y][x] = 255;
			}

			else {
				img[y][x] = 0;
			}

			//printf("(%d, %d) ", y, x);
			//printf("%d ", img[y][x]);
		}

	}

	//printf("\n height = %d, width = %d", height, width);

	ImageShow((char*)"영상보기", img, height, width);

	return(0);
}


int EX0916_2(int** img, int height, int width)      // 1번에서 외부 참조로 수정 // 영상 이진화 (실습3)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] >= 200) {
				img[y][x] = 0;
			}
			else {
				img[y][x] = 255;
			}
		}
	}

	ImageShow((char*)"영상보기", img, height, width);

	return(0);
}


int EX0916_3()        // 2번 실행용
{
	int height, width;
	int** img0 = ReadImage((char*)"barbara.png", &height, &width);
	int** img1 = ReadImage((char*)"lena.png", &height, &width);

	EX0916_2(img0, height, width);
	EX0916_2(img1, height, width);

	return(0);
}

int EX0916_4(int** img, int height, int width)      // (실습4) 
{
	int half_height = (height / 2);
	int half_width = (width / 2);
	int x_j, y_j = 0;


	for (int y = 0; y < height; y++) {
		if (y < half_height) y_j = 0;
		else y_j = 100;
		for (int x = 0; x < width; x++) {
			if (x < half_width) x_j = 50;
			else x_j = 100;

			if (img[y][x] >= (x_j + y_j)) {
				img[y][x] = 255;
			}
			else {
				img[y][x] = 0;
			}
		}

	}

	ImageShow((char*)"영상보기", img, height, width);

	return(0);
}


int main()
{


	int height, width;
	int** img = ReadImage((char*)"barbara.png", &height, &width);

	int x0 = 192, y0 = 256, x1 = 320, y1 = 256, r0 = 150, r1 = 100, a = 16, b = 8;


	//EX0916_4(img, height, width);

	//원 그리기
	/*
	for (int y = 0; y < height; y++) {
	for (int x = 0; x < width; x++) {
	if ((x - x0) * (x - x0) + (y - y0) * (y - y0) < r0 * r0) {	//흰색원
	img[y][x] = 250;
	}
	else if ((x - x1) * (x - x1) + (y - y1) * (y - y1) < r1 * r1) {	//검은색원
	img[y][x] = 100;
	}
	else {	//주변을 검은색으로 만듬.
	img[y][x] = 0;
	}
	if (((x - x0) * (x - x0) + (y - y0) * (y - y0) < r0 * r0) && ((x - x1) * (x - x1) + (y - y1) * (y - y1) < r1 * r1)) {	//겹치는 부분
	img[y][x] = 180;
	}
	}
	}
	*/


	//타원 그리기

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if ((x - x0) * (x - x0) / (a*a) + (y - y0) * (y - y0) / (b*b) < 50)
				img[y][x] = 0;
		}
	}


	ImageShow((char*)"영상보기", img, height, width);

	return(0);
}
