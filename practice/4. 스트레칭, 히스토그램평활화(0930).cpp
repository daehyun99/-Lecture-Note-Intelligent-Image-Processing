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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define GetMax(x, y) ((x > y) ? x : y)
#define GetMin(x, y) ((x < y) ? x : y)
#define Clipping(x) (GetMax(GetMin(x, 255), 0))


void AddValue2Image(  // 밝기 조절 9.23
	int value,		// 더해줄 값
	int** img_in,	// 입력영상
	int height,		// 영상의 높이
	int width,		// 영상의 폭
	int** img_out)	// 출력영상
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = img_in[y][x] + value;
		}
	}
}

void ImageClipping(int** img_in, int height, int width, int** img_out) { // 클리핑 9.23

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			/*
			// if 문을 이용한 클리핑
			img_out[y][x] = img_in[y][x];
			if (img_in[y][x] < 0) {
			img_out[y][x] = 0;
			}
			else if (img_in[y][x] > 255) {
			img_out[y][x] = 255;
			}
			else {}
			*/

			//img_out[y][x] = GetMax(0, GetMin(255, img_in[y][x])); // 클리핑(대소비교 매크로 사용)

			img_out[y][x] = Clipping(img_in[y][x]); // 클리핑(대소비교 매크로 사용)2
		}
	}
}


void ImageMixing(float alpha, int height, int width, int** img1, int** img2, int** img_out) { // 영상혼합 9.23
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = alpha * img1[y][x] + (1.0 - alpha) * img2[y][x];
		}
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//0930

#define RoundUp(x) ((int)(x+0.5))	//반올림 + cast 연산자

void Stretching_1(int a, int height, int width, int**img, int**img_out) { // 스트레칭 실습1 함수화
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] < a)
				img_out[y][x] = 255.0 / a * img[y][x]; // <- a를 int로 선언했을 경우 정수/정수 = 정수의 형태로 나와서 문제가 있음
			else
				img_out[y][x] = 255;
		}
	}
}


void Stretching_2(int a, int b, int c, int d, int height, int width, int** img, int** img_out) { // 스트레칭 실습3 함수화 + 반올림
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] < a)
				img_out[y][x] = RoundUp((float)c / a * img[y][x]);
			else if (img[y][x] < b)
				img_out[y][x] = RoundUp(((float)d - c) / (b - a) * (img[y][x] - a) + c);
			else
				img_out[y][x] = RoundUp((255.0 - d) / (255 - b) * (img[y][x] - b) + d);
		}
	}
}

void Stretching_3(int a, int height, int width, int** img, int** img_out) { // 스트레칭 실습1을 반올림처리
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] < a)
				img_out[y][x] = RoundUp(255.0 / a * img[y][x]);
			else
				img_out[y][x] = 255;
		}
	}
}

void GetHistogram_1(int height, int width, int** img, int *hist) // 히스토그램 함수화(실습2)
{
	for (int brightness = 0; brightness < 256; brightness++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (img[y][x] == brightness) hist[brightness]++;
			}
		}
	}
}

void GetHistogram_2(int height, int width, int** img, int* hist) // 히스토그램 함수화(실습2)의 간략화
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			hist[img[y][x]]++;
		}
	}
}


void EX0930_1() // 스트레칭 실습1
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int a = 50; // y = (255/a) * x, x < a

	Stretching_1(a, height, width, img, img_out);

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);
}

void EX0930_2() // 스트레칭 실습3
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int a = 100, b = 200, c = 100, d = 150; // y = (255/a) * x, x < a

	Stretching_2(a, b, c, d, height, width, img, img_out);

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);
}

void EX0930_3() // 스트레칭 실습1을 반올림처리
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int a = 150; // y = (255/a) * x, x < a

	Stretching_3(a, height, width, img, img_out);

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);
}


void EX0930_4() // 히스토그램(실습1)
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);

	int Hist[256] = { 0 };

	for (int brightness = 0; brightness < 256; brightness++) {
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				if (img[y][x] == brightness) Hist[brightness]++;
			}
		}
	}

	ImageShow((char*)"입력영상보기", img, height, width);
	DrawHistogram((char*)"히스토그램", Hist); // histogram을 그려주는 함수
}


void EX0930_5() // 히스토그램 함수화(실습2)
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);

	int Hist[256] = { 0 };
	//GetHistogram_1(height, width, img, Hist);
	GetHistogram_2(height, width, img, Hist);

	ImageShow((char*)"입력영상보기", img, height, width);
	DrawHistogram((char*)"히스토그램", Hist); // histogram을 그려주는 함수
}

void main()
{
	EX0930_5();

}