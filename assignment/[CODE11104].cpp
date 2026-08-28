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

float** FloatAlloc2(int height, int width)
{
	float** tmp;
	tmp = (float**)calloc(height, sizeof(float*));
	for (int i = 0; i < height; i++)
		tmp[i] = (float*)calloc(width, sizeof(float));
	return(tmp);
}

void FloatFree2(float** image, int height, int width)
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

#define RoundUp(x) ((int)(x+0.5))

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






////////////( 10/14 )////////////////

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

void C_Histogram(int** img, int height, int width, int* C_Hist) {
	int Hist[256] = { 0 };
	GetHistogram_2(width, height, img, Hist);

	C_Hist[0] = Hist[0];

	for (int n = 1; n < 256; n++)
		C_Hist[n] = Hist[n] + C_Hist[n - 1];
	
}

void Norm_C_Histogram(int** img, int height, int width, int* NC_Hist)
{
	int C_Hist[256] = { 0 };
	C_Histogram(img, height, width, C_Hist); // 누적히스토그램 함수 재활용

	for (int n = 0; n < 256; n++)							// 여기 
		NC_Hist[n] = C_Hist[n] * 255 / (width * height);
		// NC_Hist[n] = C_Hist[n] / (width * height) * 255; // C언어 연산순서 때문에 값이 0이 된다.
		// NC_Hist[n] = (float)C_Hist[n] / (width * height) * 255 + 0.5; // 0.5 : 반올림 // float로 선언

}

void HistogramEqualization(int** img, int height, int width, int** img_out)
{
	int NC_Hist[256] = { 0 };

	Norm_C_Histogram(img, height, width, NC_Hist);

	// 히스토그램 평활화 : Y = T[X]
	// 입력 : img, 출력 : img_out
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			img_out[y][x] = NC_Hist[img[y][x]];
	//
}

////////////( 10/21 )////////////////

void Avg3x3(int** img, int height, int width, int** img_out) {
	// (3 X 3) 필터
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{

			if (x == 0 || x == (width - 1) || y == 0 || y == (height - 1))
			{
				img_out[y][x] = img[y][x];
			}
			else
			{
				img_out[y][x] = (img[y - 1][x - 1] + img[y - 1][x] + img[y - 1][x + 1]
					+ img[y][x - 1] + img[y][x] + img[y][x + 1]
					+ img[y + 1][x - 1] + img[y + 1][x] + img[y + 1][x + 1]) / 9.0;

			}
		}
}

void Avg3x3_two(int** img, int height, int width, int** img_out) {
	// (3 X 3) 필터
	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++)
		{

			if (x == 0 || x == (width - 1) || y == 0 || y == (height - 1))
			{
				img_out[y][x] = img[y][x];
			}
			else
			{
				
				for (int a = -1; a <= 1; a++)
					for (int b = -1; b <= 1; b++)
					{
						img_out[y][x] += img[y + b][x + a];
					}
				img_out[y][x] / 9.0 + 0.5;
			}
		}
	}
}


void AvgNxN(int N, int** img, int height, int width, int** img_out)
{
	int delta = (N - 1) / 2;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++)
		{

			if (x < delta || x >= (width - delta) || y < delta || y >= (height - delta))
			{
				img_out[y][x] = img[y][x];
			}
			else
			{
				int sum = 0;

				for (int a = -delta; a <= delta; a++)
					for (int b = -delta; b <= delta; b++)
						sum += img[y + b][x + a];

				img_out[y][x] = (float)sum / (N*N) + 0.5;
			}
		}
	}

}

void AvgNxN_two(int N, int** img, int height, int width, int** img_out)
{
	int delta = (N - 1) / 2;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			int sum = 0;

			for (int a = -delta; a <= delta; a++)
				for (int b = -delta; b <= delta; b++)
					sum += img[GetMin(GetMax(y + b, 0), height-1)][GetMin(GetMax(x + a,0),width-1)];

			img_out[y][x] = (float)sum / (N*N) + 0.5;
		}
	}

}




void Avg3x3_withMask(float mask[3][3], int **img,int height, int width, int ** img_out) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++)
		{

			if (x == 0 || x == (width - 1) || y == 0 || y == (height - 1)) {
				img_out[y][x] = img[y][x];
			}
			else {
				/*
				img_out[y][x] =
				mask[0][0] * img[y - 1][x - 1] + mask[0][1] * img[y - 1][x] + mask[0][2] * img[y - 1][x + 1]
				+ mask[1][0] * img[y][x - 1] + mask[1][1] * img[y][x] + mask[1][2] * img[y][x + 1]
				+ mask[2][0] * img[y + 1][x - 1] + mask[2][1] * img[y + 1][x] + mask[2][2] * img[y + 1][x + 1] + 0.5;
				*/


				float avg = 0.0;
				for (int i = -1; i <= 1; i++) {
					for (int j = -1; j <= 1; j++) {
						avg += mask[i + 1][j + 1] * img[y + i][x + j];
					}
				}
				img_out[y][x] = avg + 0.5;
			}
		}
	}
}

void Avg3x3_withMask_two(float** mask, int **img, int height, int width, int ** img_out) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++)
		{

			if (x == 0 || x == (width - 1) || y == 0 || y == (height - 1)) {
				img_out[y][x] = img[y][x];
			}
			else {
				/*
				img_out[y][x] =
				mask[0][0] * img[y - 1][x - 1] + mask[0][1] * img[y - 1][x] + mask[0][2] * img[y - 1][x + 1]
				+ mask[1][0] * img[y][x - 1] + mask[1][1] * img[y][x] + mask[1][2] * img[y][x + 1]
				+ mask[2][0] * img[y + 1][x - 1] + mask[2][1] * img[y + 1][x] + mask[2][2] * img[y + 1][x + 1] + 0.5;
				*/


				float avg = 0.0;
				for (int i = -1; i <= 1; i++) {
					for (int j = -1; j <= 1; j++) {
						avg += mask[i + 1][j + 1] * img[y + i][x + j];
					}
				}
				img_out[y][x] = Clipping(avg + 0.5) ; // 0~255로 값을 잘라줘서, 엣지 연산을할 때 지워줘야한다.
			}
		}
	}
}

void Avg3x3_withMask_NoClipping(float** mask, int **img, int height, int width, int ** img_out) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++)
		{

			if (x == 0 || x == (width - 1) || y == 0 || y == (height - 1)) {
				img_out[y][x] = img[y][x];
			}
			else {
				/*
				img_out[y][x] =
				mask[0][0] * img[y - 1][x - 1] + mask[0][1] * img[y - 1][x] + mask[0][2] * img[y - 1][x + 1]
				+ mask[1][0] * img[y][x - 1] + mask[1][1] * img[y][x] + mask[1][2] * img[y][x + 1]
				+ mask[2][0] * img[y + 1][x - 1] + mask[2][1] * img[y + 1][x] + mask[2][2] * img[y + 1][x + 1] + 0.5;
				*/


				float avg = 0.0;
				for (int i = -1; i <= 1; i++) {
					for (int j = -1; j <= 1; j++) {
						avg += mask[i + 1][j + 1] * img[y + i][x + j];
					}
				}
				img_out[y][x] = avg + 0.5; // 0~255로 값을 잘라줘서, 엣지 연산을할 때 지워줘야한다.
			}
		}
	}
}

void MagGradient_X(int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width-1; x++) {
		
			img_out[y][x] = abs(img[y][x + 1] - img[y][x]);
		
		
		}
	}




}

void MagGradient_Y(int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height-1; y++) {
		for (int x = 0; x < width; x++) {

			img_out[y][x] = abs(img[y+1][x] - img[y][x]);


		}
	}








}

void MagGradient(int** img, int height, int width, int** img_out)
{
	
	for (int y = 0; y < height-1; y++)
	{
		for (int x = 0; x < width - 1; x++)
			img_out[y][x] = abs(img[y + 1][x] - img[y][x]) + abs(img[y][x + 1] - img[y][x]);
	}

}

int FindMaxValue(int** img, int height, int width)	//int로 선언된 이유 : void로 선언되면 반환되는 값이 없기 때문이다.
{
	int maxvalue = img[0][0];

	for (int j = 0; j < height; j++)
		for (int i = 0; i < width; i++)
			maxvalue = GetMax(maxvalue, img[j][i]);



	return(maxvalue);
}


void NormalizeByMax(int** img, int height, int width, int** img_out)
{

	// 정규화를 위해 최대값 찾기
	int max_value = FindMaxValue(img, height, width);

	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			img_out[j][i] = (img[j][i] / (float)max_value) * 255;	//float를 붙여야함
		}
	}


	
}

////////////////////(11/04)//////////////////


void AbsImage(int** img, int height, int width)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img[y][x] = abs(img[y][x]);
		}
	}
}

void MagSOBEL_X(int** img, int height, int width, int** img_out)
{
	int N = 3;
	float** mask = (float**)FloatAlloc2(N, N);
	mask[0][0] = -1; mask[0][1] = -2; mask[0][2] = -1;
	mask[1][0] = 0;	 mask[1][1] = 0;  mask[1][2] = 0;
	mask[2][0] = 1;	 mask[2][1] = 2;  mask[2][2] = 1;

	Avg3x3_withMask_NoClipping(mask, img, height, width, img_out);

	for (int y = 0; y < height; y++)	//절댓값 찾기
	{
		for (int x = 0; x < width; x++)
			img_out[y][x] = abs(img_out[y][x]);
	}


	FloatFree2(mask, 3, 3);	//메모리 해제(프로그램이 커지면 문제가 된다.)
}

void MagSOBEL_Y(int** img, int height, int width, int** img_out)
{
	int N = 3;
	float** mask = (float**)FloatAlloc2(N, N);
	mask[0][0] = 1; mask[0][1] = 0; mask[0][2] = -1;
	mask[1][0] = 2;	 mask[1][1] = 0;  mask[1][2] = -2;
	mask[2][0] = 1;	 mask[2][1] = 0;  mask[2][2] = -1;
	Avg3x3_withMask_NoClipping(mask, img, height, width, img_out);

	for (int y = 0; y < height; y++)	//절댓값 찾기
	{
		for (int x = 0; x < width; x++)
			img_out[y][x] = abs(img_out[y][x]);
	}


	FloatFree2(mask, 3, 3);	//메모리 해제(프로그램이 커지면 문제가 된다.)
}

void MagSOBEL(int** img, int height, int width, int** img_out)
{
	int N = 3;
	float** mask = (float**)FloatAlloc2(N, N);
	mask[0][0] = 0; mask[0][1] = -2; mask[0][2] = -2;
	mask[1][0] = 2;	 mask[1][1] = 0;  mask[1][2] = -2;
	mask[2][0] = 2;	 mask[2][1] = 2;  mask[2][2] = 0;
	Avg3x3_withMask_NoClipping(mask, img, height, width, img_out);

	for (int y = 0; y < height; y++)	//절댓값 찾기
	{
		for (int x = 0; x < width; x++)
			img_out[y][x] = abs(img_out[y][x]);
	}


	FloatFree2(mask, 3, 3);	//메모리 해제(프로그램이 커지면 문제가 된다.)

}



void EX1104_1() {	// 라플라시안 구현
	int height, width;
	int** img = (int**)ReadImage("lena.png", &height, &width);
	int** laplacian = (int**)IntAlloc2(height, width);

	int N = 3;
	float** mask = (float**)FloatAlloc2(N, N);
	mask[0][0] = -1; mask[0][1] = -1; mask[0][2] = -1;
	mask[1][0] = -1; mask[1][1] = 8; mask[1][2] = -1;
	mask[2][0] = -1; mask[2][1] = -1; mask[2][2] = -1;
	Avg3x3_withMask_two(mask, img, height, width, laplacian);
	//Avg3x3_withMask
	//Avg3x3_withMask_two 둘의 차이 : mask를 이중포인터로 받는지 배열로 받는지

	// AbsImage(laplacian, height, width);   // 절대치를 취함
	{ // 절대치 취함
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
			laplacian[y][x] = abs(laplacian[y][x]);
	}
	}

	// c++은 abs 쓰면 알아서 다 한다?


	NormalizeByMax(laplacian, height, width, laplacian); // img_out을 0~255 내로정규화

	ImageShow("입력영상보기", img, height, width);
	ImageShow("출력영상보기", laplacian, height, width);



}

void EX1104_2_SOBEL_XY() {	// 실습9, 10, 11, 12  // 클리핑 조심
	int height, width;
	int** img = (int**)ReadImage("lena.png", &height, &width);
	int** sobel_x = (int**)IntAlloc2(height, width);
	int** sobel_y = (int**)IntAlloc2(height, width);
	

	MagSOBEL_X(img, height, width, sobel_x);
	MagSOBEL_Y(img, height, width, sobel_y);

	
	// img_out을 0~255 내로정규화, Display하기 전에 하는 것?
	NormalizeByMax(sobel_x, height, width, sobel_x);
	NormalizeByMax(sobel_y, height, width, sobel_y);

	//	ImageShow((char*)"출력 y영상보기", sobel_y, height, width);  // 안돼면 (char*) 삽입
	ImageShow("입력영상보기", img, height, width);
	ImageShow("출력 x영상보기", sobel_x, height, width);
	ImageShow("출력 y영상보기", sobel_y, height, width);

}

void EX1104_3_SOBEL() {	// 실습9, 10, 11, 12  // 클리핑 조심
	int height, width;
	int** img = (int**)ReadImage("lena.png", &height, &width);
	int** sobel_x = (int**)IntAlloc2(height, width);
	int** sobel_y = (int**)IntAlloc2(height, width);
	int** img_out = (int**)IntAlloc2(height, width);

	MagSOBEL_X(img, height, width, sobel_x);
	MagSOBEL_Y(img, height, width, sobel_y);

	for (int y = 0; y < height; y++) for (int x = 0; x < width; x++)
		img_out[y][x] = sobel_x[y][x] + sobel_y[y][x];


	// img_out을 0~255 내로정규화, Display하기 전에 하는 것?
	NormalizeByMax(img_out, height, width, img_out);

	//	ImageShow((char*)"출력 y영상보기", sobel_y, height, width);  // 안돼면 (char*) 삽입
	ImageShow("입력영상보기", img, height, width);
	ImageShow("출력 영상보기", img_out, height, width);

}

void EX1104_4_Sharpening_1()
{
	int height, width;
	int** img = (int**)ReadImage("lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int N = 3;
	float** mask = (float**)FloatAlloc2(N, N);
	//	mask[0][0] = -1; mask[0][1] = -1; mask[0][2] = -1;
	//	mask[1][0] = -1; mask[1][1] = 9; mask[1][2] = -1;
	//	mask[2][0] = -1; mask[2][1] = -1; mask[2][2] = -1;
	mask[0][0] = 0; mask[0][1] = -1; mask[0][2] = 0;
	mask[1][0] = -1; mask[1][1] = 5; mask[1][2] = -1;
	mask[2][0] = 0; mask[2][1] = -1; mask[2][2] = 0;

	Avg3x3_withMask_two(mask, img, height, width, img_out);


	//	ImageShow((char*)"출력 y영상보기", sobel_y, height, width);  // 안돼면 (char*) 삽입
	ImageShow("입력영상보기", img, height, width);
	ImageShow("출력 영상보기", img_out, height, width);

}

void EX1104_5_Sharpening_mask()
{
	int height, width;
	int** img = (int**)ReadImage("lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);


	int N = 3;
	float alpha = 0.25; // 클수록 에지가 더 강조됨

	char a[3] = { '1','2','3' };

	float** mask = (float**)FloatAlloc2(N, N);


	ImageShow("입력영상보기", img, height, width);

	for (int i = 1; i < 4; i++)
	{
		alpha = 0.25;
		mask[0][0] = -alpha*i; mask[0][1] = -alpha*i;        mask[0][2] = -alpha*i;
		mask[1][0] = -alpha*i; mask[1][1] = 1 + 8.0*alpha*i; mask[1][2] = -alpha*i;
		mask[2][0] = -alpha*i; mask[2][1] = -alpha*i;         mask[2][2] = -alpha*i;

		Avg3x3_withMask_two(mask, img, height, width, img_out);
		if (i ==1)
		{
			ImageShow("출력 영상보기1", img_out, height, width);
		}
		if (i == 2)
		{
			ImageShow("출력 영상보기2", img_out, height, width);
		}
		if (i == 3)
		{
			ImageShow("출력 영상보기3", img_out, height, width);
		}
	}
	

}

void main()
{
	EX1104_5_Sharpening_mask();

}

