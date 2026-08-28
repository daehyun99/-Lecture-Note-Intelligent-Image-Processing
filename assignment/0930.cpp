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
// 0916

void EX0916_1()
{
	int height, width;
	int** img = ReadImage((char*)"barbara.png", &height, &width);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] >= 200) {
				img[y][x] = 0;
			}

			else {
				img[y][x] = 255;
			}

			//printf("(%d, %d) ", y, x);
			//printf("%d ", img[y][x]);
		}

	}

	//printf("\n height = %d, width = %d", height, width);

	ImageShow((char*)"영상보기", img, height, width);
}


void EX0916_2(int** img, int height, int width)      // 1번에서 외부 참조로 수정
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
}


void EX0916_3()        // 2번 실행용
{
	int height, width;
	int** img0 = ReadImage((char*)"barbara.png", &height, &width);
	int** img1 = ReadImage((char*)"lena.png", &height, &width);

	EX0916_2(img0, height, width);
	EX0916_2(img1, height, width);
}

void EX0916_4()
{
	int height, width;
	int** img = ReadImage((char*)"barbara.png", &height, &width);

	int x0 = 192, y0 = 256, x1 = 320, y1 = 256, r0 = 150, r1 = 100, a = 16, b = 8;

	//원 그리기
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if ((x - x0) * (x - x0) + (y - y0) * (y - y0) < r0 * r0) {
				img[y][x] = 250;
			}
			else if ((x - x1) * (x - x1) + (y - y1) * (y - y1) < r1 * r1) {
				img[y][x] = 100;
			}
			else {
				img[y][x] = 0;
			}
			if (((x - x0) * (x - x0) + (y - y0) * (y - y0) < r0 * r0) && ((x - x1) * (x - x1) + (y - y1) * (y - y1) < r1 * r1)) {
				img[y][x] = 180;
			}
		}
	}


	/*
	//타원 그리기
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if ((x - x0) * (x - x0)/(a*a) + (y - y0) * (y - y0)/(b*b) < 256)
				img[y][x] = 0;
		}
	}
	*/

	ImageShow((char*)"영상보기", img, height, width);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 0923
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


void EX0923_1() // 함수화 없이 색변화
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out1[y][x] = img[y][x] + 50;
			img_out2[y][x] = img[y][x] - 50;
		}
	}

	ImageShow((char*)"출력1영상보기", img_out1, height, width);
	ImageShow((char*)"출력2영상보기", img_out2, height, width);
}



void EX0923_2() // 색변화 함수화하여 사용
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	AddValue2Image(50, img, height, width, img_out1);
	AddValue2Image(-50, img, height, width, img_out2);

	ImageShow((char*)"출력1영상보기", img_out1, height, width);
	ImageShow((char*)"출력2영상보기", img_out2, height, width);
}

void EX0923_3() // 클리핑 함수화하여 사용
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	AddValue2Image(50, img, height, width, img);

	ImageClipping(img, height, width, img_out);

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);
}

void EX0923_4()  // 영상혼합 함수화X
{
	int height, width;
	int** img1 = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img2 = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	float alpha = 0.5;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = alpha * img1[y][x] + (1.0 - alpha) * img2[y][x];
		}
	}

	ImageShow((char*)"입력1영상보기", img1, height, width);
	ImageShow((char*)"입력2영상보기", img2, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);
}


void EX0923_5(char* winname1, char* winname2, char* winname3) // 영상혼합 함수화
{
	int height, width;
	int** img1 = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img2 = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	float alpha = 0.5;
	ImageMixing(alpha, height, width, img1, img2, img_out);

	ImageShow(winname1, img1, height, width);
	ImageShow(winname2, img2, height, width);
	ImageShow(winname3, img_out, height, width);

	//EX0923_5((char*)"입력1영상보기", (char*)"입력2영상보기", (char*)"출력영상보기"); // 실행 시 이런식으로 실행
}


void EX0923_6() // 영상혼합 0.1~0.8
{
	int height, width;
	int** img1 = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img2 = (int**)ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	for (float alpha = 0.1; alpha < 0.9; alpha += 0.1) {
		ImageMixing(alpha, height, width, img1, img2, img_out);
		ImageShow((char*)"출력영상보기", img_out, height, width);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 0930
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

	int a = 150; // y = (255/a) * x, x < a

	Stretching_1(a, height, width, img, img_out);

	ImageShow((char*)"입력영상보기", img, height, width);
	ImageShow((char*)"출력영상보기", img_out, height, width);
}

void EX0930_2() // 스트레칭 실습3
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 1014


void C_Histogram(int height, int width, int** img, int* C_Hist) // 누적 히스토그램 구하기(실습3)
{
	int Hist[256] = { 0 };
	GetHistogram_1(height, width, img, Hist);

	for (int a = 0; a < 256; a++) {
		if (a == 0) C_Hist[a] = Hist[a];
		else C_Hist[a] = Hist[a] + C_Hist[a - 1];
	}
}


void EX1014_1() // 누적히스토그램 구하기(실습3)
{
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);

	int C_Hist[256] = { 0 };
	C_Histogram(height, width, img, C_Hist);

	ImageShow((char*)"입력영상보기", img, height, width);
	DrawHistogram((char*)"누적히스토그램", C_Hist); // histogram을 그려주는 함수
}




void main()
{
	EX1014_1();

}