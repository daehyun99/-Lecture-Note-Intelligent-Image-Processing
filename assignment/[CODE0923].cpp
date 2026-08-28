#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>


using namespace cv;

typedef struct {
	int r, g, b;
}int_rgb;


int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i<height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i<height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i<img.rows; i++)
		for (int j = 0; j<img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
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

	for (int i = 0; i<img.rows; i++)
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
	for (int i = 0; i<height; i++)
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
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++) {
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
		= (1.0 - dx)*(1.0 - dy)*A + dx*(1.0 - dy)*B
		+ (1.0 - dx)*dy*C + dx*dy*D;

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
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// Display
	namedWindow(comments, WINDOW_AUTOSIZE);
	imshow(comments, histImage);

	waitKey(0);

}

/////(실습1) 밝기값에 일정한 값을 더해주기

void EX0923_1()	
{
	int height, width;
	int** img = (int**)ReadImage((char*)"barbara.png", &height, &width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);
	
	
	// img_out1은 밝기값이 +50
	img_out1 = img;
	img_out2 = img;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			img_out1[i][j] += 50;
		}
	}
	


	// img_out2은 밝기값이 -50

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			img_out2[i][j] -= 50;
		}
	}
	
	/*
	//교수님 코드
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out1[y][x] = img[y][x] + 50;
			img_out2[y][x] = img[y][x] - 50;
		}
	}

	
	for (int y = 0; y; y= height; y++){
		if (img[y][x] < 0)
		{
			img[y][x] = 0;
		}
		else if (img[y][x] > 255)
		{
			img[y][x] = 255;
		}
		else
		{

		}
	}
	/////*/

	ImageShow((char*)"출력1영상보기", img_out1, height, width);
	ImageShow((char*)"출력2영상보기", img_out2, height, width);
}


/////(실습1) 함수화
/*
void AddValue2Image(int value, int** img_in, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = img_in[y][x] + value;
		}
	}
}
void main()
{
	int height, width;
	int** img = (int**)ReadImage("barbara.png", &height, &width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	AddValue2Image(50, img, height, width, img_out1);
	AddValue2Image(-50, img, height, width, img_out2);

	ImageShow("출력1영상보기", img_out1, height, width);
	ImageShow("출력2영상보기", img_out2, height, width);
}
*/

/////(실습1) 함수화 

	void AddValue2Image(
		int value,		//더해줄 값	//여기있는 변수들이 main함수에 있는 변수들의 값들을 복사한다. 메모리 주소를 다르게 사용한다.
		int** img,		//입력영상
		int height,		//영상의 높이
		int width,		//영상의 폭
		int** out_img	//출력영상 // out_img와 main함수 안의 img_out1은 다른 변수이다.
	);
	
	/*
void EX0923_2()
{
	int height, width;
	int** img = (int**)ReadImage("barbara.png", &height, &width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	AddValue2Image(50, img, height, width, img_out1);
	AddValue2Image(-100, img, height, width, img_out2);

	ImageShow("출력1영상보기", img_out1, height, width);
	ImageShow("출력2영상보기", img_out2, height, width);
}
*/


/////(실습2)클리핑 구현, 클리핑의 함수화
void AddValue2Image(
	int value,		//더해줄 값	//여기있는 변수들이 main함수에 있는 변수들의 값들을 복사한다. 메모리 주소를 다르게 사용한다.
	int** img,		//입력영상
	int height,		//영상의 높이
	int width,		//영상의 폭
	int** out_img	//출력영상 // out_img와 main함수 안의 img_out1은 다른 변수이다.
)
{
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			out_img[y][x] = img[y][x] + value;


}

//#define SQ(x) x*x	//SQ(x) ((x)*(x))형태로 해야함? 
//SQ(x - 1) + 2; //(x-1) * (x-1) + 2
#define GetMax(x, y) ((x>y) ? x : y)
#define GetMin(x, y) ((x<y) ? x : y)
#define Clipping(x) GetMax(GetMin(x, 255), 0)

void ImageClipping(int** img_in, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			//////////////////////if문을 이용한 클리핑 //////////////////
			//if (img_in[y][x] < 0)	img_out[y][x] = 0;
			//else if (img_in[y][x] > 255)	img_out[y][x] = 255;
			//else     img_out[y][x] = img_in[y][x];	//이 조건이 없으면 화면이 검은색으로 나옴

			///////////////////////매크로를 이용한 클리핑2///////////////
			//img_out[y][x] = GetMax(GetMin(img_in[y][x], 255), 0);

			///////////////////////매크로를 이용한 클리핑3///////////////
			img_out[y][x] = Clipping(img_in[y][x]);

		}
	}
}
/*
void EX0923_3()
{
	int height, width;
	int** img = (int**)ReadImage("barbara.png", &height, &width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	AddValue2Image(50, img, height, width, img_out1);	//AddValue2Image(50, img, height, width, img);입력값 위에 출력값이 덮어써진다?

	ImageClipping(img_out1, height, width, img_out2);

	ImageShow("출력영상보기", img, height, width);
	ImageShow("출력1영상보기", img_out1, height, width);
	ImageShow("출력2영상보기", img_out2, height, width);
}

*/
/////(실습3)매크로의 활용
/*
//#define GetMax(x, y) ((x>y) ? x : y)
//#define GetMin(x, y) ((x<y) ? x : y)

void main()
{
	int height, width;
	int** img = (int**)ReadImage("barbara.png", &height, &width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	int maxvalue = GetMax(3, 2);
	int minvalue = GetMin(3, 2);
	
	int a = 300, b = -10, c = 200;

	//GEtMax, GetMin 함수로 클리핑함수 구현하기
	//a = GetMax(GetMin(a, 255), 0);
	//b = GetMax(GetMin(b, 255), 0);
	//c = GetMax(GetMin(c, 255), 0);

	printf("%d %d %d", a, b, c);
	ImageClipping(img, height, width, img_out1);
	ImageShow("출력1영상보기", img_out1, height, width);
	//ImageShow("출력2영상보기", img_out2, height, width);
}
*/


void ImageMixing(float alpha, int height, int width, int** img1, int** img2, int** img_out)
{
	for (int y = 0; y<height; y++) {
		for (int x = 0; x<width; x++) {
			img_out[y][x] = alpha*img1[y][x] + (1.0 - alpha)*img2[y][x];
		}
	}
}

void EX0923_4()
{
	int height, width;
	int** img1 = (int**)ReadImage("barbara.png", &height, &width);
	int** img2 = (int**)ReadImage("lena.png", &height, &width);
		int** img_out = (int**)IntAlloc2(height, width);

	float alpha = 0.2;

	ImageMixing(alpha, height, width, img1, img2, img_out);


	ImageShow("입력1영상보기", img1, height, width);
	ImageShow("입력2영상보기", img2, height, width);
	ImageShow("출력영상보기", img_out, height, width);
}

void main()
{
	EX0923_4();
}