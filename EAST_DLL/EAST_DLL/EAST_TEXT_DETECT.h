#pragma once

using namespace std;
using namespace cv::dnn;
using namespace cv;

//使用Avdance East模型
 class ADVANCE_EAST_DETECT
{
public:
	int TextDetect(const char* img_path, float Threshold, vector<vector<double>> &TextPos);
	int LoadModel(const char* model_path);
	void _decode_(const Mat &scores, const Mat &geometry, float Threshold, Mat &heibai);

public:
	cv::dnn::dnn4_v20200609::Net net;

private:
	static bool descendSort(std::vector<cv::Point> a, std::vector<cv::Point> b);
 };

 //使用EAST模型
 class EAST_MODEL_DETECT
 {
 public:
	 int TextDetect(Mat srcImg, float Threshold, vector<vector<double>> &TextPos);
	 int LoadModel(const char* model_path);
	 void _decode_(const Mat& scores, const Mat& geometry, float scoreThresh,
		 std::vector<RotatedRect>& detections, std::vector<float>& confidences);
 public:
	 cv::dnn::dnn4_v20200609::Net net;

 private:
	 static bool descendSort(std::vector<cv::Point> a, std::vector<cv::Point> b);
	 static bool GetCutWidthAuto(cv::Mat ImageIn, int XCutLength, int YCutLength, int XCutTolerance, int YCutTolerance, int & XRealCutLength, int & YRealCutLength, int& XNum, int & YNum);
	 static bool ImageParticion(cv::Mat ImageIn,int XCutLength,int YCutLength, int XNum, int YNum, vector<Mat>& ceil_img, vector<Rect>& rect);

 };