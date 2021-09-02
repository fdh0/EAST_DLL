#include "pch.h"
#include<iostream>
#include<fstream>
#include<opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "EAST_TEXT_DETECT.h"

#include <ppl.h>
using namespace Concurrency;

using namespace std;
using namespace cv;
using namespace cv::dnn;

int ADVANCE_EAST_DETECT::LoadModel(const char* model_path)
{
	int nRet = 0;
	try
	{
		net = readNet(model_path);
	}
	catch(...)
	{

	}
	return nRet;
}

int ADVANCE_EAST_DETECT::TextDetect(const char* img_path,float Threshold, vector<vector<double>> &TextPos)
{
	int nRet = 0;
	try
	{	
		//Net net = readNet(model_path);
		auto detect_image_path = img_path;

		Mat srcImg = imread(detect_image_path);
		int or_h = srcImg.rows;
		int or_w = srcImg.cols;
	
		if (srcImg.empty())
		{
			nRet = 1; //针对异常情况
			return nRet;
			//cout << "read image success!" << endl;
		}

		//输出
		std::vector<Mat> output;
		std::vector<String> outputLayers(4);
		outputLayers[0] = "side_vertex_coord/convolution";
		outputLayers[1] = "side_vertex_code/convolution";
		outputLayers[2] = "inside_score/convolution";
		outputLayers[3] = "east_detect/concat";

		//检测图像
		Mat frame, blob;
		frame = srcImg.clone();
		//获取深度学习模型的输入
		blobFromImage(frame, blob, 1.0, Size(640,  640), Scalar(123.68, 116.78, 103.94), true, false);
		net.setInput(blob);
		//输出结果
		net.forward(output, outputLayers);

		//置信度
		Mat scores = output[3];
		//位置参数
		Mat geometry = output[0];

		// Decode predicted bounding boxes， 对检测框进行解码，获取文本框位置方向
		//文本框位置参数0
		Mat heibai( 160, 160, CV_8UC1);
	    _decode_(scores, geometry, Threshold, heibai);
		
		vector<vector< Point>> contours;  //用于保存所有轮廓信息


		findContours(heibai, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		//轮廓按照面积大小进行升序排序
		
		sort(contours.begin(), contours.end(), descendSort);//升序排序
		vector<vector<Point>>::iterator itc = contours.begin();

		while (itc != contours.end())
		{
			int y = itc->size();
			if (itc->size() < 20)
			{
				itc = contours.erase(itc);
			}
			else
			{
				++itc;
			}
		}

		//draw
		Mat B;
		heibai.copyTo(B);
		drawContours(B, contours, -1, Scalar(150, 0, 0), FILLED);
		heibai = heibai - B;
		findContours(heibai, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
		heibai.copyTo(B);

		vector<vector<Point>>::iterator itr = contours.begin();
		int last_i = 0;
		for (int i = 0; i < contours.size(); i++)
		{
			Mat tmp(contours.at(i));
			Moments moment = moments(tmp, false);
			if (moment.m00 != 0)//除数不能为0
			{
				int x = cvRound(moment.m10 / moment.m00);//计算重心横坐标
				int y = cvRound(moment.m01 / moment.m00);//计算重心纵坐标
				if (x < 30 || y < 30 || x>120 || y>120)
				{
					vector< vector< Point> > contours2; //用于保存面积不足100的轮廓
					for (int j = last_i; j < i; j++)
					{
						++itr;
					}
					last_i = i;
					contours2.push_back(*itr);
					drawContours(heibai, contours2, -1, Scalar(0, 0, 0), FILLED);
				}
			}

		}
		
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		dilate(heibai, heibai, element);
		findContours(heibai, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

		for (int i = 0; i < (int)contours.size(); i++)
		{
			RotatedRect rect = minAreaRect(contours[i]);
			Point2f fourPoint2f[4];
			rect.points(fourPoint2f);
			float ratio_h = (float)or_h / 640;
			float ratio_w = (float)or_w / 640;
			fourPoint2f[0].x = fourPoint2f[0].x * 4 * ratio_w;
			fourPoint2f[0].y = fourPoint2f[0].y * 4 * ratio_h;
			fourPoint2f[1].x = fourPoint2f[1].x * 4 * ratio_w;
			fourPoint2f[1].y = fourPoint2f[1].y * 4 * ratio_h;
			fourPoint2f[2].x = fourPoint2f[2].x * 4 * ratio_w;
			fourPoint2f[2].y = fourPoint2f[2].y * 4 * ratio_h;
			fourPoint2f[3].x = fourPoint2f[3].x * 4 * ratio_w;
			fourPoint2f[3].y = fourPoint2f[3].y * 4 * ratio_h;
			line(srcImg, fourPoint2f[0], fourPoint2f[1], Scalar(0, 255, 0), 2);
			line(srcImg, fourPoint2f[1], fourPoint2f[2], Scalar(0, 255, 0), 2);
			line(srcImg, fourPoint2f[2], fourPoint2f[3], Scalar(0, 255, 0), 2);
			line(srcImg, fourPoint2f[3], fourPoint2f[0], Scalar(0, 255, 0), 2);
			
			double t[8] = { fourPoint2f[0].x ,fourPoint2f[0].y ,fourPoint2f[1].x ,fourPoint2f[1].y,fourPoint2f[2].x,fourPoint2f[2].y, fourPoint2f[3].x ,fourPoint2f[3].y };
			vector<double> b;
			b.clear();
				for (int i = 0; i <= 7; i++)
				{
					b.push_back(t[i]);
				}
			TextPos.push_back(b);
		}
	}
	catch (...)
	{
		nRet = 1; //针对异常情况

	}

	return nRet;
}

void ADVANCE_EAST_DETECT::_decode_(const Mat & scores, const Mat & geometry, float Threshold, Mat & heibai)
{
	const int height = geometry.size[2];
	const int width = geometry.size[3];
	for (int y = 0; y < height; y++)
	{
		//识别概率
		const float *isinside = scores.ptr<float>(0, 0, y);
		const float *isbound = scores.ptr<float>(0, 1, y);
		const float *headorwei = scores.ptr<float>(0, 2, y);
		const float *x_1 = geometry.ptr<float>(0, 0, y);
		const float *y_1 = geometry.ptr<float>(0, 1, y);
		const float *x_2 = geometry.ptr<float>(0, 2, y);
		const float *y_2 = geometry.ptr<float>(0, 3, y);
		//遍历所有检测到的检测框
		for (int x = 0; x < width; x++)
		{
			float isinside_1 = isinside[x];
			float isbound_1 = isbound[x];
			float headorwei_1 = headorwei[x];

			float x_11 = x_1[x];
			float y_11 = y_1[x];
			float x_22 = x_2[x];
			float y_22 = y_2[x];

			float offsetX = x * 4.0f, offsetY = y * 4.0f;
			//低于阈值忽略该检测框

			if (isinside_1 < Threshold)
			{
				heibai.at<uchar>(y, x) = 0;
				continue;
			}
			//circle(srcImg, Point(offsetX, offsetY), 2, Scalar(0, 255, 0), -1);
			heibai.at<uchar>(y, x) = 255;

		}
	}
}

bool ADVANCE_EAST_DETECT::descendSort(vector<Point> a, vector<Point> b) {

	return a.size() > b.size();
}

int EAST_MODEL_DETECT::TextDetect(Mat srcImg, float Threshold, vector<vector<double>>& TextPos)
{
	int nRet = 0;
	try
	{
		float confThreshold = 0.5;  //0.6  0.5
		float nmsThreshold = 0.4; //0.2  0.4

		int inpWidth = 320;  //960
		int inpHeight =320;

		//auto detect_image_path = img_path;
		//Mat srcImg = imread(detect_image_path);

		int or_h = srcImg.rows;
		int or_w = srcImg.cols;

		if (srcImg.empty())
		{
			nRet = 1; //针对异常情况
			return nRet;
		}

		//检测图像
		//Mat frame, blob;
		//frame = srcImg.clone();

		//增加图像分块并行操作
		int XCutLength = inpWidth*2;
		int YCutLength = inpHeight*2;
		int XCutTolerance = 50;
		int YCutTolerance = 50;
		//Output
		int XRealCutLength = 0;
		int YRealCutLength = 0;
		int XNum = 0;
		int YNum = 0;

		//you can add resize if you want to get the bigger Character Text Pos
		//resize(srcImg, srcImg, Size(640, 640));

		// Get the cut size of image
		GetCutWidthAuto(srcImg, XCutLength, YCutLength, XCutTolerance, YCutTolerance, XRealCutLength, YRealCutLength, XNum, YNum);
		// Participate Image
		vector<Mat> ceil_img;
		vector<Rect> rectPos;
		//https://blog.csdn.net/jyjhv/article/details/83588402
		ImageParticion(srcImg, XRealCutLength, YRealCutLength, XNum, YNum, ceil_img, rectPos);

		// [7/29/2020 fdh] 增加并行处理 
		int nLabelCount = YNum * XNum;
		int nProcNum = nLabelCount;;
		int nCellNum = 1;
		int nMeanNum = 3;
		if (nLabelCount > 6)
		{
			nCellNum = (nLabelCount + nMeanNum - 1) / nMeanNum;
			nProcNum = (nLabelCount + nCellNum - 1) / nCellNum;
		}

		//获取深度学习模型的输入
		for (int nCellIndex = 0; nCellIndex < ceil_img.size(); nCellIndex++)
		{
		//parallel_for((UINT)0, (UINT)nProcNum, [&](UINT nProcIndex)//票面并行
		//{
			//for (int nInnerIndex = 0; nInnerIndex < nCellNum; nInnerIndex++)
			//{
				//int nCellIndex = nProcIndex * nCellNum + nInnerIndex;
				TRACE("%d", nCellIndex);
				if (nCellIndex >= nLabelCount)
				{
					break;
				}
				Mat blob;
				std::vector<Mat> output;

				vector<Rect> rectPos1 = rectPos;
				vector<Mat> ceil_img_ll = ceil_img;

				Mat fram_img = ceil_img_ll[nCellIndex];

				int nFrameWidth = fram_img.size[1];
				int nFrameHeight = fram_img.size[0];

				blobFromImage(fram_img, blob, 1.0, Size(inpWidth, inpHeight), Scalar(123.68, 116.78, 103.94), true, false);
				
				cv::dnn::dnn4_v20200609::Net net_ll = net;
				//输出
				std::vector<String> outputLayers(2);
				outputLayers[0] = "feature_fusion/Conv_7/Sigmoid";
				outputLayers[1] = "feature_fusion/concat_3";

				net_ll.setInput(blob);
				//输出结果
				net_ll.forward(output, outputLayers);// outputLayers

				//置信度
				Mat scores = output[0];
				//位置参数
				Mat geometry = output[1];

				// Decode predicted bounding boxes.
				std::vector<RotatedRect> boxes;
				std::vector<float> confidences;
				_decode_(scores, geometry, confThreshold, boxes, confidences);

				// Apply non-maximum suppression procedure.
				std::vector<int> indices;
				NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

				// Render detections.
				Point2f ratio((float)fram_img.cols / inpWidth, (float)fram_img.rows / inpWidth);
				//Point2f ratio((float)or_w / 640, (float)or_h / 640);
				
				for (size_t i = 0; i < indices.size(); ++i)
				{
					RotatedRect& box = boxes[indices[i]];

					Point2f vertices[4];
					box.points(vertices);
					for (int j = 0; j < 4; ++j)
					{
						vertices[j].x *= ratio.x;
						vertices[j].y *= ratio.y;
					}
					vector<double> b;
					b.clear();
					b.push_back(rectPos1[nCellIndex].x + vertices[0].x);
					b.push_back(rectPos1[nCellIndex].y + vertices[0].y);
					b.push_back(rectPos1[nCellIndex].x + vertices[2].x);
					b.push_back(rectPos1[nCellIndex].y + vertices[2].y);
					TextPos.push_back(b);
				}
			//}
		}//);
#ifdef DEBUG
		std::vector< double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		std::string label = format("Inference time: %.2f ms", t);

		//Convert
		size_t origsize = label.length() + 1;
		const size_t newsize = 100;
		size_t convertedChars = 0;
		wchar_t *wcstring = (wchar_t *)malloc(sizeof(wchar_t)*(label.length() - 1));
		mbstowcs_s(&convertedChars, wcstring, origsize, label.c_str(), _TRUNCATE);
		AfxMessageBox(wcstring);
#endif
	}
	catch (exception e)
	{
		nRet = 1; //针对异常情况
	}
	return nRet;
}

int EAST_MODEL_DETECT::LoadModel(const char * model_path)
{
	int nRet = 0;
	try
	{
		net = readNet(model_path);
	}
	catch (exception e)
	{
		nRet = 1;
	}
	return nRet;
}

void EAST_MODEL_DETECT::_decode_(const Mat& scores, const Mat& geometry, float scoreThresh,
	std::vector<RotatedRect>& detections, std::vector<float>& confidences)
{
	detections.clear();
	CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
	CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
	CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);

	const int height = scores.size[2];
	const int width = scores.size[3];
	for (int y = 0; y < height; ++y)
	{
		const float* scoresData = scores.ptr<float>(0, 0, y);
		const float* x0_data = geometry.ptr<float>(0, 0, y);
		const float* x1_data = geometry.ptr<float>(0, 1, y);
		const float* x2_data = geometry.ptr<float>(0, 2, y);
		const float* x3_data = geometry.ptr<float>(0, 3, y);
		const float* anglesData = geometry.ptr<float>(0, 4, y);
		for (int x = 0; x < width; ++x)
		{
			float score = scoresData[x];
			if (score < scoreThresh)
				continue;

			// Decode a prediction.
			// Multiple by 4 because feature maps are 4 time less than input image.
			float offsetX = x * 4.0f, offsetY = y * 4.0f;
			float angle = anglesData[x];
			float cosA = std::cos(angle);
			float sinA = std::sin(angle);
			float h = x0_data[x] + x2_data[x];
			float w = x1_data[x] + x3_data[x];

			Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
				offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
			Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
			Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
			RotatedRect r(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);
			detections.push_back(r);
			confidences.push_back(score);
		}
	}
}
/*
函数功能：自动计算裁切图像的尺寸
输入参数：图像，欲打算的裁切大小
		容忍度
输出参数：
		实际裁切的大小
		实际裁切的数目
*/
bool EAST_MODEL_DETECT::GetCutWidthAuto(cv::Mat ImageIn, int XCutLength, int YCutLength, int XCutTolerance, int YCutTolerance, int & XRealCutLength, int & YRealCutLength, int & XNum, int & YNum)
{
	int nWidth = ImageIn.size[0];
	int nHeight = ImageIn.size[1];

	if (nWidth <= 0 || nHeight <= 0)
	{
		return false;
	}

	if (nWidth < XCutLength || nHeight < YCutLength)
	{
		XRealCutLength = nWidth;
		YRealCutLength = nHeight;
		XNum = YNum = 1;
	}
	// Row
	int nXCutNum = nWidth / XCutLength;

	//增加约束最大横向分3分
	if (nXCutNum > 3)
	{
		XNum = nXCutNum = 3;
		XRealCutLength = nWidth * 1.0 / nXCutNum;
	}
	else
	{
		if (nWidth % XCutLength < XCutTolerance)
		{
			XRealCutLength = nWidth * 1.0 / nXCutNum;
			XNum = nXCutNum;
		}
		else
		{
			XNum = nXCutNum + 1;
			XRealCutLength = nWidth * 1.0 / XNum;
		}
	}

	//Column 
	int nYCutNum = nHeight / YCutLength;
	//增加约束最大z向分3分
	if (nYCutNum > 3)
	{
		YNum = nYCutNum=3;
		YRealCutLength = nHeight * 1.0 / nYCutNum;
	}
	else
	{
		if (nHeight % YCutLength < YCutTolerance)
		{
			YRealCutLength = nHeight * 1.0 / nYCutNum;
			YNum = nYCutNum;
		}
		else
		{
			YNum = nYCutNum + 1;
			YRealCutLength = nHeight * 1.0 / YNum;
		}
	}

	return true;
}

bool EAST_MODEL_DETECT::ImageParticion(cv::Mat ImageIn, int XCutLength, int YCutLength,int XNum,int YNum,vector<Mat>& ceil_img,vector<Rect>& rectPos)
{
	vector<int> name;
	for (int t = 0; t < XNum * YNum; t++) name.push_back(t);
	Mat image_cut, roi_img;
	for (int j = 0; j < XNum; j++)
	{
		for (int i = 0; i < YNum; i++)
		{
			Rect rect(i * YCutLength, j * XCutLength, YCutLength, XCutLength);
			rectPos.push_back(rect);
			image_cut = Mat(ImageIn, rect);
			roi_img = image_cut.clone();
			ceil_img.push_back(roi_img);
		}
	}
	return true;
}
