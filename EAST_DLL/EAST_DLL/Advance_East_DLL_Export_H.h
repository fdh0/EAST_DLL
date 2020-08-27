#pragma once
#include<vector>
#include<opencv2/opencv.hpp>

using std::vector;

#define GIPDFINSPECT_LIBRARY

#ifdef GIPDFINSPECT_LIBRARY
#  define DLL_EXT __declspec(dllexport)
#else
#  define DLL_EXT __declspec(dllimport)
#endif

class ADVANCE_EAST_DETECT; // 实现类

class DLL_EXT Advance_East_DLL_Export_H
{
public:
	Advance_East_DLL_Export_H();
	~Advance_East_DLL_Export_H();

	// 加载模型
	int LoadModel(const char* model_path);

	//检测文本 
	int TextDetect(const char* img_path, float dThreshold, vector<vector<double>>& TextPos);

private:
	std::shared_ptr<ADVANCE_EAST_DETECT>m_pGCIAlgSynCheck;
};



