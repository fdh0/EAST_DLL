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

class EAST_MODEL_DETECT; // ʵ����

class DLL_EXT East_DLL_Export_H
{
public:
	East_DLL_Export_H();
	~East_DLL_Export_H();

	// ����ģ��
	int LoadModel(const char* model_path);

	//����ı� 
	int TextDetect(const char* img_path, float dThreshold, vector<vector<double>>& TextPos);

private:
	std::shared_ptr<EAST_MODEL_DETECT>m_pGCIAlgSynCheck;
};

