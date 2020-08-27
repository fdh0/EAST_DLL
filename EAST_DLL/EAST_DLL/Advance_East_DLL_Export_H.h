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

class ADVANCE_EAST_DETECT; // ʵ����

class DLL_EXT Advance_East_DLL_Export_H
{
public:
	Advance_East_DLL_Export_H();
	~Advance_East_DLL_Export_H();

	// ����ģ��
	int LoadModel(const char* model_path);

	//����ı� 
	int TextDetect(const char* img_path, float dThreshold, vector<vector<double>>& TextPos);

private:
	std::shared_ptr<ADVANCE_EAST_DETECT>m_pGCIAlgSynCheck;
};



