#include "pch.h"
#include "East_Dll_Export_H.h"
#include "EAST_TEXT_DETECT.h"

East_DLL_Export_H::East_DLL_Export_H()
{
	m_pGCIAlgSynCheck = std::make_shared<EAST_MODEL_DETECT>();
}

East_DLL_Export_H::~East_DLL_Export_H()
{
}

int East_DLL_Export_H::LoadModel(const char* model_path)
{
	int nResult = 0;
	try 
	{
		nResult = m_pGCIAlgSynCheck->LoadModel(model_path);
	}
	catch (...)
	{
		nResult = 1;
	}
	return nResult;
}

int East_DLL_Export_H::TextDetect(cv::Mat srcImg, float dThreshold, vector<vector<double>>& TextPos)
{
	int nResult = 0;
	try
	{
		nResult = m_pGCIAlgSynCheck->TextDetect(srcImg, dThreshold, TextPos);
	}
	catch (...)
	{
		nResult = 1;
	}
	return nResult;
}
