#include "pch.h"
#include "Advance_East_DLL_Export_H.h"
#include "EAST_TEXT_DETECT.h"

Advance_East_DLL_Export_H::Advance_East_DLL_Export_H()
{
	m_pGCIAlgSynCheck = std::make_shared<ADVANCE_EAST_DETECT>();
}

Advance_East_DLL_Export_H::~Advance_East_DLL_Export_H()
{
}

int Advance_East_DLL_Export_H::LoadModel(const char* model_path)
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

int Advance_East_DLL_Export_H::TextDetect(const char* img_path, float dThreshold, vector<vector<double>>& TextPos)
{
	int nResult = 0;
	try
	{
		nResult = m_pGCIAlgSynCheck->TextDetect(img_path, dThreshold, TextPos);
	}
	catch (...)
	{
		nResult = 1;
	}
	return nResult;
}

