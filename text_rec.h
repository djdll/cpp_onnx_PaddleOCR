#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

class TextRecognizer
{
public:
	TextRecognizer();
	string predict_text(Mat cv_image);

private:
	Mat preprocess(Mat srcimg);
	void normalize_(Mat img);
	const int inpWidth = 560;
	const int inpHeight = 48;

	vector<float> input_image_;
	vector<string> alphabet;
	int names_len;
	vector<int> preb_label;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "CRNN");
	std::unique_ptr<Ort::Session> ort_session;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; 
	vector<vector<int64_t>> output_node_dims; 
};