// paddleocr.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include "text_det.h"
#include "text_angle_cls.h"
#include "text_rec.h"

using namespace cv;
using namespace std;
using namespace Ort;
int main()
{
	TextDetector detect_model;
	TextClassifier angle_model;
	TextRecognizer rec_model;
	string imgpath = "C:\\Users\\DELL\\Desktop\\2.jpg";
    Mat srcimg = imread("C:\\Users\\DELL\\Desktop\\2.jpg");
    if (srcimg.empty()) {
        cout << "could not load image..." << endl;
        return -1;
    }

	// test hole image
	vector< vector<Point2f> > results = detect_model.detect(srcimg);
	for (size_t i = 0; i < results.size(); i++)
	{
		Mat textimg = detect_model.get_rotate_crop_image(srcimg, results[i]);
		if (angle_model.predict(textimg) == 1)
		{
			cv::rotate(textimg, textimg, 1);
		}

		string text = rec_model.predict_text(textimg);
		cout << text << endl;
	}
	detect_model.draw_pred(srcimg, results);
	namedWindow("PaddleOCR", WINDOW_NORMAL);
	resizeWindow("PaddleOCR", 512, 512);
	imshow("PaddleOCR", srcimg);
	waitKey(0);
	destroyAllWindows();

}


