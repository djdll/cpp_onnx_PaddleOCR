﻿#include"text_angle_cls.h"

TextClassifier::TextClassifier()
{
	string model_path = "C:\\Users\\DELL\\Desktop\\c++_onnx_PaddleOCR\\onnx_model/cls_large.onnx";
	std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	ort_session = new Session(env, widestr.c_str(), sessionOptions);
	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	num_out = output_node_dims[0][1];
}

Mat TextClassifier::preprocess(Mat srcimg)
{
	Mat dstimg;
	int h = srcimg.rows;
	int w = srcimg.cols;
	const float ratio = w / float(h);
	int resized_w = int(ceil((float)this->inpHeight * ratio));
	if (ceil(this->inpHeight * ratio) > this->inpWidth)
	{
		resized_w = this->inpWidth;
	}

	resize(srcimg, dstimg, Size(resized_w, this->inpHeight), INTER_LINEAR);
	return dstimg;
}
void TextClassifier::normalize_(Mat img)
{
    int row = img.rows;
    int col = img.cols;
    int channels = img.channels();
    
    // 预计算归一化因子
    const float norm_factor = 1.0 / 127.5;
    
    this->input_image_.resize(this->inpHeight * this->inpWidth * channels);

    for (int c = 0; c < channels; c++)
    {
        for (int i = 0; i < row; i++)
        {
            // 处理 j < col 的部分
            const uchar* img_ptr = img.ptr<uchar>(i);
            int base_index = c * row * inpWidth + i * inpWidth;
            
            for (int j = 0; j < col; j++)
            {
                int pixel_index = j * 3 + c;
                float pix = img_ptr[pixel_index];
                this->input_image_[base_index + j] = (pix * norm_factor) - 1.0;
            }

            // 处理 j >= col 的部分
            for (int j = col; j < inpWidth; j++)
            {
                this->input_image_[base_index + j] = 0;
            }
        }
    }
}


int TextClassifier::predict(Mat cv_image)
{
	Mat dstimg = this->preprocess(cv_image);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };
	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // ��ʼ����
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();

	int max_id = 0;
	float max_prob = -1;
	for (int i = 0; i < num_out; i++)
	{
		if (pdata[i] > max_prob)
		{
			max_prob = pdata[i];
			max_id = i;
		}
	}

	return max_id;
}