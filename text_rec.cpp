#include"text_rec.h"

TextRecognizer::TextRecognizer()
{
    const std::string model_path = "C:\\Users\\DELL\\Desktop\\c++_onnx_PaddleOCR\\onnx_model/en_rec_model.onnx";
    const std::string dict_path = "D:/OCR/code/PaddleOCR-cpp-main/paddleOCR/en_dict.txt";

    try {
        // 创建宽字符字符串
        std::wstring widestr(model_path.begin(), model_path.end());
        sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

        // 使用智能指针管理内存
        ort_session = std::make_unique<Ort::Session>(env, widestr.c_str(), sessionOptions);

        size_t numInputNodes = ort_session->GetInputCount();
        size_t numOutputNodes = ort_session->GetOutputCount();
        Ort::AllocatorWithDefaultOptions allocator;

        // 预分配空间提高性能
        input_names.reserve(numInputNodes);
        input_node_dims.reserve(numInputNodes);
        output_names.reserve(numOutputNodes);
        output_node_dims.reserve(numOutputNodes);

        for (size_t i = 0; i < numInputNodes; i++)
        {
            input_names.push_back(ort_session->GetInputName(i, allocator));
            Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_dims = input_tensor_info.GetShape();
            input_node_dims.push_back(input_dims);
        }

        for (size_t i = 0; i < numOutputNodes; i++)
        {
            output_names.push_back(ort_session->GetOutputName(i, allocator));
            Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            auto output_dims = output_tensor_info.GetShape();
            output_node_dims.push_back(output_dims);
        }

        // 读取字典文件
        std::ifstream ifs(dict_path);
        if (!ifs.is_open()) {
            throw std::runtime_error("Cannot open dictionary file: " + dict_path);
        }

        std::string line;
        while (std::getline(ifs, line))
        {
            this->alphabet.push_back(line);
        }
        this->alphabet.push_back(" ");
        names_len = this->alphabet.size();
    } catch (const std::exception& e) {
        std::cerr << "Error initializing TextRecognizer: " << e.what() << std::endl;
        throw;  // 继续抛出异常以便外部捕获处理
    }
}
Mat TextRecognizer::preprocess(Mat srcimg)
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

std::string TextRecognizer::predict_text(Mat cv_image)
{
    Mat preprocessed_image = this->preprocess(cv_image);
    this->normalize_(preprocessed_image);

    std::array<int64_t, 4> input_shape{1, 3, this->inpHeight, this->inpWidth};
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        allocator_info, input_image_.data(), input_image_.size(), input_shape.data(), input_shape.size());

    try {
        // 执行模型推理
        std::vector<Ort::Value> ort_outputs = ort_session->Run(
            Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());

        const float* pdata = ort_outputs[0].GetTensorMutableData<float>();

        int width = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape().at(1);
        int height = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape().at(2);

        preb_label.resize(width);

        // 获取最大概率对应的标签
        for (int x = 0; x < width; x++)
        {
            int best_label_index = 0;
            float max_score = std::numeric_limits<float>::lowest();
            for (int y = 0; y < height; y++)
            {
                float score = pdata[x * height + y];
                if (score > max_score)
                {
                    max_score = score;
                    best_label_index = y;
                }
            }
            preb_label[x] = best_label_index;
        }

        std::vector<int> no_repeat_blank_label;
        no_repeat_blank_label.reserve(width);

        for (size_t index = 0; index < width; ++index)
        {
            if (preb_label[index] != 0 && !(index > 0 && preb_label[index - 1] == preb_label[index]))
            {
                no_repeat_blank_label.push_back(preb_label[index] - 1);
            }
        }

        std::string result_text;
        for (int label : no_repeat_blank_label)
        {
            result_text += alphabet[label];
        }

        return result_text;

    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return "";  // 返回空字符串表示出错
    }
}
