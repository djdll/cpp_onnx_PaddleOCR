#include"text_det.h"

TextDetector::TextDetector()
	: binaryThreshold(0.3),
	polygonThreshold(0.5),
	unclipRatio(1.6),
	maxCandidates(1000)
{
	const std::string model_path = "C:\\Users\\DELL\\Desktop\\c++_onnx_PaddleOCR\\onnx_model/en_det_model.onnx";

	try {
		// 创建宽字符字符串
		std::wstring widestr(model_path.begin(), model_path.end());
		sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

		// 使用智能指针管理内存
		net = std::make_unique<Ort::Session>(env, widestr.c_str(), sessionOptions);

		size_t numInputNodes = net->GetInputCount();
		size_t numOutputNodes = net->GetOutputCount();
		Ort::AllocatorWithDefaultOptions allocator;

		// 预分配空间提高性能
		input_names.reserve(numInputNodes);
		output_names.reserve(numOutputNodes);

		for (size_t i = 0; i < numInputNodes; i++)
		{
			input_names.push_back(net->GetInputName(i, allocator));
		}

		for (size_t i = 0; i < numOutputNodes; i++)
		{
			output_names.push_back(net->GetOutputName(i, allocator));
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Error initializing TextDetector: " << e.what() << std::endl;
		throw;  // 继续抛出异常以便外部捕获处理
	}
}

Mat TextDetector::preprocess(Mat srcimg)
{
	Mat dstimg;
	cvtColor(srcimg, dstimg, COLOR_BGR2RGB);
	int h = srcimg.rows;
	int w = srcimg.cols;
	float scale_h = 1;
	float scale_w = 1;
	if (h < w)
	{
		scale_h = (float)this->short_size / (float)h;
		float tar_w = (float)w * scale_h;
		tar_w = tar_w - (int)tar_w % 32;
		tar_w = max((float)32, tar_w);
		scale_w = tar_w / (float)w;
	}
	else
	{
		scale_w = (float)this->short_size / (float)w;
		float tar_h = (float)h * scale_w;
		tar_h = tar_h - (int)tar_h % 32;
		tar_h = max((float)32, tar_h);
		scale_h = tar_h / (float)h;
	}
	resize(dstimg, dstimg, Size(int(scale_w * dstimg.cols), int(scale_h * dstimg.rows)), INTER_LINEAR);
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
            for (int j = 0; j < col; j++)
            {
                int pixel_index = j * 3 + c;
                float pix = img_ptr[pixel_index];
                this->input_image_[c * row * col + i * col + j] = (pix * norm_factor) - 1.0;
            }

        }
    }
}
vector<vector<Point2f>> TextDetector::detect(Mat& srcimg)
{
    int h = srcimg.rows;
    int w = srcimg.cols;
    Mat dstimg = this->preprocess(srcimg);
    this->normalize_(dstimg);
    array<int64_t, 4> input_shape_{1, 3, dstimg.rows, dstimg.cols};

    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

    vector<Value> ort_outputs = net->Run(RunOptions{nullptr}, input_names.data(), &input_tensor_, 1, output_names.data(), output_names.size());

    const float* floatArray = ort_outputs[0].GetTensorMutableData<float>();
    vector<int64_t> output_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int outputCount = 1;
    for (int dim : output_shape) {
        outputCount *= dim;
    }

    Mat binary(dstimg.rows, dstimg.cols, CV_32FC1, const_cast<float*>(floatArray));

    // Threshold
    Mat bitmap;
    threshold(binary, bitmap, binaryThreshold, 255, THRESH_BINARY);
    // Scale ratio
    float scaleHeight = static_cast<float>(h) / static_cast<float>(binary.size[0]);
    float scaleWidth = static_cast<float>(w) / static_cast<float>(binary.size[1]);
    // Find contours
    vector<vector<Point>> contours;
    bitmap.convertTo(bitmap, CV_8UC1);
    findContours(bitmap, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // Candidate number limitation
    size_t numCandidate = min(contours.size(), static_cast<size_t>(maxCandidates > 0 ? maxCandidates : INT_MAX));
    vector<float> confidences(contours.size(), 1.0f);
    vector<vector<Point2f>> results;
    results.reserve(numCandidate);

    for (size_t i = 0; i < numCandidate; i++) {
        vector<Point>& contour = contours[i];

        // Calculate text contour score
        if (contourScore(binary, contour) < polygonThreshold)
            continue;

        // Rescale
        vector<Point> contourScaled;
        contourScaled.reserve(contour.size());
        for (const Point& pt : contour) {
            contourScaled.emplace_back(Point(static_cast<int>(pt.x * scaleWidth), static_cast<int>(pt.y * scaleHeight)));
        }

        // Unclip
        RotatedRect box = minAreaRect(contourScaled);
        float longSide = max(box.size.width, box.size.height);
        if (longSide < longSideThresh) {
            continue;
        }

        // Adjust for expected horizontal text
        if (box.size.width < box.size.height || fabs(box.angle) >= 60) {
            swap(box.size.width, box.size.height);
            box.angle = box.angle < 0 ? box.angle + 90 : box.angle - 90;
        }

        Point2f vertex[4];
        box.points(vertex);
        vector<Point2f> approx(vertex, vertex + 4);
        vector<Point2f> polygon;
        unclip(approx, polygon);

        box = minAreaRect(polygon);
        longSide = max(box.size.width, box.size.height);
        if (longSide < longSideThresh + 2) {
            continue;
        }

        results.push_back(polygon);
    }

    return results;
}

vector<vector<Point2f>> TextDetector::order_points_clockwise(vector<vector<Point2f>> results)
{
	vector<vector<Point2f>> ordered_points(results.size(), vector<Point2f>(4));

	for (size_t i = 0; i < results.size(); i++)
	{
		const auto& quadrilateral = results[i];

		// Initialize extreme values
		float max_sum = numeric_limits<float>::lowest();
		float min_sum = numeric_limits<float>::max();
		float max_diff = numeric_limits<float>::lowest();
		float min_diff = numeric_limits<float>::max();

		size_t top_left = 0, bottom_right = 0, top_right = 0, bottom_left = 0;

		for (size_t j = 0; j < quadrilateral.size(); j++)
		{
			const Point2f& point = quadrilateral[j];
			float sum = point.x + point.y;
			float diff = point.y - point.x;

			if (sum > max_sum)
			{
				max_sum = sum;
				bottom_right = j;
			}
			if (sum < min_sum)
			{
				min_sum = sum;
				top_left = j;
			}

			if (diff > max_diff)
			{
				max_diff = diff;
				top_right = j;
			}
			if (diff < min_diff)
			{
				min_diff = diff;
				bottom_left = j;
			}
		}

		// Assign points in clockwise order: top-left, top-right, bottom-right, bottom-left
		ordered_points[i][0] = quadrilateral[top_left];
		ordered_points[i][1] = quadrilateral[top_right];
		ordered_points[i][2] = quadrilateral[bottom_right];
		ordered_points[i][3] = quadrilateral[bottom_left];
	}

	return ordered_points;
}

void TextDetector::draw_pred(Mat& srcimg, const vector<vector<Point2f>>& results)
{
	const Scalar circle_color(0, 0, 255); // Red
	const Scalar line_color(0, 255, 0);   // Green
	const int radius = 2;
	const int thickness = -1;

	for (const auto& polygon : results)
	{
		for (size_t j = 0; j < polygon.size(); j++)
		{
			Point pt1 = Point(static_cast<int>(polygon[j].x), static_cast<int>(polygon[j].y));
			circle(srcimg, pt1, radius, circle_color, thickness);

			Point pt2 = Point(static_cast<int>(polygon[(j + 1) % polygon.size()].x),
				static_cast<int>(polygon[(j + 1) % polygon.size()].y));
			line(srcimg, pt1, pt2, line_color);
		}
	}
}

float TextDetector::contourScore(const Mat& binary, const vector<Point>& contour)
{
	Rect rect = boundingRect(contour);
	int xmin = max(rect.x, 0);
	int xmax = min(rect.x + rect.width, binary.cols - 1);
	int ymin = max(rect.y, 0);
	int ymax = min(rect.y + rect.height, binary.rows - 1);

	Mat binROI = binary(Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));

	Mat mask = Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8U);
	vector<Point> roiContour;
	for (size_t i = 0; i < contour.size(); i++) {
		Point pt = Point(contour[i].x - xmin, contour[i].y - ymin);
		roiContour.push_back(pt);
	}
	vector<vector<Point>> roiContours = { roiContour };
	fillPoly(mask, roiContours, Scalar(1));
	float score = mean(binROI, mask).val[0];
	return score;
}

void TextDetector::unclip(const vector<Point2f>& inPoly, vector<Point2f>& outPoly)
{
	// 计算多边形面积和周长
	float area = contourArea(inPoly);
	float length = arcLength(inPoly, true);
	float distance = area * unclipRatio / length;

	size_t numPoints = inPoly.size();
	vector<vector<Point2f>> newLines;
	newLines.reserve(numPoints);

	// 创建新边
	for (size_t i = 0; i < numPoints; i++)
	{
		vector<Point2f> newLine;
		Point2f pt1 = inPoly[i];
		Point2f pt2 = inPoly[(i + numPoints - 1) % numPoints];
		Point2f vec = pt1 - pt2;
		float normVec = norm(vec);
		Point2f rotateVec(vec.y * distance / normVec, -vec.x * distance / normVec);

		newLine.push_back(pt1 + rotateVec);
		newLine.push_back(pt2 + rotateVec);
		newLines.push_back(newLine);
	}

	outPoly.reserve(newLines.size());

	// 计算新的顶点
	for (size_t i = 0; i < newLines.size(); i++)
	{
		Point2f a = newLines[i][0];
		Point2f b = newLines[i][1];
		Point2f c = newLines[(i + 1) % newLines.size()][0];
		Point2f d = newLines[(i + 1) % newLines.size()][1];
		Point2f pt;

		Point2f v1 = b - a;
		Point2f v2 = d - c;

		float cosAngle = v1.dot(v2);

		if (fabs(cosAngle) > 0.7)
		{
			pt = (b + c) * 0.5f;
		}
		else
		{
			float denom = a.x * (d.y - c.y) + b.x * (c.y - d.y) + d.x * (b.y - a.y) + c.x * (a.y - b.y);
			float num = a.x * (d.y - c.y) + c.x * (a.y - d.y) + d.x * (c.y - a.y);
			float s = num / denom;

			pt = a + s * (b - a);
		}
		outPoly.push_back(pt);
	}
}

Mat TextDetector::get_rotate_crop_image(const Mat& frame, vector<Point2f> vertices)
{
	Rect rect = boundingRect(Mat(vertices));
	if (rect.x < 0) rect.x = 0;
	if (rect.x + rect.width > frame.cols) rect.x = frame.cols - rect.width;
	if (rect.y < 0) rect.y = 0;
	if (rect.y + rect.height > frame.rows) rect.y = frame.rows - rect.height;
	Mat crop_img = frame(rect);

	const Size outputSize = Size(rect.width, rect.height);

	vector<Point2f> targetVertices{ Point2f(0, outputSize.height),Point2f(0, 0), Point2f(outputSize.width, 0), Point2f(outputSize.width, outputSize.height) };
	
	for (int i = 0; i < 4; i++)
	{
		vertices[i].x -= rect.x;
		vertices[i].y -= rect.y;
	}

	Mat rotationMatrix = getPerspectiveTransform(vertices, targetVertices);
	Mat result;
	warpPerspective(crop_img, result, rotationMatrix, outputSize, cv::BORDER_REPLICATE);
	return result;
}