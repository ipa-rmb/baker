#include "dirt_detection_standalone_spectral.h"

#include <boost/algorithm/string.hpp>


IpaDirtDetection::ServerDirtDetection::ServerDirtDetection()
{
	//PARAMS
	std::cout << "\n========== dirt_detection_standalone_spectral parameters ==========\n";

	spectral_residual_gaussian_blur_iterations_ = 3;
	std::cout << "ServerDirtDetection: spectral_residual_gaussian_blur_iterations: " << spectral_residual_gaussian_blur_iterations_ << std::endl;
	spectral_residual_image_size_ratio_ = 0.25;
	std::cout << "ServerDirtDetection: spectral_residual_image_size_ratio = " << spectral_residual_image_size_ratio_ << std::endl;
	spectral_residual_normalization_highest_max_value_ = 1500.;
	std::cout << "ServerDirtDetection: spectral_residual_normalization_highest_max_value = " << spectral_residual_normalization_highest_max_value_ << std::endl;
	remove_lines_ = false;
	std::cout << "ServerDirtDetection: remove_lines = " << remove_lines_ << std::endl;
	dirt_threshold_ = 0.3;
	std::cout << "ServerDirtDetection: dirt_threshold = " << dirt_threshold_ << std::endl;
	dirt_check_std_dev_factor_ = 3.0;
	std::cout << "ServerDirtDetection: dirt_check_std_dev_factor = " << dirt_check_std_dev_factor_ << std::endl;

	//DEBUG PARAMS
	debug_["show_warped_original_image"] = false;
	std::cout << "DEBUG_ServerDirtDetection: show_warped_original_image = " << debug_["show_warped_original_image"] << std::endl;
	debug_["show_dirt_detections"] = false;
	std::cout << "DEBUG_ServerDirtDetection: show_dirt_detections = " << debug_["show_dirt_detections"] << std::endl;
	debug_["save_data_for_test"] = false;
	std::cout << "DEBUG_ServerDirtDetection: save_data_for_test = " << debug_["save_data_for_test"] << std::endl;
	debug_["show_color_with_artificial_dirt"] = false;
	std::cout << "DEBUG_ServerDirtDetection: show_color_with_artificial_dirt = " << debug_["show_color_with_artificial_dirt"] << std::endl;
	debug_["show_saliency_with_artificial_dirt"] = false;
	std::cout << "DEBUG_ServerDirtDetection: show_saliency_with_artificial_dirt = " << debug_["show_saliency_with_artificial_dirt"] << std::endl;
	debug_["show_saliency_bad_scale"] = false;
	std::cout << "DEBUG_ServerDirtDetection: show_saliency_bad_scale = " << debug_["show_saliency_bad_scale"] << std::endl;
	debug_["show_detected_lines"] = false;
	std::cout << "DEBUG_ServerDirtDetection: show_detected_lines = " << debug_["show_detected_lines"] << std::endl;
	debug_["show_saliency_detection"] = false;
	std::cout << "DEBUG_ServerDirtDetection: show_saliency_detection = " << debug_["show_saliency_detection"] << std::endl;
}

IpaDirtDetection::ServerDirtDetection::~ServerDirtDetection()
{	}


//###############################################################################################################
//======================== Function declaration =================================================================
//===============================================================================================================

void IpaDirtDetection::ServerDirtDetection::detectDirt(const cv::Mat& C3_color_image, const cv::Mat& mask, cv::Mat& scaled_C1_saliency_image)
{
	std::cout << "Dirt detection started." << std::endl;

	// detect dirt on the floor
	cv::Mat C1_saliency_image;
	SaliencyDetection_C3(C3_color_image, C1_saliency_image, &mask, spectral_residual_gaussian_blur_iterations_);

	// post processing, dirt/stain selection
	cv::Mat dirt_detections_mask;
	cv::Mat new_C3_color_image = C3_color_image.clone();
	std::vector<cv::RotatedRect> dirt_detections;
	Image_Postprocessing_C1_rmb(C1_saliency_image, scaled_C1_saliency_image, new_C3_color_image, dirt_detections, dirt_detections_mask, mask);

	if (debug_["show_warped_original_image"] == true)
	{
		cv::imshow("warped original image", C3_color_image);
		cv::waitKey(10);
	}

	if (debug_["show_dirt_detections"] == true)
	{
		cv::imshow("dirt detections", new_C3_color_image);
		cvMoveWindow("dirt detections", 650, 530);
		cv::waitKey(10);
	}

	std::cout << "-----------------------------------------------------\nAction done. " << std::endl;
}

void IpaDirtDetection::ServerDirtDetection::SaliencyDetection_C3(const cv::Mat& C3_color_image, cv::Mat& C1_saliency_image, const cv::Mat* mask, int gaussianBlurCycles)
{
	cv::Mat fci; // "fci"<-> first channel image
	cv::Mat sci; // "sci"<-> second channel image
	cv::Mat tci; //"tci"<-> second channel image

	std::vector<cv::Mat> vec;
	vec.push_back(fci);
	vec.push_back(sci);
	vec.push_back(tci);

	cv::split(C3_color_image, vec);

	fci = vec[0];
	sci = vec[1];
	tci = vec[2];

	cv::Mat res_fci; // "fci"<-> first channel image
	cv::Mat res_sci; // "sci"<-> second channel image
	cv::Mat res_tci; //"tci"<-> second channel image

	SaliencyDetection_C1(fci, res_fci);
	SaliencyDetection_C1(sci, res_sci);
	SaliencyDetection_C1(tci, res_tci);

	cv::Mat realInput;

	realInput = (res_fci + res_sci + res_tci) / 3;
//	std::cout << "		The fft output size after 1 channel is: " << realInput.cols << " " << realInput.rows << std::endl;

	cv::Size2i ksize;
	ksize.width = 3;
	ksize.height = 3;
	for (int i = 0; i < gaussianBlurCycles; i++)
		cv::GaussianBlur(realInput, realInput, ksize, 0); //necessary!? --> less noise

	cv::resize(realInput, C1_saliency_image, C3_color_image.size());
//	std::cout << "		The fft output size after resize is: " << C1_saliency_image.cols << " " << C1_saliency_image.rows << std::endl;

	// remove borders of the ground plane because of artifacts at the border like lines
	if (mask != 0)
	{
		// maske erodiere
		cv::Mat mask_eroded = mask->clone();
		cv::dilate(*mask, mask_eroded, cv::Mat(), cv::Point(-1, -1), 2);
		cv::erode(mask_eroded, mask_eroded, cv::Mat(), cv::Point(-1, -1), 25.0 / 640.0 * C3_color_image.cols);
		cv::Mat neighborhood = cv::Mat::ones(3, 1, CV_8UC1);
		cv::erode(mask_eroded, mask_eroded, neighborhood, cv::Point(-1, -1), 15.0 / 640.0 * C3_color_image.cols);

		//cv::erode(mask_eroded, mask_eroded, cv::Mat(), cv::Point(-1, -1), 35.0/640.0*C3_color_image.cols);
		// todo: hack for autonomik
		//cv::erode(mask_eroded, mask_eroded, cv::Mat(), cv::Point(-1, -1), 45.0/640.0*C3_color_image.cols);

		cv::Mat temp;
		C1_saliency_image.copyTo(temp, mask_eroded);
		C1_saliency_image = temp;
	}

	// remove saliency at the image border (because of artifacts in the corners)
	int borderX = C1_saliency_image.cols / 20;
	int borderY = C1_saliency_image.rows / 20;
	if (borderX > 0 && borderY > 0)
	{
		cv::Mat smallImage_ = C1_saliency_image.colRange(borderX, C1_saliency_image.cols - borderX);
		cv::Mat smallImage = smallImage_.rowRange(borderY, smallImage_.rows - borderY);

		copyMakeBorder(smallImage, C1_saliency_image, borderY, borderY, borderX, borderX, cv::BORDER_CONSTANT, cv::Scalar(0));
	}

	// display the individual channels
//	for (int i=0; i<gaussianBlurCycles; i++)
//		cv::GaussianBlur(res_fci, res_fci, ksize, 0); //necessary!? --> less noise
//	for (int i=0; i<gaussianBlurCycles; i++)
//		cv::GaussianBlur(res_sci, res_sci, ksize, 0); //necessary!? --> less noise	//calculate number of test samples
//	int NumTestSamples = 10; //ceil(NumSamples*percentage_testdata);
//	printf("Anzahl zu ziehender test samples: %d \n", NumTestSamples);
//	for (int i=0; i<gaussianBlurCycles; i++)
//		cv::GaussianBlur(res_tci, res_tci, ksize, 0); //necessary!? --> less noise
//
//	cv::resize(res_fci,res_fci,C3_color_image.size());
//	cv::resize(res_sci,res_sci,C3_color_image.size());
//	cv::resize(res_tci,res_tci,C3_color_image.size());
//
//	// scale input_image
//	double minv, maxv;
//	cv::Point2i minl, maxl;
//	cv::minMaxLoc(res_fci,&minv,&maxv,&minl,&maxl);
//	res_fci.convertTo(res_fci, -1, 1.0/(maxv-minv), 1.0*(minv)/(maxv-minv));
//	cv::minMaxLoc(res_sci,&minv,&maxv,&minl,&maxl);
//	res_sci.convertTo(res_sci, -1, 1.0/(maxv-minv), 1.0*(minv)/(maxv-minv));
//	cv::minMaxLoc(res_tci,&minv,&maxv,&minl,&maxl);
//	res_tci.convertTo(res_tci, -1, 1.0/(maxv-minv), 1.0*(minv)/(maxv-minv));
//
//	cv::imshow("b", res_fci);
//	cv::imshow("g", res_sci);
//	cv::imshow("r", res_tci);
}

void IpaDirtDetection::ServerDirtDetection::SaliencyDetection_C1(const cv::Mat& C1_image, cv::Mat& C1_saliency_image)
{
	//given a one channel image
	//int scale = 6;
	//unsigned int size = (int)floor((float)pow(2.0,scale)); //the size to do the saliency at
	unsigned int size_cols = (int) (C1_image.cols * spectral_residual_image_size_ratio_);
	unsigned int size_rows = (int) (C1_image.rows * spectral_residual_image_size_ratio_);

	//create different images
	cv::Mat bw_im;
	cv::resize(C1_image, bw_im, cv::Size(size_cols, size_rows));

	cv::Mat realInput(size_rows, size_cols, CV_32FC1); //calculate number of test samples
	//int NumTestSamples = 10; //ceil(NumSamples*percentage_testdata);
	//printf("Anzahl zu ziehender test samples: %d \n", NumTestSamples);
	cv::Mat imaginaryInput(size_rows, size_cols, CV_32FC1);
	cv::Mat complexInput(size_rows, size_cols, CV_32FC2);

	bw_im.convertTo(realInput, CV_32F, 1.0 / 255, 0);
	imaginaryInput = cv::Mat::zeros(size_rows, size_cols, CV_32F);

	std::vector<cv::Mat> vec;
	vec.push_back(realInput);
	vec.push_back(imaginaryInput);
	cv::merge(vec, complexInput);

	cv::Mat dft_A(size_rows, size_cols, CV_32FC2);

	cv::dft(complexInput, dft_A, cv::DFT_COMPLEX_OUTPUT, size_rows);
	vec.clear();
	cv::split(dft_A, vec);
	realInput = vec[0];
	imaginaryInput = vec[1];

	// Compute the phase angle
	cv::Mat image_Mag(size_rows, size_cols, CV_32FC1);
	cv::Mat image_Phase(size_rows, size_cols, CV_32FC1);

	//compute the phase of the spectrum
	cv::cartToPolar(realInput, imaginaryInput, image_Mag, image_Phase, 0);
//	std::vector<DirtDetection::CarpetFeatures> test_feat_vec;
//	std::vector<DirtDetection::CarpetClass> test_class_vec;

	cv::Mat log_mag(size_rows, size_cols, CV_32FC1);
	cv::log(image_Mag, log_mag);

//	cv::Mat log_mag_;
//	cv::normalize(log_mag, log_mag_, 0, 1, NORM_MINMAX);
//	cv::imshow("log_mag", log_mag_);

	//Box filter the magnitude, then take the difference
	cv::Mat log_mag_Filt(size_rows, size_cols, CV_32FC1);

	cv::Mat filt = cv::Mat::ones(3, 3, CV_32FC1) * 1. / 9.;
	//filt.convertTo(filt,-1,1.0/9.0,0);

	cv::filter2D(log_mag, log_mag_Filt, -1, filt);
	//cv::GaussianBlur(log_mag, log_mag_Filt, cv::Size2i(25,25), 0);
	//log_mag = log_mag_Filt.clone();
	//cv::medianBlur(log_mag, log_mag_Filt, 5);

//	cv::Mat log_mag_filt_;
//	cv::normalize(log_mag_Filt, log_mag_filt_, 0, 1, NORM_MINMAX);
//	cv::imshow("log_mag_filt", log_mag_filt_);

	//cv::subtract(log_mag, log_mag_Filt, log_mag);
	log_mag -= log_mag_Filt;

//	cv::Mat log_mag_sub_a_, log_mag_sub_;
//	cv::normalize(log_mag, log_mag_sub_a_, 0, 1, NORM_MINMAX);
//	cv::GaussianBlur(log_mag_sub_a_, log_mag_sub_, cv::Size2i(21,21), 0);
//	cv::imshow("log_mag_sub", log_mag_sub_);
//	log_mag_Filt = log_mag.clone();
//	cv::GaussianBlur(log_mag_Filt, log_mag, cv::Size2i(21,21), 0);
//	void GBTreeEvaluation(	std::vector<CarpetFeatures>& train_feat_vec, std::vector<CarpetClass>& train_class_vec,
//						std::vector<CarpetFeatures>& test_feat_vec, std::vector<CarpetClass>& test_class_vec,
//						CvGBTrees &carpet_GBTree);
	cv::exp(log_mag, image_Mag);

	cv::polarToCart(image_Mag, image_Phase, realInput, imaginaryInput, 0);

	vec.clear();
	vec.push_back(realInput);
	vec.push_back(imaginaryInput);
	cv::merge(vec, dft_A);

	cv::dft(dft_A, dft_A, cv::DFT_INVERSE, size_rows);

	dft_A = abs(dft_A);
	dft_A.mul(dft_A);

	cv::split(dft_A, vec);

	C1_saliency_image = vec[0];
//	std::cout << "		The size of the output of 1 channel fft is: " << C1_saliency_image.cols << " " << C1_saliency_image.rows << " "
//			<< C1_saliency_image.channels() << std::endl;
}

void IpaDirtDetection::ServerDirtDetection::Image_Postprocessing_C1_rmb(const cv::Mat& C1_saliency_image, cv::Mat& scaled_C1_saliency_image, cv::Mat& C3_color_image,
		std::vector<cv::RotatedRect>& dirtDetections, cv::Mat& dirt_detections_mask, const cv::Mat& mask)
{
	// dirt detection on image with artificial dirt
	cv::Mat color_image_with_artifical_dirt = C3_color_image.clone();
	cv::Mat mask_with_artificial_dirt = mask.clone();
	// add dirt
	int dirtSize = cvRound(3.0 / 640.0 * C3_color_image.cols);
	cv::Point2f ul(0.4375 * C3_color_image.cols, 0.416666667 * C3_color_image.rows);
	cv::Point2f ur(0.5625 * C3_color_image.cols, 0.416666667 * C3_color_image.rows);
	cv::Point2f ll(0.4375 * C3_color_image.cols, 0.583333333 * C3_color_image.rows);
	cv::Point2f lr(0.5625 * C3_color_image.cols, 0.583333333 * C3_color_image.rows);
	cv::ellipse(color_image_with_artifical_dirt, cv::RotatedRect(ul, cv::Size2f(dirtSize, dirtSize), 0), cv::Scalar(255, 255, 255), dirtSize);
	cv::ellipse(mask_with_artificial_dirt, cv::RotatedRect(ul, cv::Size2f(dirtSize, dirtSize), 0), cv::Scalar(255, 255, 255), dirtSize);
	cv::ellipse(color_image_with_artifical_dirt, cv::RotatedRect(lr, cv::Size2f(dirtSize, dirtSize), 0), cv::Scalar(255, 255, 255), dirtSize);
	cv::ellipse(mask_with_artificial_dirt, cv::RotatedRect(lr, cv::Size2f(dirtSize, dirtSize), 0), cv::Scalar(255, 255, 255), dirtSize);
	cv::ellipse(color_image_with_artifical_dirt, cv::RotatedRect(ll, cv::Size2f(dirtSize, dirtSize), 0), cv::Scalar(0, 0, 0), dirtSize);
	cv::ellipse(mask_with_artificial_dirt, cv::RotatedRect(ll, cv::Size2f(dirtSize, dirtSize), 0), cv::Scalar(255, 255, 255), dirtSize);
	cv::ellipse(color_image_with_artifical_dirt, cv::RotatedRect(ur, cv::Size2f(dirtSize, dirtSize), 0), cv::Scalar(0, 0, 0), dirtSize);
	cv::ellipse(mask_with_artificial_dirt, cv::RotatedRect(ur, cv::Size2f(dirtSize, dirtSize), 0), cv::Scalar(255, 255, 255), dirtSize);
	cv::Mat C1_saliency_image_with_artifical_dirt;
	SaliencyDetection_C3(color_image_with_artifical_dirt, C1_saliency_image_with_artifical_dirt, &mask_with_artificial_dirt,
			spectral_residual_gaussian_blur_iterations_);
	//cv::imshow("ai_dirt", color_image_with_artifical_dirt);

	// display of images with artificial dirt
	if (debug_["show_color_with_artificial_dirt"] == true)
		cv::imshow("color with artificial dirt", color_image_with_artifical_dirt);
    if (debug_["show_saliency_with_artificial_dirt"] == true) {
        cv::Mat C1_saliency_image_with_artifical_dirt_scaled;
        double salminv, salmaxv;
        cv::Point2i salminl, salmaxl;
        cv::minMaxLoc(C1_saliency_image_with_artifical_dirt, &salminv, &salmaxv, &salminl, &salmaxl, mask_with_artificial_dirt);
        C1_saliency_image_with_artifical_dirt.convertTo(C1_saliency_image_with_artifical_dirt_scaled, -1,
                                                        1.0 / (salmaxv - salminv),
                                                        -1.0 * (salminv) / (salmaxv - salminv));
        cv::imshow("saliency with artificial dirt", C1_saliency_image_with_artifical_dirt_scaled);
    }

	// scale C1_saliency_image to value obtained from C1_saliency_image with artificially added dirt (range [0,1])
//	std::cout << "res_img: " << C1_saliency_image_with_artifical_dirt.at<float>(300,200);
//	C1_saliency_image_with_artifical_dirt = C1_saliency_image_with_artifical_dirt.mul(C1_saliency_image_with_artifical_dirt);	// square C1_saliency_image_with_artifical_dirt to emphasize the dirt and increase the gap to background response
//	std::cout << " -> " << C1_saliency_image_with_artifical_dirt.at<float>(300,200) << std::endl;
	double minv, maxv;
	cv::Point2i minl, maxl;
	cv::minMaxLoc(C1_saliency_image_with_artifical_dirt, &minv, &maxv, &minl, &maxl, mask_with_artificial_dirt);
	cv::Scalar mean, stdDev;
	cv::meanStdDev(C1_saliency_image_with_artifical_dirt, mean, stdDev, mask);
	double newMaxVal = std::min(1.0, maxv / spectral_residual_normalization_highest_max_value_); ///mean.val[0] / spectralResidualNormalizationHighestMaxMeanRatio_);
//	std::cout << "dirtThreshold=" << dirtThreshold_ << "\tmin=" << minv << "\tmax=" << maxv << "\tmean=" << mean.val[0] << "\tstddev=" << stdDev.val[0] << "\tnewMaxVal (r)=" << newMaxVal << std::endl;

	////C1_saliency_image.convertTo(scaled_input_image, -1, 1.0/(maxv-minv), 1.0*(minv)/(maxv-minv));
	scaled_C1_saliency_image = C1_saliency_image.clone(); // square C1_saliency_image_with_artifical_dirt to emphasize the dirt and increase the gap to background response
	//scaled_C1_saliency_image = scaled_C1_saliency_image.mul(scaled_C1_saliency_image);
	scaled_C1_saliency_image.convertTo(scaled_C1_saliency_image, -1, newMaxVal / (maxv - minv), -newMaxVal * (minv) / (maxv - minv));

	double newMean = mean.val[0] * newMaxVal / (maxv - minv) - newMaxVal * (minv) / (maxv - minv);
	double newStdDev = stdDev.val[0] * newMaxVal / (maxv - minv);
//	std::cout << "newMean=" << newMean << "   newStdDev=" << newStdDev << std::endl;

//	// scale C1_saliency_image
//	cv::Mat scaled_C1_saliency_image;
//	double minv, maxv;
//	cv::Point2i minl, maxl;
//	cv::minMaxLoc(C1_saliency_image,&minv,&maxv,&minl,&maxl, mask);
    if (debug_["show_saliency_bad_scale"] == true)
    {
        cv::Mat badscale;
        double badminv, badmaxv;
        cv::Point2i badminl, badmaxl;
        cv::minMaxLoc(C1_saliency_image, &badminv, &badmaxv, &badminl, &badmaxl, mask);
        C1_saliency_image.convertTo(badscale, -1, 1.0 / (badmaxv - badminv), -1.0 * (badminv) / (badmaxv - badminv));
//	std::cout << "bad scale:   " << "\tmin=" << badminv << "\tmax=" << badmaxv << std::endl;
		cv::imshow("bad scale", badscale);
		cvMoveWindow("bad scale", 650, 0);
	}
	//cvMoveWindow("bad scale", 650, 520);
//	cv::Scalar mean, stdDev;
//	cv::meanStdDev(C1_saliency_image, mean, stdDev, mask);
//	std::cout << "min=" << minv << "\tmax=" << maxv << "\tmean=" << mean.val[0] << "\tstddev=" << stdDev.val[0] << std::endl;
//
//	double newMaxVal = min(1.0, maxv/mean.val[0] /5.);
//	//C1_saliency_image.convertTo(scaled_C1_saliency_image, -1, 1.0/(maxv-minv), 1.0*(minv)/(maxv-minv));
//	C1_saliency_image.convertTo(scaled_C1_saliency_image, -1, newMaxVal/(maxv-minv), -newMaxVal*(minv)/(maxv-minv));

	// remove responses that lie on lines
	if (remove_lines_ == true)
	{
		cv::Mat src, dst, color_dst;

		cv::cvtColor(C3_color_image, src, CV_BGR2GRAY);

		// hack: autonomik
		//cv::Canny(src, dst, 150, 200, 3);
		cv::Canny(src, dst, 90, 170, 3);
		cv::cvtColor(dst, color_dst, CV_GRAY2BGR);

		std::vector<cv::Vec4i> lines;
		cv::HoughLinesP(dst, lines, 1, CV_PI / 180, 80, 30, 10);
		for (size_t i = 0; i < lines.size(); i++)
		{
			line(color_dst, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
			line(scaled_C1_saliency_image, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 0), 13, 8);
			// todo: hack for autonomik
			//line(scaled_C1_saliency_image, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0,0,0), 35, 8);
		}

		if (debug_["show_detected_lines"] == true)
		{
			cv::namedWindow("Detected Lines", 1);
			cv::imshow("Detected Lines", color_dst);
		}
	}

	if (debug_["show_saliency_detection"] == true)
	{
		cv::imshow("saliency detection", scaled_C1_saliency_image);
		cvMoveWindow("saliency detection", 0, 530);
	}

	//set dirt pixel to white
	cv::Mat C1_BlackWhite_image = cv::Mat::zeros(C1_saliency_image.size(), CV_8UC1);
	cv::threshold(scaled_C1_saliency_image, C1_BlackWhite_image, dirt_threshold_, 1, cv::THRESH_BINARY);
//	cv::threshold(scaled_C1_saliency_image, C1_BlackWhite_image, mean.val[0] + stdDev.val[0] * dirtCheckStdDevFactor_, 1, cv::THRESH_BINARY);

//	std::cout << "(C1_saliency_image channels) = (" << C1_saliency_image.channels() << ")" << std::endl;

	cv::Mat CV_8UC_image;
	C1_BlackWhite_image.convertTo(CV_8UC_image, CV_8UC1);

//	Mat dst = Mat::zeros(img.rows, img.cols, CV_8UC3);
//	dst = C3_color_image;

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;

	cv::findContours(CV_8UC_image, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	const cv::Scalar green(0, 255, 0);
	const cv::Scalar red(0, 0, 255);
	dirt_detections_mask = cv::Mat::zeros(C1_saliency_image.rows, C1_saliency_image.cols, CV_32F);
	for (int i = 0; i < (int) contours.size(); i++)
	{
		cv::RotatedRect rec = minAreaRect(contours[i]);
		double meanIntensity = 0;
		for (int t = 0; t < (int) contours[i].size(); t++)
		{
			meanIntensity += scaled_C1_saliency_image.at<float>(contours[i][t].y, contours[i][t].x);
			dirt_detections_mask.at<float>(contours[i][t].y, contours[i][t].x) += 1.;
		}
		meanIntensity /= (double) contours[i].size();
		if (meanIntensity > newMean + dirt_check_std_dev_factor_ * newStdDev)
		{
			// todo: hack: for autonomik only detect green ellipses
			//dirtDetections.push_back(rec);
			cv::ellipse(C3_color_image, rec, green, 2);
		}
		else
			cv::ellipse(C3_color_image, rec, red, 2);
		dirtDetections.push_back(rec);
	}
	if (debug_["show_saliency_detection"] == true)
	{
		cv::imshow("dirt_detections_mask", dirt_detections_mask * 255.);
	}
}

std::vector<std::string> readTxt(std::string file)
{
    std::vector<std::string> img_list;
    std::ifstream infile;
    infile.open(file.data());
    assert(infile.is_open());

    std::string line;
    std::string last_img;
    while(getline(infile,line))
    {
        std::vector<std::string> items;
        boost::split(items, line, boost::is_space());
        if (items[0] != last_img)
        {
            img_list.push_back(items[0] + ".png");
            last_img = items[0];
        }
    }
    infile.close();
    return img_list;
}

int main(int argc, char **argv)
{
    // define the label path
    std::string label_path = "/home/rmb-jx/DirtDetectionData/dataset_2019/blended_floor_images/blended_test_floor_images_dirt_only/bbox_labels_no7.txt";

    // get the image path list
    std::vector<std::string> img_list;
    img_list = readTxt(label_path);
    std::vector<std::string>::iterator it;

    // dilate element
    int element_shape = cv::MORPH_RECT;
    cv::Mat element = getStructuringElement(element_shape, cv::Size(13, 13));

    IpaDirtDetection::ServerDirtDetection server_dirt_detection;
    for(it=img_list.begin(); it!=img_list.end(); it++)
    {
        std::cout << *it << std::endl;

        // preprocess
        cv::Mat image_ori, image;
        image_ori = cv::imread("/home/rmb-jx/DirtDetectionData/dataset_2019/blended_floor_images/blended_test_floor_images_dirt_only/"+*it, 1 );
        cv::Size dsize = cv::Size(640, 480);
//        cv::Size dsize = cv::Size(1280, 1024);
        cv::resize(image_ori, image, dsize);
        cv::Mat mask(dsize.height, dsize.width, CV_8UC1, cv::Scalar::all(255));
        cv::Mat prob_map, prob_map_out, prob_map_out_dilate, bin_prob_map;

        // detection
        server_dirt_detection.detectDirt(image, mask, prob_map);
//        prob_map = frame.compute_dirt(mask, image);
//        std::cout << prob_map.rows << ',' << prob_map.cols << ',' << prob_map.channels();

        cv::Size rsize = cv::Size(1280, 1024);
        cv::resize(prob_map, prob_map_out, rsize, (0, 0), (0, 0), cv::INTER_LINEAR);

        // dilate
        cv::morphologyEx(prob_map_out, prob_map_out_dilate, cv::MORPH_CLOSE, element);
        prob_map_out_dilate.convertTo(bin_prob_map, CV_8UC1, 255.);
        // visualize result
        // dirt_map just 0 and 1, suitable for imshow, but imwrite needs 255 s
        cv::imwrite("dirt_only_test/prob_map_close_resize_640/" + *it, bin_prob_map);
//        dirt_map_out.convertTo(save_dirt_map, CV_8UC1, 255);
//        cv::imwrite("dirt_only_test/dirt_map/" + *it, save_dirt_map);
//        cv::imshow("dirt", prob_map_out);
//        cv::waitKey(0);
    }

	return 0;
}
