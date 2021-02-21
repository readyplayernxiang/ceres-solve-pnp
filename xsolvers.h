#pragma once
#ifndef GOOGLE_GLOG_DLL_DECL
#define GOOGLE_GLOG_DLL_DECL
#endif
#include "utils.h"
#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct XCostFunctorPnP
{
	cv::Point3f object_point_;
	cv::Point2f image_point_;
	cv::Mat K_;
	cv::Mat D_;
	XCostFunctorPnP(const cv::Point3f objpoint, const cv::Point2f imgpoint, const cv::Mat camera_intrinsic, const cv::Mat dist_coeffs):
		object_point_(objpoint), image_point_(imgpoint), K_(camera_intrinsic), D_(dist_coeffs){}

	template<typename T>
	bool operator()(const T* const rot, const T* const tra, T* residual) const
	{
		T point_in[3];
		T point_out[3];
		point_in[0] = T(object_point_.x);
		point_in[1] = T(object_point_.y);
		point_in[2] = T(object_point_.z);
		// rotation
		ceres::AngleAxisRotatePoint(rot, point_in, point_out);  // rotate point by the given rot value
		// translation
		point_out[0] += tra[0];
		point_out[1] += tra[1];
		point_out[2] += tra[2];
		
		// projection 
		T x = point_out[0] / point_out[2];
		T y = point_out[1] / point_out[2];
		// undistortation with dist coefficients as [k1, k2, p1, p2, k3]
		// if (!D_empty())
		T r2 = x * x + y * y;
		T xy = x * y;
		x = x * (1.0 + D_.at<double>(0, 0) * r2 + D_.at<double>(0, 1) * r2 * r2 + D_.at<double>(0, 4) * r2 * r2 * r2) + 2.0 * xy * D_.at<double>(0, 2)
			+ (r2 + x * x) * D_.at<double>(0, 3);
		y = y * (1.0 + D_.at<double>(0, 0) * r2 + D_.at<double>(0, 1) * r2 * r2 + D_.at<double>(0, 4) * r2 * r2 * r2) + 2.0 * xy * D_.at<double>(0, 3)
			+ (r2 + y * y) * D_.at<double>(0, 2);
		// to image plane
		T u = x * K_.at<double>(0, 0) + K_.at<double>(0, 2);
		T v = y * K_.at<double>(1, 1) + K_.at<double>(1, 2);
		
		T u_img = T(image_point_.x);
		T v_img = T(image_point_.y);

		residual[0] = u - u_img;
		residual[1] = v - v_img;

		return true;
	}

	static ceres::CostFunction* create(const cv::Point3f objpoint, 
		const cv::Point2f imgpoint, const cv::Mat camera_intrinsic, const cv::Mat dist_coeffs)
	{
		return new ceres::AutoDiffCostFunction<XCostFunctorPnP, 2, 3, 3>
			(new XCostFunctorPnP(objpoint, imgpoint, camera_intrinsic, dist_coeffs));
	}
};


namespace xsolvers
{
	bool solve_pnp(const std::vector<cv::Point3f>& objpoints, const std::vector<cv::Point2f>& imgpoints, 
		const cv::Mat& camera_intrinsic, const cv::Mat& dist_coeffs, cv::Mat& rvec, cv::Mat& tvec);
}

