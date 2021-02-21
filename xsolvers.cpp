#include "xsolvers.h"


bool xsolvers::solve_pnp(const std::vector<cv::Point3f>& objpoints, const std::vector<cv::Point2f>& imgpoints, 
	const cv::Mat& camera_intrinsic, const cv::Mat& dist_coeffs, cv::Mat& rvec, cv::Mat& tvec)
{
	assert(rvec.type() == CV_32F);
	assert(tvec.type() == CV_32F);
	double rot[3];
	double tra[3];
	rot[0] = rvec.at<float>(0, 0);
	rot[1] = rvec.at<float>(1, 0);
	rot[2] = rvec.at<float>(2, 0);
	tra[0] = tvec.at<float>(0, 0);
	tra[1] = tvec.at<float>(1, 0);
	tra[2] = tvec.at<float>(2, 0);

	ceres::Problem problem;
	for (int i = 0; i < imgpoints.size(); i++)
	{
		ceres::CostFunction* cost = XCostFunctorPnP::create(objpoints[i], imgpoints[i], camera_intrinsic, dist_coeffs);
		problem.AddResidualBlock(cost, nullptr, rot, tra);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.max_num_iterations = 80;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	options.minimizer_progress_to_stdout = false;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	rvec = (cv::Mat_<float>(3, 1) << rot[0], rot[1], rot[2]);
	tvec = (cv::Mat_<float>(3, 1) << tra[0], tra[1], tra[2]);

	return true;
}
