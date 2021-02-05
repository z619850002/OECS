#include "../../include/selection/pixel_selection.h"
#include <opencv2/opencv.hpp>
#include <time.h>


using namespace std;



//Constructor.
PixelSelection::PixelSelection(){
	//flag = 0   dense
	//flag = 1   sparse
	m_nFlag = 1;
}



vector<cv::Point2d> PixelSelection::GetPixels(cv::Mat & mROI){
	vector<cv::Point2d> mNullVector;
	if (this->m_nFlag == 0){
		return GetDensePixels(mROI);	
	}else{
		return GetSparsePixels(mROI);
	}
	return mNullVector;
}


vector<cv::Point2d> PixelSelection::GetDensePixels(cv::Mat & mROI){
	vector<cv::Point2d> gPoints;
	gPoints.reserve(mROI.size().width * mROI.size().height);
	for (int u=0;u<mROI.size().width;u++){
		for (int v=0;v<mROI.size().height;v++){
			gPoints.push_back(cv::Point2d(u , v));
		}
	}
	return gPoints;
}



vector<cv::Point2d> PixelSelection::GetSparsePixels(cv::Mat & mROI){

	cv::Mat mGradients;
	cv::Sobel(mROI, mGradients, CV_64FC1, 3, 3, 7);
	cv::Scalar     iMean;  
    cv::Scalar     iDev;
	cv::meanStdDev(mGradients , iMean , iDev);
	double nMean = iMean.val[0];
	double nDev = iDev.val[0];

	vector<cv::Point2d> gPoints;
	gPoints.reserve(mROI.size().width * mROI.size().height);
	for (int u=0;u<mROI.size().width;u++){
		for (int v=0;v<mROI.size().height;v++){
			if (mGradients.at<double>(v , u) > nMean + nDev/2 && mGradients.at<double>(v , u) > 60){
				gPoints.push_back(cv::Point2d(u , v));
			}
		}
	}


	return gPoints;
}



double CalculateSigma(vector<cv::Vec3b> iPointVec1 ,vector<cv::Vec3b> iPointVec2){
		
		int nSize = 0;
		double nTotalSigma = 0.0;
		int nTotalSize = iPointVec1.size();
		
		for (int i=0;i<nTotalSize;i++){

			cv::Vec3b mColorPointA = iPointVec1[i];
			cv::Vec3b mColorPointB = iPointVec2[i];
			if (mColorPointB[0] == 0 && 
					mColorPointB[1] == 0 && 
					mColorPointB[2] == 0){
					continue;
			}

			if (mColorPointB[0] == 0){
				mColorPointB[0] = 1;
			}

			if (mColorPointB[1] == 0){
				mColorPointB[1] = 1;
			}

			if (mColorPointB[2] == 0){
				mColorPointB[2] = 1;
			}

			float nScale1 = (float)mColorPointA[0]/(float)mColorPointB[0];
			float nScale2 = (float)mColorPointA[1]/(float)mColorPointB[1];
			float nScale3 = (float)mColorPointA[2]/(float)mColorPointB[2];
			float nAverageScale = (nScale1 + nScale2 + nScale3)/3;
			float nSigma = sqrt((nScale1 - nAverageScale) * (nScale1 - nAverageScale) + 
								(nScale2 - nAverageScale) * (nScale2 - nAverageScale) + 
								(nScale3 - nAverageScale) * (nScale3 - nAverageScale));
			nTotalSigma += nSigma;
		}

		if (nTotalSize == 0){
			return -1.0;
		}
				

		return nTotalSigma / nTotalSize;

		
}





double CalculateSigma(cv::Mat mImage1 , cv::Mat mGeneratedImage, cv::Point2f iPoint){
		int u = iPoint.x;
		int v = iPoint.y;

		int nCol = mImage1.cols;
		int nRow = mImage1.rows;

		int nSize = 0;
		double nTotalSigma = 0.0;
		for (int i = -6;i<=6;i+=3){
			for (int j=-6;j<=6;j+=3){
				if (u + i <= 0 || v + j <=0 || u+i >= nCol || v+j >= nRow){
					continue;
				}
				cv::Vec3b mColorPointA = mImage1.at<cv::Vec3b>(v+i , u+i);
				cv::Vec3b mColorPointB = mGeneratedImage.at<cv::Vec3b>(v+i , u+i);

				if (mColorPointB[0] == 0 && 
					mColorPointB[1] == 0 && 
					mColorPointB[2] == 0){
					continue;
				}

				if (mColorPointB[0] == 0){
					mColorPointB[0] = 1;
				}

				if (mColorPointB[1] == 0){
					mColorPointB[1] = 1;
				}

				if (mColorPointB[2] == 0){
					mColorPointB[2] = 1;
				}

				float nScale1 = (float)mColorPointA[0]/(float)mColorPointB[0];
				float nScale2 = (float)mColorPointA[1]/(float)mColorPointB[1];
				float nScale3 = (float)mColorPointA[2]/(float)mColorPointB[2];
				float nAverageScale = (nScale1 + nScale2 + nScale3)/3;
				float nSigma = sqrt((nScale1 - nAverageScale) * (nScale1 - nAverageScale) + 
									(nScale2 - nAverageScale) * (nScale2 - nAverageScale) + 
									(nScale3 - nAverageScale) * (nScale3 - nAverageScale));
				nTotalSigma += nSigma;
				nSize++;
			}
		}


		if (nSize == 0){
			return -1.0;
		}
				

		return nTotalSigma / nSize;

		
}




vector<cv::Point2d> PixelSelection::GetPixels(cv::Mat & mROIColored, cv::Mat & mNeighbourROIColored){
	//GrayScale
	cv::Mat mROI, mNeighbourROI;
    cv::cvtColor(mROIColored, mROI, cv::COLOR_BGR2GRAY);
	cv::cvtColor(mNeighbourROIColored, mNeighbourROI, cv::COLOR_BGR2GRAY);
	mROI.convertTo(mROI, CV_64FC1);
	mNeighbourROI.convertTo(mNeighbourROI, CV_64FC1);

	vector<cv::Point2d> mNullVector;
	if (this->m_nFlag == 0){
		return GetDensePixels(mROI);	
	}else{
		vector<cv::Point2d> gTwoStep = GetSparsePixels(mROI);

		vector<float> gSigma;
		for (auto iPoint : gTwoStep){
			double nSigma = CalculateSigma(mROIColored, mNeighbourROIColored, iPoint);
			gSigma.push_back(nSigma);
		}

		double nTotal, nMeanSigma;
		//Get the mean gSigma.
		double nSig = 0.0;

		int nNum = 0;
		for (auto item : gSigma){
			nTotal += item;
			nNum++;
		}
		nMeanSigma = nTotal /(float)nNum;


		vector<cv::Point2d> gResult;
		gResult.reserve(gTwoStep.size());

		for (int i=0;i<gTwoStep.size();i++)
		{
			float nSigma = gSigma[i];
			cv::Point2d iPoint = gTwoStep[i];
			if (nSigma < nMeanSigma){
				gResult.push_back(iPoint);
			}
		}
		return gResult;
	}
	return mNullVector;
}

