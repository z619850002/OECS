#include "../include/surround/surround_view_system.h"
#include "../include/loader/frame_loader.h"
#include "../include/initializer/initializer.h"
#include <sstream>
using namespace std;




int main(int argc, char * argv[]){

	int nIndex;
	cout << "Please input the index of image pair you want to use: " << endl;
	cin >> nIndex;

	string aSamplePath = "./samples";

	Initializer iInitializer;
	Camera * pFrontCamera, * pLeftCamera, * pBackCamera, * pRightCamera;
	iInitializer.InitializeCameras(	pFrontCamera,
									pLeftCamera,
									pBackCamera,
									pRightCamera);


	pLeftCamera->BlurPose();
	pRightCamera->BlurPose();






	FrameLoader iLoader(aSamplePath , pFrontCamera,
						pLeftCamera, pBackCamera, 
						pRightCamera);

	cout << "Load pairs" << endl;
	vector<int> gIndices = {};
	for (int i=0;i<=nIndex;i++){
		gIndices.push_back(i);
	}
	vector<SVPair> gPairs = iLoader.LoadFramePairs(gIndices);

	SurroundView iSurround(pFrontCamera, pLeftCamera, pBackCamera, pRightCamera);

	cout << "Bind images" << endl;
	iSurround.BindImagePairs(gPairs);

	cout << "Init K_G" << endl;
	iSurround.InitK_G(1000, 1000, 0.1, 0.1);

	cout << "Finish init K_G" << endl;
	cv::Mat mSurroundView = iSurround.GenerateSurroundView(nIndex, 1000, 1000);
	cv::imshow("Before Optimization", mSurroundView);
	cv::waitKey(0);

	iSurround.OptimizePoseWithOneFrame(nIndex);

	mSurroundView = iSurround.GenerateSurroundView(nIndex, 1000, 1000);
	cv::imshow("After Optimization", mSurroundView);
	cv::waitKey(0);


	return 0;
}