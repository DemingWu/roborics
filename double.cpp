
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <cvaux.h>
#include <cv.h>
#include <cxcore.h>

#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;
//摄像头的分辨率
const int imageWidth = 640;
const int imageHeight = 480;
//横向的角点数目
const int boardWidth = 8;
//纵向的角点数目
const int boardHeight = 6;
//总的角点数目
const int boardCorner = boardWidth * boardHeight;
//相机标定时需要采用的图像帧数
const int frameNumber = 9;
//标定板黑白格子的大小 单位是mm
const int squareSize = 28;
//标定板的总内角点
const Size boardSize = Size(boardWidth, boardHeight);
Size imageSize = Size(imageWidth, imageHeight);

Mat R, T, E, F;
//R旋转矢量 T平移矢量 E本征矩阵 F基础矩阵
vector<Mat> rvecs; //R
vector<Mat> tvecs; //T
//左边摄像机所有照片角点的坐标集合
vector<vector<Point2f>> imagePointL;
//右边摄像机所有照片角点的坐标集合
vector<vector<Point2f>> imagePointR;
//各图像的角点的实际的物理坐标集合
vector<vector<Point3f>> objRealPoint;
//左边摄像机某一照片角点坐标集合
vector<Point2f> cornerL;
//右边摄像机某一照片角点坐标集合
vector<Point2f> cornerR;

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;

Mat intrinsic;
Mat distortion_coeff;
//校正旋转矩阵R，投影矩阵P，重投影矩阵Q
Mat Rl, Rr, Pl, Pr, Q;
//映射表
Mat mapLx, mapLy, mapRx, mapRy;
Rect validROIL, validROIR;
//图像校正之后，会对图像进行裁剪，其中，validROI裁剪之后的区域
/*事先标定好的左相机的内参矩阵
fx 0 cx
0 fy cy
0  0  1
*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 439.92114, 0, 304.73199,
    0, 439.73741, 233.06868,
    0, 0, 1);
//对应matlab里的左相机标定矩阵
Mat distCoeffL = (Mat_<double>(5, 1) << 0.02394, 0.05374, 0.00207, -0.00303, 0.00000);
//对应Matlab所得左i相机畸变参数

Mat cameraMatrixR = (Mat_<double>(3, 3) << 441.99942, 0, 338.33506,
    0, 441.82529, 257.90994,
    0, 0, 1);
//对应matlab里的右相机标定矩阵

Mat distCoeffR = (Mat_<double>(5, 1) << 0.08135, -0.15556, -0.00010, 0.00126, 0.00000);
//对应Matlab所得右相机畸变参数


/*计算标定板上模块的实际物理坐标*/
void calRealPoint(vector<vector<Point3f>>& obj, int boardWidth, int boardHeight, int imgNumber, int squareSize)
{
    vector<Point3f> imgpoint;
    for (int rowIndex = 0; rowIndex < boardHeight; rowIndex++)
    {
        for (int colIndex = 0; colIndex < boardWidth; colIndex++)
        {
            imgpoint.push_back(Point3f(rowIndex * squareSize, colIndex * squareSize, 0));
        }
    }
    for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
    {
        obj.push_back(imgpoint);
    }
}



void outputCameraParam(void)
{
    /*保存数据*/
    /*输出数据*/
    FileStorage fs("intrisics.yml", FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "cameraMatrixL" << cameraMatrixL << "cameraDistcoeffL" << distCoeffL << "cameraMatrixR" << cameraMatrixR << "cameraDistcoeffR" << distCoeffR;
        fs.release();
        cout << "cameraMatrixL=:" << cameraMatrixL << endl << "cameraDistcoeffL=:" << distCoeffL << endl << "cameraMatrixR=:" << cameraMatrixR << endl << "cameraDistcoeffR=:" << distCoeffR << endl;
    }
    else
    {
        cout << "Error: can not save the intrinsics!!!!" << endl;
    }

    fs.open("extrinsics.yml", FileStorage::WRITE);
    if (fs.isOpened())
    {
        fs << "R" << R << "T" << T << "Rl" << Rl << "Rr" << Rr << "Pl" << Pl << "Pr" << Pr << "Q" << Q;
        cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr" << Rr << endl << "Pl" << Pl << endl << "Pr" << Pr << endl << "Q" << Q << endl;
        fs.release();
    }
    else
    {
        cout << "Error: can not save the extrinsic parameters\n";
    }

}


int main(int argc, char* argv[])
{
    Mat img;
    int goodFrameCount = 0;
    while (goodFrameCount < frameNumber)
    {
        char filename[100];

        sprintf(filename, "../Picture/left%d.jpg", goodFrameCount );
        rgbImageL = imread(filename, CV_LOAD_IMAGE_COLOR);
        //imshow("chessboardL", rgbImageL);
        cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
        sprintf(filename, "../Picture/right%d.jpg", goodFrameCount );
        rgbImageR = imread(filename, CV_LOAD_IMAGE_COLOR);
        cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);

        bool isFindL, isFindR;
        isFindL = findChessboardCorners(rgbImageL, boardSize, cornerL);
        isFindR = findChessboardCorners(rgbImageR, boardSize, cornerR);
        if (isFindL == true && isFindR == true)
        {
            cornerSubPix(grayImageL, cornerL, Size(5, 5), Size(-1, 1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
            drawChessboardCorners(rgbImageL, boardSize, cornerL, isFindL);
            //imshow("chessboardL", rgbImageL);
            imagePointL.push_back(cornerL);

            cornerSubPix(grayImageR, cornerR, Size(5, 5), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1));
            drawChessboardCorners(rgbImageR, boardSize, cornerR, isFindR);
            //imshow("chessboardR", rgbImageR);
            imagePointR.push_back(cornerR);
            //cout << "the image" << goodFrameCount << " is good" << endl;
            goodFrameCount++;
        }
        else
        {
            //cout << "the image" << goodFrameCount << " is bad" << endl;
            goodFrameCount++;
        }

    }

    calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
    cout << "cal real successful" << endl;


    double rms =  stereoCalibrate(objRealPoint, imagePointL, imagePointR,
                                  cameraMatrixL, distCoeffL,
                                  cameraMatrixR, distCoeffR,
                                  Size(imageWidth, imageHeight), R, T, E, F,CALIB_USE_INTRINSIC_GUESS,
                                  TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
           // CALIB_USE_INTRINSIC_GUESS;
    cout << "Stereo Calibration done with RMS error = " << rms << endl;

    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl,
        Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);



    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    /*
    Mat rectifyImageL, rectifyImageR;
    cvtColor(grayImageL, rectifyImageL, CV_GRAY2BGR);
    cvtColor(grayImageR, rectifyImageR, CV_GRAY2BGR);


    imshow("Recitify Before", rectifyImageL);
    cout << "按Q1退出..." << endl;

    Mat rectifyImageL2, rectifyImageR2;
    remap(rectifyImageL, rectifyImageL2, mapLx, mapLy, INTER_LINEAR);
    remap(rectifyImageR, rectifyImageR2, mapRx, mapRy, INTER_LINEAR);
    cout << "按Q2退出..." << endl;

    imshow("rectifyImageL", rectifyImageL2);
    imwrite("rectifyImageL.jpg", rectifyImageL2);
    imshow("rectifyImageR", rectifyImageR2);
    imwrite("rectifyImageR.jpg", rectifyImageR2);
    outputCameraParam();

    //显示校正结果
    Mat canvas;
    double sf;
    int w, h;
    sf = 600. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width * sf);
    h = cvRound(imageSize.height * sf);
    canvas.create(h, w * 2, CV_8UC3);

    //左图像画到画布上
    Mat canvasPart = canvas(Rect(0, 0, w, h));
    resize(rectifyImageL2, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
    Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),
        cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
    rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);

    cout << "Painted ImageL" << endl;

    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));
    resize(rectifyImageR2, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    Rect vroiR(cvRound(validROIR.x*sf), cvRound(validROIR.y*sf),
        cvRound(validROIR.width*sf), cvRound(validROIR.height*sf));
    rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);

    cout << "Painted ImageR" << endl;

    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);

    imshow("rectified", canvas);
    imwrite("rectified.jpg", canvas);
    cout << "wait key" << endl;
    waitKey(0);
*/



    Mat grayImageL = imread("../left22.jpg", 0);
    Mat grayImageR = imread("../right22.jpg", 0);
    Mat rectifyImageL, rectifyImageR;
    cvtColor(grayImageL, rectifyImageL, CV_GRAY2BGR);
    cvtColor(grayImageR, rectifyImageR, CV_GRAY2BGR);
    Mat rectifyImageL2, rectifyImageR2;
    remap(rectifyImageL, rectifyImageL2, mapLx, mapLy, INTER_LINEAR);
    remap(rectifyImageR, rectifyImageR2, mapRx, mapRy, INTER_LINEAR);
    imwrite("../rgbRectifyImageL.jpg", rectifyImageL2);
    imwrite("../rgbRectifyImageR.jpg", rectifyImageR2);






   // IplImage * img1 = cvLoadImage("../rgbRectifyImageL.jpg", 0);
  //  IplImage * img2 = cvLoadImage("../rgbRectifyImageR.jpg", 0);
    Mat img1 = imread("../rgbRectifyImageL.jpg", 0);
    Mat img2 = imread("../rgbRectifyImageR.jpg", 0);
    //Mat img1 = imread("../left.png", 1);
    //Mat img2 = imread("../right.png", 1);

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();

    //StereoSGBM sgbm = createStereoBM();
    int SADWindowSize = 11;
    sgbm->setPreFilterCap(63);
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);
    int cn = img1.channels();
    int numberOfDisparities = 144;
    sgbm->setP1( 8 * cn*sgbm->getSpeckleWindowSize()*sgbm->getSpeckleWindowSize());
    sgbm->setP2( 32 * cn*sgbm->getSpeckleWindowSize()*sgbm->getSpeckleWindowSize());
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(300) ;
    sgbm->setSpeckleRange(10);
    sgbm->setDisp12MaxDiff(1);
    Mat disp, disp8;
    int64 t = getTickCount();
    //sgbm((Mat)img1, (Mat)img2, disp);
    sgbm->compute(img1, img2, disp);
    t = getTickCount() - t;
    cout << "Time elapsed:" << t * 1 / getTickFrequency() << endl;
    disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));

    Mat img3,img4,img5;
    threshold(disp8, img3, 160, 255, THRESH_BINARY);
    threshold(disp8, img4, 80, 255, THRESH_BINARY);
    threshold(disp8, img5, 0, 255, THRESH_BINARY);

    img5 = img5 - img4;
    img4 = img4 - img3;
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    // morphologyEx(disp8, disp8, MORPH_CLOSE, element);

    /* cvNamedWindow("1", 0);
     imshow("1", img3);
     cvNamedWindow("2", 0);
     imshow("2", img4);
     cvNamedWindow("3", 0);
     imshow("3", img5);*/
    namedWindow("left", 1);
    imshow("left", img1);
    namedWindow("right", 1);
    imshow("right", img2);
    namedWindow("disparity", 1);
    imshow("disparity", disp8);
    imwrite("sgbm_disparity.png", disp8);
    waitKey();
    cvDestroyAllWindows();

    return 0;
}