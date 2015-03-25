#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d.hpp"


#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <unordered_map>


#include "ceres_optimization.h"
#include "ceres/ceres.h"
#include "glog/logging.h"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;



/// Global variables
float k[3][3] = {{1077.9,0,594.0},{0,1077.9,393.3},{0,0,1}};
cv::Mat K(3,3,CV_32FC1,k); //Camera Matrix
cv::Mat Kd(3,3,CV_64FC1,Scalar::all(0));
const int num_pic = 9;
cv::Mat src[num_pic], src_gray[num_pic];
//Initializing Hash Table For correspondence
vector<std::unordered_map<int, int>> threeD_point2img;
vector<Vec3b> threeD_point_rgb;
vector<cv::Mat> threeD_point_loc;
std::unordered_map<int, int> threeD_img2point_map[num_pic]; //Number of imgs

vector<cv::KeyPoint> keypoints[num_pic];
cv::Mat P[num_pic];




const char* source_window = "Source image";
const char* corners_window = "Sift detected";
const char* matches_window = "Good Matches";

const float rows = 1200;
const float cols = 1600;


const float offset_rows = 300;
const float offset_cols = 1600;

#define TO_EXTENDED_COORD 1
#define TO_REFERED_COORD 0
#define REFEREE_ID 1
#define RANSAC_TIMES 34
#define NORM 1
#define NON_NORM 0

float norm_ma[3][3] = {{2/cols,0,-1},{0,2/rows,-1},{0,0,1}};
cv::Mat norm_matrix = cv::Mat(3, 3, CV_32FC1,norm_ma);

float tmp[4][3] = {{0,0,1},{1600,0,1},{0,1200,1},{1600,1200,1}};
cv::Mat m_tmp(4,3,CV_32FC1,tmp);
cv::Mat board_coord = m_tmp.t();

/// Function header
void siftDetector( int, void* );
void goodMatches(cv::Mat descriptor1, cv::Mat descriptor2,std::vector<DMatch>* good_matches,float par1, float par2);
vector<Point2d>  normCoord(vector<cv::KeyPoint> keypoints);
void findMatchingPoint(vector<KeyPoint> K1, vector<KeyPoint> K2,vector<DMatch> good_matches, vector<Point2d> *match_point1, vector<Point2d> *match_point2, bool norm_flag);
cv::Mat findE(vector<cv::KeyPoint> keypoints1,vector<cv::KeyPoint> keypoints2,vector<DMatch> good_matches);
void convertP2DtoMat(vector<Point2d> point, cv::Mat *mat_point);
void convertP2DtoMat(Point2d point, cv::Mat *mat_point);
void convertP3ftoMat(vector<Point3f> point, cv::Mat *mat_point);
void convertP2fToP2d(vector<Point2f> src, vector<Point2d> *dst);
void convertVec2CrossMat(cv::Mat vec_origin, cv::Mat *cross_mat);
void convertP2DtoMatVec(vector<Point2d> point, vector<cv::Mat> *mat_point);
void convertVec2CrossMat_Vec(vector<cv::Mat> vec_origin_src, vector<cv::Mat> *cross_mat);


double computeProjectPointError(cv::Mat Pk, Point2d point, cv::Mat point_4D);
cv::Mat myLinearTriangulation(Point2d vec_point1, Point2d vec_point2,cv::Mat P, cv::Mat P_prime);
bool ransacTriangulation(cv::Mat P1, cv::Mat P2, vector<Point2f> vec_key_point_1, vector<Point2f> vec_key_point_2, cv::Mat  *triangulated_point);

double computeProjectPointError(cv::Mat Pk, Point2d point, cv::Mat point_4D);
void doTriangulation2Images(vector<cv::KeyPoint> K1,vector<cv::KeyPoint> K2, vector<DMatch> good_matches, cv::Mat img1, cv::Mat img2, cv::Mat P, cv::Mat P_prime);
void doMulTriangulation(int imgIdx1, int imgIdx2, vector<DMatch> good_matches, vector<KeyPoint> K1, vector<KeyPoint> K2, vector<int> inlier_mask);
void testRT(cv::Mat R1, cv::Mat R2, cv::Mat T, cv::Mat *P2, vector<cv::KeyPoint> K1, vector<cv::KeyPoint> K2, std::vector<DMatch> good_matches);
cv::Mat findF(vector<cv::KeyPoint> keypoints1,vector<cv::KeyPoint> keypoints2,vector<DMatch> good_matches);
void computeReconstructPoint(int imgIdx1, int imgIdx2, vector<DMatch> good_matches,vector<int> inliers);
void computePkUsing3D2D(int imgIdx1, int imgIdx2, vector<DMatch> good_matches, vector<KeyPoint> K1, vector<KeyPoint> K2,  cv::Mat *Pk);
void doRansacChooseInlier(vector<Point3f> vec_reconstructed_point, vector<Point2f> vec_img2_point, cv::Mat Pk, vector<int> *inliers);

void drawEpilines(vector<Point2d> points1, vector<Point2d> points2, cv::Mat F, cv::Mat src1, cv::Mat src2);
void drawEpilinesHelper(vector<Point2d> points1, vector<Point2d> points2, cv::Mat F, cv::Mat *img1, cv::Mat *img2);
void convertMat3Dto2d(cv::Mat mat_point, vector<Point2d> *vec_point);

cv::Mat buildA(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> good_matches);
cv::Mat buildCoord(int x_min, int x_max, int y_min, int y_max);
cv::Mat coordTransX2Xprime(cv::Mat x,cv::Mat H); // H X2Xprime H.inv() Xprime2X
cv::Mat coordCalib(cv::Mat x,bool flag);
cv::Mat getTrans(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> good_matches);
void affineTrans(cv::Mat* s_image, cv::Mat* out_img, cv::Mat coord_source, cv::Mat coord_out);
cv::Mat linearBlend(cv::Mat img1, cv::Mat img2);
void getImgNSrcCoord(cv::Mat H, cv::Mat *img_coord, cv::Mat *out_img_coord);
void ransac(vector<DMatch> good_matches, vector<DMatch>* inlier_matches, vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2);
void randomArray(int size, int range, int *array);


/**
 * @function main
 */

int main( int, char** argv )
{
    srand (time(NULL));
    K.convertTo(Kd, 6);

    /// Load source image and convert it to grayn
    for (int i=0; i<num_pic; i++) {
        src[i] = imread( argv[1+i], 1 );
        cvtColor( src[i], src_gray[i], COLOR_BGR2GRAY );
    }
    
    
    siftDetector( 0, 0 );
    
    waitKey(0);
    return(0);
}


/**
 * @function siftDetector
 * @brief Executes the sift detection and draw a circle around the possible keypoints
 */
void siftDetector( int, void* )
{
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(5000,3,0.04,10,1.6);
  //  vector<cv::KeyPoint> keypoints[num_pic];
    cv::Mat mat_keypoints[num_pic];
    cv::Mat descriptor[num_pic];
    vector<DMatch> good_matches[num_pic];
    cv::Mat out_img[num_pic];
    for (int i=0; i<num_pic; i++) {
        f2d->detect(src_gray[i], keypoints[i]);
        f2d->compute(src_gray[i], keypoints[i], descriptor[i]);
    }
    
    goodMatches(descriptor[0], descriptor[1],&(good_matches[0]),2,0.02);

    for (int i=1; i<num_pic; i++) {
        goodMatches(descriptor[i], descriptor[(i+1)%num_pic],&(good_matches[i]),3,0.02);
    }
    
    //Initializing Projection Mat
    //  cv::Mat P[num_pic];
    cv::Mat I = Mat::eye(3, 3, CV_64FC1);
    hconcat(I, cv::Mat(3,1,CV_64FC1,Scalar::all(0)), P[0]);
    
    cout << "Good Matches  " << good_matches[0].size() << endl;
    
    //Do 2 Images Triangulation
    cv::Mat E = findE(keypoints[0], keypoints[1], good_matches[0]);
    cv:Mat R1,R2,T;
    decomposeEssentialMat(E, R1, R2, T);
    testRT(R1, R2, T, &P[1], keypoints[0], keypoints[1], good_matches[0]);
    doTriangulation2Images(keypoints[0], keypoints[1], good_matches[0], src[0], src[1], P[0], P[1]);
    cout << "threeDpointRGB  " << threeD_point_rgb.at(3) << endl;
    cout << "threeDpointLoc " << threeD_point_loc.at(3) << endl;
    cout << "threeDpointSize " << threeD_point_loc.size() << endl;
    
    vector<int> tmp_inliers;
    computeReconstructPoint(0, 1, good_matches[0], tmp_inliers); //Compute Map For First Triangulation
    
    cout << "threeDpointsPointMapsize" << threeD_point2img[0].size() << endl;
    
    for (int i=2; i<num_pic; i++) {
        computePkUsing3D2D(i-1, i, good_matches[i-1], keypoints[i-1], keypoints[i],&P[i]);

    }
    
    cout << "threeDpointsMapsize img 0" << threeD_img2point_map[0].size() << endl;
    
    cout << "New P2 is " << P[1] << endl;
    
    ofstream myfile;
    myfile.open("./plyFiles/threeDpointsMapsize.txt");
    
    for (int i=0; i<threeD_point_rgb.size(); i++) {
        myfile << threeD_point_loc.at(i)<< endl;
        myfile << threeD_point_rgb.at(i) << endl;
    }
    
    myfile.close();

    cout << "Total Points " <<  threeD_point_loc.size() << endl;

    
//    //Draw Sift Keypoints
//    Mat img_keypoints_1;
//    drawKeypoints(src_gray[0], keypoints[0], img_keypoints_1);
//    imshow("Keypoints 1", img_keypoints_1);

    
    //Find Fundamental Matrix
  //  cv::Mat F = findF(keypoints[0], keypoints[1], good_matches[0]);
   // cout << F << endl;
    
    vector<Point2d> vec_match_point1, vec_match_point2;
    findMatchingPoint(keypoints[0], keypoints[1], good_matches[0], &vec_match_point1, &vec_match_point2,NON_NORM);
    
      cv::Mat F = ((Kd.t()).inv())*E*(Kd.inv());
    
  //  cout << "F is " << F << endl;
    drawEpilines(vec_match_point1, vec_match_point2, F,src[0],src[1]);
  //  cout << "E  " << endl << E << endl;
    
    
    Mat img_matches;
    drawMatches(src_gray[0], keypoints[0], src_gray[1], keypoints[1], good_matches[0], img_matches);
    
    resize(img_matches, img_matches, Size(img_matches.cols/2,img_matches.rows/2));
    imshow( "Good Matches", img_matches );
    
}


/**
 * @function goodMatches
 * @find Mathces
 */
void goodMatches(cv::Mat descriptor1, cv::Mat descriptor2,std::vector<DMatch> *good_matches,float par1, float par2)
{
    //Feature Matching
    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptor1, descriptor2, matches);
    double max_dist = 0; double min_dist = 100;
    
    //Caculate max and min distances between keypoints
    for (int i =0 ; i < descriptor1.rows; i++)
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    
    //Draw Good Matches
    for( int i = 0; i < descriptor1.rows; i++ )
    { if( matches[i].distance <= max((par1*min_dist), 0.02) )
    { (*good_matches).push_back( matches[i]); }
    }
}

vector<Point2d> normCoord(vector<cv::KeyPoint> keypoints) //Normalize Keypoinst coordinate
{
    int num = keypoints.size();
    double coord[3][num]; // 3*N Matrix
    for (int i=0; i<num; i++) {
        coord[0][i] = (double)keypoints.at(i).pt.x;
        coord[1][i]  = (double)keypoints.at(i).pt.y;
        coord[2][i]  = (double)1.0;
    }
    cv::Mat Coord(3,num,CV_64FC1,coord);
    Coord = (Kd.inv())*(Coord);
    
    vector<Point2d> vec_coord;
    for (int i=0; i<num; i++) {
        Point2d tmp_point;
        tmp_point.x = Coord.at<double>(0,i)/Coord.at<double>(2,i);
        tmp_point.y = Coord.at<double>(1,i)/Coord.at<double>(2,i);
        vec_coord.push_back(tmp_point);
    }
 //   cout << vec_coord[0] << endl;
    
    return vec_coord;
};

void findMatchingPoint(vector<KeyPoint> K1, vector<KeyPoint> K2,vector<DMatch> good_matches, vector<Point2d> *match_point1, vector<Point2d> *match_point2,bool norm_flag)
{
    if (norm_flag) {
        vector<cv::KeyPoint> point1, point2;
        int num = good_matches.size();
        for (int ii=0; ii<num; ii++) {
            // cout << keypoints1[good_matches[ii].queryIdx].pt << endl;
            
            point1.push_back(K1[good_matches[ii].queryIdx]);
            point2.push_back(K2[good_matches[ii].trainIdx]);
        }
        *match_point1 = normCoord(point1);
        *match_point2 = normCoord(point2);

    } else {
        int num = good_matches.size();
        for (int ii=0; ii<num; ii++) {
            Point2d tmp_point1,tmp_point2;
            tmp_point1 = K1[good_matches[ii].queryIdx].pt;
            (*match_point1).push_back(tmp_point1);
            tmp_point2 = K2[good_matches[ii].trainIdx].pt;
            (*match_point2).push_back(tmp_point2);
        }
    }
}

cv::Mat findE(vector<cv::KeyPoint> keypoints1,vector<cv::KeyPoint> keypoints2,vector<DMatch> good_matches) //Find Essential Matrix using 2 images
{
    vector<Point2d> vec_match_point1, vec_match_point2;
    findMatchingPoint(keypoints1, keypoints2, good_matches, &vec_match_point1, &vec_match_point2,NORM);
    
    cv::Mat E = findEssentialMat(vec_match_point1, vec_match_point2);
    return E;
}

cv::Mat findF(vector<cv::KeyPoint> keypoints1,vector<cv::KeyPoint> keypoints2,vector<DMatch> good_matches) //Find Fundamental Matrix
{
    vector<Point2d> vec_match_point1, vec_match_point2;
    findMatchingPoint(keypoints1, keypoints2, good_matches, &vec_match_point1, &vec_match_point2,NON_NORM);
    cv::Mat F = findFundamentalMat(vec_match_point1, vec_match_point2);
    return F;
}

void convertP2DtoMatVec(vector<Point2d> point, vector<cv::Mat> *mat_point)
{
    int num = point.size();
    for (int i=0; i<num; i++) {
        cv::Mat tmp_mat_point(3,1,CV_64FC1);
        (tmp_mat_point).at<double>(0,0) = point.at(i).x;
        (tmp_mat_point).at<double>(1,0) = point.at(i).y;
        (tmp_mat_point).at<double>(2,0) = (double)1.0;
        (*mat_point).push_back(tmp_mat_point);
        tmp_mat_point.release();
    }
}

void convertP2DtoMat(vector<Point2d> point, cv::Mat *mat_point)
{
    int num = point.size();
    for (int i=0; i<num; i++) {
        (*mat_point).at<double>(0,i) = point.at(i).x;
        (*mat_point).at<double>(1,i) = point.at(i).y;
        (*mat_point).at<double>(2,i) = (double)1.0;
    }
}


void convertP3ftoMat(vector<Point3f> point, cv::Mat *mat_point)
{
    int num = point.size();
    for (int i=0; i<num; i++) {
        (*mat_point).at<double>(0,i) = (double)point.at(i).x;
        (*mat_point).at<double>(1,i) = (double)point.at(i).y;
        (*mat_point).at<double>(2,i) = (double)point.at(i).z;
        (*mat_point).at<double>(3,i) = (double) 1.0;
    }
    //tmp_mat_point.release();

}

void convertMat3Dto2f(cv::Mat mat_point, vector<Point2f> *vec_point)
{
    int num = mat_point.cols;
    for (int i=0; i<num; i++) {
        Point2f tmp2f;
        tmp2f.x = (float)(mat_point.at<double>(0, i)/mat_point.at<double>(2,i));
        tmp2f.y = (float)(mat_point.at<double>(1, i)/mat_point.at<double>(2,i));
        (*vec_point).push_back(tmp2f);
    }
}

void convertMat3Dto2d(cv::Mat mat_point, vector<Point2d> *vec_point)
{
    int num = mat_point.cols;
    for (int i=0; i<num; i++) {
        Point2d tmp2d;
        tmp2d.x = (mat_point.at<double>(0, i)/mat_point.at<double>(2,i));
        tmp2d.y = (mat_point.at<double>(1, i)/mat_point.at<double>(2,i));
        (*vec_point).push_back(tmp2d);
    }
}

void convertP2fToP2d(vector<Point2f> src, vector<Point2d> *dst){
    for (int i=0; i<src.size(); i++) {
        Point2d tmp2d;
        tmp2d.x = (double) src.at(i).x;
        tmp2d.y = (double) src.at(i).y;
        (*dst).push_back(tmp2d);
    }
}


void convertP2DtoMat(Point2d point, cv::Mat *mat_point)
{
        (*mat_point).at<double>(0,0) = point.x;
        (*mat_point).at<double>(1,0) = point.y;
        (*mat_point).at<double>(2,0) = (double)1.0;
}

void convertVec2CrossMat(cv::Mat vec_origin, cv::Mat *cross_mat)
{
    cv::Mat tmp = cv::Mat::zeros(3, 3, CV_64FC1);
    tmp.copyTo(*cross_mat);
    (*cross_mat).at<double>(1,0) = vec_origin.at<double>(0,2);//a3
    (*cross_mat).at<double>(2,0) = -1*(vec_origin.at<double>(0,1));//-a2
    (*cross_mat).at<double>(0,1) = -1*(vec_origin.at<double>(0,2));//-a3
    (*cross_mat).at<double>(2,1) = vec_origin.at<double>(0,0);//a1
    (*cross_mat).at<double>(0,2) = vec_origin.at<double>(0,1);//a2
    (*cross_mat).at<double>(1,2) = -1*(vec_origin.at<double>(0,0));//-a1

}

void convertVec2CrossMat_Vec(vector<cv::Mat> vec_origin_src, vector<cv::Mat> *cross_mat)
{
    for (int i=0; i<vec_origin_src.size(); i++) {
        cv::Mat tmp = cv::Mat::zeros(3, 3, CV_64FC1);
        cv::Mat vec_origin;
        vec_origin_src.at(i).copyTo(vec_origin);
        (tmp).at<double>(1,0) = vec_origin.at<double>(0,2);//a3
        (tmp).at<double>(2,0) = -1*(vec_origin.at<double>(0,1));//-a2
        (tmp).at<double>(0,1) = -1*(vec_origin.at<double>(0,2));//-a3
        (tmp).at<double>(2,1) = vec_origin.at<double>(0,0);//a1
        (tmp).at<double>(0,2) = vec_origin.at<double>(0,1);//a2
        (tmp).at<double>(1,2) = -1*(vec_origin.at<double>(0,0));//-a1
        (*cross_mat).push_back(tmp);
    }
}



void drawEpilinesHelper(vector<Point2d> points1, vector<Point2d> points2, cv::Mat F, cv::Mat *img1, cv::Mat *img2)
{
    cv::Mat mat_point1(3,points1.size(),CV_64FC1);
    convertP2DtoMat(points1, &mat_point1);
    cv::Mat lines(3,points1.size(),CV_64FC1);
    lines = F*mat_point1;
   // cout << lines << endl;
    
    double a,b,c,d;
    vector<Point2i> vec_A,vec_B;
    for (int i=0; i<mat_point1.cols; i++) {
        a = lines.at<double>(0,i);
        b = lines.at<double>(1,i);
        c = lines.at<double>(2,i);
        d = (*img2).cols;

        Point2i A,B;
        A.x = 0; A.y = (-1*c)/b;
        B.x = d; B.y = -1*(a*d + c)/b;
        vec_A.push_back(A);
        vec_B.push_back(B);
        
        cv::line(*img2, A, B, Scalar(0, 0, 255));
        cv::circle(*img2, points2.at(i), 8, Scalar(rand()%255, rand()%255, rand()%255),-1);

    }
    resize(*img2, *img2, Size((*img2).cols/2,(*img2).rows/2));
   // imshow("img", *img2);
}

void drawEpilines(vector<Point2d> points1, vector<Point2d> points2, cv::Mat F, cv::Mat src1, cv::Mat src2)
{
    cv::Mat img1,img2;
    src1.copyTo(img1); src2.copyTo(img2);
    drawEpilinesHelper(points1, points2, F, &img1, &img2);
    drawEpilinesHelper(points2, points1, F.t(), &img2, &img1);
    
    imshow("img1", img1);
    imshow("img2", img2);
}

double computeProjectPointError(cv::Mat Pk, Point2d point, cv::Mat point_4D){
    cv::Mat reconstructed_point = Pk*point_4D;
    vector<Point2d> vec_point;
    convertMat3Dto2d(reconstructed_point, &vec_point);
    
    Point2d tmp = point - vec_point.at(0);
    double err = pow(tmp.x, 2) + pow(tmp.y, 2);
    return err;
}

cv::Mat myLinearTriangulation(Point2d vec_point1, Point2d vec_point2,cv::Mat P, cv::Mat P_prime) //Using Non-Normalized Coordinate
{
    cv::Mat ho_point1(3,1,CV_64FC1);
    cv::Mat ho_point2(3,1,CV_64FC1);
  //  cout << "vec_point1 " << endl << vec_point1 << endl;
  //  cout << "ho_point1 " << endl << ho_point1 << endl;
    
    convertP2DtoMat(vec_point1,&ho_point1); //3*N
    convertP2DtoMat(vec_point2, &ho_point2); //3*N
    cv::Mat A1,A2,A;
    cv::Mat cross_mat1(3,3,CV_64FC1);
    cv::Mat cross_mat2(3,3,CV_64FC1);
    convertVec2CrossMat(ho_point1, &cross_mat1);
    convertVec2CrossMat(ho_point2, &cross_mat2);

   // cout << "P is " << endl << P << endl;
    A1 = cross_mat1*P;
    A2 = cross_mat2*P_prime;
    
    vconcat(A1, A2, A);
    cv::Mat w,u,vt;
    SVD::compute(A.t()*A, w, u, vt);
    cv::Mat X_hat((vt.t()).col(3));
  //  cout << "Vt " << vt.t().col(3) << endl;
    return X_hat;
}

cv::Mat myMulTriHelper(vector<cv::Mat> inlier_Ps, vector<Point2d> inlier_locs)
{
    int num = inlier_locs.size();
    
    vector<cv::Mat> ho_point;
    convertP2DtoMatVec(inlier_locs, &ho_point); //Location is after NormCoord
    
    vector<cv::Mat> smallA;
    cv::Mat A;
    
    vector<cv::Mat> cross_mat;
    convertVec2CrossMat_Vec(ho_point, &cross_mat);

    for (int i=0; i< num; i++) {
    //    cout << "inlier locs " << endl << inlier_locs.at(i) << endl;
    //    cout << "ho_point " << endl << ho_point.at(i) << endl;
    //    cout << "cross_mat " << endl << cross_mat.at(i) << endl;
    //    cout << "inlier P " <<  endl << inlier_Ps.at(i) << endl;
        cv::Mat tmp_A = cross_mat.at(i)*(inlier_Ps.at(i));
        smallA.push_back(tmp_A);
    }
    
    vconcat(smallA.at(0), smallA.at(1), A);
    for (int i=2; i<num; i++) {
        //Concat All A
        vconcat(A, smallA.at(i), A);
     //   cout << "small A (i)" << smallA.at(i) << endl;
     //   cout << "A is " << A << endl;
    }
    
    
    
    cv::Mat w,u,vt;
    SVD::compute(A.t()*A, w, u, vt);
    cv::Mat X_hat((vt.t()).col(3));
  //  cout << "Vt " << vt.t().col(3) << endl;
    return X_hat;
}

cv::Mat myMulTriangulation(int pointIdx)
{
    
    std::unordered_map<int,int>::iterator got;
    vector<int> imgs;
    vector<int> points;
    vector<KeyPoint> points_loc;
    vector<cv::Mat> rnd_P;
    cv::Mat P_best;
    
    for (int i=0; i<11; i++) { //11 Images
        got = threeD_point2img.at(pointIdx).find(i);
        if (got != threeD_point2img.at(pointIdx).end()) {
            int keypointIdx = threeD_point2img.at(pointIdx).at(i);
            imgs.push_back(i);
            points.push_back(keypointIdx);
            KeyPoint tmp2k = keypoints[i].at(keypointIdx);// The point in that image
            points_loc.push_back(tmp2k);
            rnd_P.push_back(P[i]);
        }
    }
    
    vector<Point2d> points_loc_2d = normCoord(points_loc);
    int num_imgs = imgs.size();
    
    Point2d rnd_poin[2];
    double error_thresh_hold = 0.0002;
    double tmp_err = 0;
    int most_inliers = -1;
    int good_p_indx[2] = {0,1};
    
    for (int i=0; i<(num_imgs*num_imgs); i++) { //ransac times 100
        int rnd_array[2];
        int tmp_inliers = 0;
        randomArray(2, num_imgs, rnd_array);
        cv::Mat rnd_points_loc = myLinearTriangulation(points_loc_2d.at(rnd_array[0]), points_loc_2d.at(rnd_array[1]),rnd_P.at(rnd_array[0]) , rnd_P.at(rnd_array[1]));

        for (int j=0; j<num_imgs ; j++) {
            if ((j!=rnd_array[0])&&(j!=rnd_array[1])) {
                tmp_err = computeProjectPointError(rnd_P.at(j), points_loc_2d.at(j), rnd_points_loc);
            //    cout << "tmp_err " << tmp_err << endl;
                if (tmp_err < error_thresh_hold) {
                    tmp_inliers +=1; //Inliers Count
                }
            }
        }
        if (tmp_inliers > most_inliers) {
            good_p_indx[0] = rnd_array[0];
            good_p_indx[1] = rnd_array[1];
        }
    }
    
    //Push All Inliers Together To do Final Triangulation
    cv::Mat rnd_points_loc = myLinearTriangulation(points_loc_2d.at(good_p_indx[0]), points_loc_2d.at(good_p_indx[1]),rnd_P.at(good_p_indx[0]) , rnd_P.at(good_p_indx[1]));
    vector<cv::Mat> inlier_P;
    vector<Point2d> inlier_locs;
    for (int i=0; i<num_imgs; i++) {
        tmp_err = computeProjectPointError(rnd_P.at(i), points_loc_2d.at(i), rnd_points_loc); //Put All inliers, Including Random Array
      //  cout << "At Final Triangulation " << tmp_err << endl;
        if (tmp_err < error_thresh_hold) {
          //  tmp_inliers +=1; //Inliers Count
            inlier_P.push_back(rnd_P.at(i));
            inlier_locs.push_back(points_loc_2d.at(i));
        } else
        {
            cout << "Find outliers Value is" << tmp_err << endl;
        }
    }
    //Do multi Triangulations
    
    //Incase Inlierse Bigger Than 2
    if (inlier_locs.size() >= 2) {
        cv::Mat mat_reconstructed_point = myMulTriHelper(inlier_P, inlier_locs);
        //Get 4D location, Now do normalize to put it back to 3D world
        cv::Mat tmp_3d_coord = (mat_reconstructed_point.rowRange(0, 3))/(mat_reconstructed_point.at<double>(3,0));
        tmp_3d_coord.copyTo(mat_reconstructed_point);
        mat_reconstructed_point = mat_reconstructed_point.t();
        // cout << "mat_reconstructed " << mat_reconstructed_point << endl;
        return mat_reconstructed_point;
    }
    else {
        return threeD_point_loc.at(pointIdx); // Return Original Loc
    }
    
}

bool ransacTriangulation(cv::Mat P1, cv::Mat P2, vector<Point2d> vec_key_point_1, vector<Point2d> vec_key_point_2, cv::Mat  *triangulated_point)
{
    bool inlier = false;
    triangulatePoints(P1, P2, vec_key_point_1, vec_key_point_2, *triangulated_point);
    
    cv::Mat mat_tmp_point1 = P1*(*triangulated_point);
    cv::Mat mat_tmp_point2 = P2*(*triangulated_point);
    
    vector<Point2d> vec_point_1_hat,vec_point_2_hat;
    convertMat3Dto2d(mat_tmp_point1, &vec_point_1_hat);
    convertMat3Dto2d(mat_tmp_point2, &vec_point_2_hat);
    
  //  cout << "vec 1 pont hat" << vec_point_1_hat.at(0) << endl;
  //  cout << "vec 1 pont " << vec_key_point_1.at(0) << endl;

    Point2d tmp2d1 =  vec_key_point_1.at(0) - vec_point_1_hat.at(0);
    Point2d tmp2d2 = vec_key_point_2.at(0) - vec_point_2_hat.at(0);
    
    float err1 = pow(tmp2d1.x, 2) + pow(tmp2d2.y, 2);
    float err2 = pow(tmp2d2.x, 2) + pow(tmp2d2.y, 2);
    
    cout << "err  " <<  (err1 + err2) << endl;
    
    mat_tmp_point1.release();
    mat_tmp_point2.release();
    if ((err1 + err2) < 2e-08) {
        inlier = true;
    }
    return inlier;
}

void doTriangulation2Images(vector<cv::KeyPoint> K1,vector<cv::KeyPoint> K2, vector<DMatch> good_matches, cv::Mat img1, cv::Mat img2, cv::Mat P, cv::Mat P_prime)
{
    vector<Point2d> vec_match_point1, vec_match_point2;
    vector<Point2d> vec_match_point11, vec_match_point22;
    findMatchingPoint(K1, K2, good_matches, &vec_match_point1, &vec_match_point2,NORM);
    findMatchingPoint(K1, K2, good_matches, &vec_match_point11, &vec_match_point22,NON_NORM);

    //P and P_prime is Normalized (Without Camera Intrinsic Info)
   // cv::Mat KP = Kd*P;
   // cv::Mat KP_prime = Kd*P_prime;
    
    cv::Mat triangulated_point;
    triangulatePoints(P, P_prime, vec_match_point1, vec_match_point2, triangulated_point);
   // triangulated_point = Kd.inv() * triangulated_point;
   // cout << "triangulated points " << triangulated_point << endl;
    cv::Mat rgb_value(vec_match_point1.size(),1,CV_8UC3);
    for (int i=0; i<vec_match_point1.size(); i++) { //Interpolate The Pixel Value
        Vec3i tmp1 =  img1.at<Vec3b>(vec_match_point11.at(i));
        Vec3i tmp2 =  img2.at<Vec3b>(vec_match_point22.at(i));
        rgb_value.at<Vec3b>(i,0) = (tmp1 + tmp2)/2;
      //  cout << " (tmp1 + tmp2)/2" <<  (tmp1 + tmp2)/2 << endl;
    }
    
 //   ofstream myfile;
 //   myfile.open("./plyFiles/img1_img2.ply");
    for (int i=0; i<vec_match_point1.size(); i++) {
        cv::Mat tmp1 = ((triangulated_point.col(i)).t())/triangulated_point.at<double>(3,i);
//        myfile << tmp1.colRange(0, 3) << endl;
//        myfile << rgb_value.row(i) << endl;
      //  cout << "rgb_value " << rgb_value.row(i) << endl;
        //Fill In 3D Point Correspondance
        threeD_point_rgb.push_back(rgb_value.at<Vec3b>(i,0));
        cv::Mat tmpMat = (tmp1.colRange(0, 3));
        threeD_point_loc.push_back(tmpMat);
    }
    
 //   myfile.close();

  //  cout << "rgb value" << rgb_value << endl;
}

void doMulTriangulation(int imgIdx1, int imgIdx2, vector<DMatch> good_matches,  vector<KeyPoint> K1, vector<KeyPoint> K2, vector<int> inlier_mask)
{
    int num = good_matches.size();
    //Do 3D point Update After Knowing Pk, Only Do for inliers
    unordered_map<int, int>::iterator got;
    
    vector<KeyPoint> key1point;
    vector<KeyPoint> key2point;
    vector<Point2d> vec_key_point_1;
    vector<Point2d> vec_key_point_2;

    
    for (int i=0; i<num; i++) { //Good Mathes Number
        int querId = good_matches.at(i).queryIdx;// imgIdx1 Sift Keypoints
        int trainId = good_matches.at(i).trainIdx;//ImgIdx2 Sift Keypoints
        got = threeD_img2point_map[imgIdx1].find(querId);
        
        key1point.clear();
        key2point.clear();
        vec_key_point_1.clear();
        vec_key_point_2.clear();
        cv::Mat triangulated_point;
        cv::Mat rgb_value(1,1,CV_8UC3);
        
       // int point_idx = got->second;
        
       // if (got == threeD_img2point_map[imgIdx1].end()) { //Not find the point
            //Triangulate this point using Pk and previous image
        if (got->second >= threeD_point_loc.size()) { // Already Updated Points
        
            Point2f key1_p2f = K1.at(querId).pt;
            key1point.push_back(K1.at(querId));
            Point2f key2_p2f = K2.at(trainId).pt;
            key2point.push_back(K2.at(trainId));
            vec_key_point_1 = normCoord(key1point);
            vec_key_point_2 = normCoord(key2point);
          
            triangulatePoints(P[imgIdx1], P[imgIdx2], vec_key_point_1, vec_key_point_2, triangulated_point);
            
            Vec3i tmp1 =  src[imgIdx1].at<Vec3b>(key1_p2f);
            Vec3i tmp2 =  src[imgIdx2].at<Vec3b>(key2_p2f);
            rgb_value.at<Vec3b>(0,0) = (tmp1 + tmp2)/2;

          //  cout << "tmp1" << tmp1 << tmp2 <<  endl;
            
            cv::Mat tmp_mat_1 = ((triangulated_point.col(0)).t())/triangulated_point.at<double>(3,0);
            threeD_point_rgb.push_back(rgb_value.at<Vec3b>(0,0));
            
          //  cout << "rgb_value.at<Vec3b>(0,0)" << rgb_value.at<Vec3b>(0,0) << endl;
            cv::Mat tmpMat = (tmp_mat_1.colRange(0, 3));
            threeD_point_loc.push_back(tmpMat);
            tmpMat.release();
            tmp_mat_1.release();
        }else
        {
            cout << "Update Point " << got->second << endl;
            int point_idx = got->second;
        
            cv::Mat tmp_mat_update = myMulTriangulation(point_idx);
       //     cout << "Original 3D point Loc " << threeD_point_loc.at(point_idx) << endl;
            tmp_mat_update.copyTo(threeD_point_loc.at(point_idx));
            tmp_mat_update.release();
           
        }
        triangulated_point.release();
        rgb_value.release();
    }
    
}

void computeReconstructPoint(int imgIdx1, int imgIdx2, vector<DMatch> good_matches,vector<int> inliers)
{
    if (inliers.empty()) { //Set All good_matches as inliers
        int good_num = good_matches.size();
        for (int i=0; i<good_num; i++) {
            inliers.push_back(i); //All Good Matches Are Inliers
        }
    }
    
    int num = inliers.size();
    // got; //Find whether exiest point
    
    for (int i=0; i<num; i++) {
        int inlierId = inliers.at(i);
        int querId = good_matches.at(inlierId).queryIdx;
        int trainId = good_matches.at(inlierId).trainIdx;
       unordered_map<int, int>::const_iterator got = threeD_img2point_map[imgIdx1].find(querId);
        if (got == threeD_img2point_map[imgIdx1].end()) {//Not Construct this point
            //Add this point to point2img_map
            unordered_map<int, int> tmp_point2img_map;
            int tmp_points_reconstructed = threeD_point2img.size(); //Size is bigger than index by 1
            tmp_point2img_map.insert({imgIdx1,querId});//Insert Query Img1 Info
            tmp_point2img_map.insert({imgIdx2,trainId});
            threeD_point2img.push_back(tmp_point2img_map);//Add this point back to Reconstructed Point
            //Add this point to img2point_map
            threeD_img2point_map[imgIdx1].insert({querId,tmp_points_reconstructed});// Sift feature Indx to Point Indx
            threeD_img2point_map[imgIdx2].insert({trainId,tmp_points_reconstructed});// Sift feature Indx to Point Indx
        }else //Already Constructed this point
        { //Add Img2 Info to this point
            int tmp_this_point_indx = got->second; //This sift feature's point Indx
            threeD_img2point_map[imgIdx2].insert({trainId,tmp_this_point_indx});
            //Update This point
            threeD_point2img.at(tmp_this_point_indx).insert({imgIdx2,trainId});
            cout << "Update 3D point" << tmp_this_point_indx << endl;
        }
    }
}

void computePkUsing3D2D(int imgIdx1, int imgIdx2, vector<DMatch> good_matches, vector<KeyPoint> K1, vector<KeyPoint> K2, cv::Mat *Pk)
{
    //First Find Reconstructed Point
    int num = good_matches.size();
    vector<Point3f> vec_reconstructed_point;
    vector<Point2f> vec_img2_point;
    // got; //Find whether exiest point
    
    unordered_map<int, int>::const_iterator got;
    for (int i=0; i<num; i++) {
        int querId = good_matches.at(i).queryIdx;// imgIdx1 Sift Keypoints
        int trainId = good_matches.at(i).trainIdx;//ImgIdx2 Sift Keypoints
        got  = threeD_img2point_map[imgIdx1].find(querId);
        
        if (got == threeD_img2point_map[imgIdx1].end() ) { //Not find the point
            //Do nothing
        }else
        {
            int tmp_reconstructed_point_indx = got->second;// 3D point Indx
            cv::Mat tmp_mat = threeD_point_loc.at(tmp_reconstructed_point_indx);// Push back the 3D coordinate
            Point3f tmp_vec = {(float)tmp_mat.at<double>(0,0),(float)tmp_mat.at<double>(0,1),(float)tmp_mat.at<double>(0,2)};
         //   cout << "tmp_vec " << tmp_vec << endl;
            vec_reconstructed_point.push_back(tmp_vec);
            vec_img2_point.push_back(K2.at(trainId).pt);//Push Back img2 pixels loc
            tmp_mat.release();
        }
    }
    
    int threeDpoint_num = vec_img2_point.size();
    
    //Using Ransac To find Pk
    cv::Mat tmp_T;//(3,1,CV_64FC1);
    cv::Mat tmp_R;//(3,3,CV_64FC1);
    vector<int> inlier_mask;
   // mat_reconstructed_point.convertTo(vec_reconstructed_point, 5);
    solvePnPRansac(vec_reconstructed_point, vec_img2_point, K, noArray(), tmp_R, tmp_T);
    Rodrigues(tmp_R,tmp_R,noArray());
    cv::Mat tmp_Pk ;
    hconcat(tmp_R, tmp_T, tmp_Pk);
    //Get Pk
    tmp_Pk.copyTo(*Pk);
    cout << "tmpPk1" << tmp_Pk << endl;
    tmp_T.release();tmp_R.release();
    tmp_Pk.release();
    
    //Using Pk to compute Inliers
 /*   doRansacChooseInlier(vec_reconstructed_point, vec_img2_point, (*Pk), &inlier_mask);
    vector<Point3f> vec_reconstructed_point2;
    vector<Point2f> vec_img2_point2;
    cout << "inlier_mask.size() " << inlier_mask.size() << endl;
    for (int i=0; i<inlier_mask.size(); i++) {
        vec_reconstructed_point2.push_back(vec_reconstructed_point.at(inlier_mask.at(i)));
        vec_img2_point2.push_back(vec_img2_point.at(inlier_mask.at(i)));
    }
    cv::Mat tmp_T2;//(3,1,CV_64FC1);
    cv::Mat tmp_R2;//(3,3,CV_64FC1);
    solvePnPRansac(vec_reconstructed_point2, vec_img2_point2, K, noArray(), tmp_R2, tmp_T2);
    Rodrigues(tmp_R2,tmp_R2,noArray());
    hconcat(tmp_R2, tmp_T2, tmp_Pk);
    cout << "tmpPK2" << tmp_Pk << endl;
    //Get Pk
    tmp_Pk.copyTo(*Pk);
    tmp_T2.release();tmp_R2.release();
 //   inlier_mask.clear();// Clear Inlier, Don't Need
*/

    
    
    
 //   cout << "Inlier is " << inlier_mask << endl;
    
    //Update 3D point Map
    //computeReconstructPoint(imgIdx1, imgIdx2, good_matches, inlier_mask);
   // doMulTriangulation( imgIdx1,  imgIdx2, good_matches,  K1, K2, inlier_mask);
    
    computeReconstructPoint(imgIdx1, imgIdx2, good_matches, inlier_mask);
    doMulTriangulation(imgIdx1 , imgIdx2, good_matches, K1, K2, inlier_mask);

    
}

void doRansacChooseInlier(vector<Point3f> vec_reconstructed_point, vector<Point2f> vec_img2_point, cv::Mat Pk, vector<int> *inliers)
{
    cv::Mat mat_point(4,vec_reconstructed_point.size(),CV_64FC1);
    convertP3ftoMat(vec_reconstructed_point, &mat_point);
    cv::Mat mat_img2_point = Kd*Pk*mat_point;
    vector<Point2f> vec_img2_hat;
    convertMat3Dto2f(mat_img2_point, &vec_img2_hat);
    
    for (int i=0; i<vec_reconstructed_point.size() ; i++) {
    //    cout << "img_hat" << vec_img2_hat.at(i) << endl;
    //    cout << "img" << vec_img2_point.at(i) << endl;
    //    cout << "deduce " << (vec_img2_hat.at(i)-vec_img2_point.at(i)) << endl;
        Point2f tmp2f = (vec_img2_hat.at(i)-vec_img2_point.at(i));
        float judge = (tmp2f.x)*(tmp2f.x) + (tmp2f.y)*(tmp2f.y);
        if (judge < 2) {
            (*inliers).push_back(i);
         //   cout << "judge " << judge << endl;
        }else{
        //    cout << "outLiers " << endl;
        }
    }
    mat_point.release();
    mat_img2_point.release();
}

void testRT(cv::Mat R1, cv::Mat R2, cv::Mat T, cv::Mat *P2, vector<cv::KeyPoint> K1, vector<cv::KeyPoint> K2, std::vector<DMatch> good_matches){
    
    vector<Point2d> vec_match_point1, vec_match_point2;
    findMatchingPoint(K1, K2, good_matches, &vec_match_point1, &vec_match_point2,NORM);
    
    //   cout << "Point2  " << vec_match_point2 << endl;
    
    double tmp1[2] = {vec_match_point1[0].x,vec_match_point1[0].y};
    double tmp2[2] = {vec_match_point2[0].x,vec_match_point2[0].y};
    double test_point[4] = {0,0,1,1};
    cv::Mat mat_point1(1,1,CV_64FC2,tmp1);
    cv::Mat mat_point2(1,1,CV_64FC2,tmp2);
    cv::Mat mat_test_point(4,1,CV_64FC1,test_point);

    cv::Mat I = Mat::eye(3, 3, CV_64FC1);
    cv::Mat P;
    hconcat(I, cv::Mat(3,1,CV_64FC1,Scalar::all(0)), P);
    
    cv::Mat P_prime[4];
    hconcat(R1, T, P_prime[0]);
    hconcat(R2, T, P_prime[1]);
    hconcat(R1, -1*T, P_prime[2]);
    hconcat(R2, -1*T, P_prime[3]);
    
    for (int i=0; i<4; i++) {
        cv::Mat t_point;
        cv::Mat t_point_c2 = P_prime[i]*mat_test_point;
        if (t_point_c2.at<double>(2,0)>0) {
            triangulatePoints(P, P_prime[i], mat_point1, mat_point2, t_point);
            if ((t_point.at<double>(2,0)*t_point.at<double>(3, 0))>0) {
                (P_prime[i]).copyTo(*P2);
                cout << "P2 is" << *P2 << endl;
            }
        }
    }
}


/**
 * @function ransac
 * @choose largest number of inliers
 */
void ransac(vector<DMatch> good_matches, vector<DMatch>* inlier_matches, vector<KeyPoint> keypoint_1, vector<KeyPoint> keypoint_2)
{
    int max = 0;
    vector<DMatch> rnd_matches(4);
    for (int count=0; count< RANSAC_TIMES; count++) {
        int rnd_array[4];
        randomArray(4, (int)good_matches.size(), rnd_array);
        for (int i=0; i<4; i++) {
            rnd_matches[i] = good_matches[rnd_array[i]];
        }
        
        cv::Mat H_tmp = getTrans(keypoint_1, keypoint_2, rnd_matches);
        cv::Mat A = buildA(keypoint_1, keypoint_2, good_matches);
        cv::Mat A_H = A*(H_tmp.reshape(1,1)).t(); // 2n * 1
        A_H = A_H.mul(A_H);
        cv::Mat error((int)good_matches.size(),1,CV_32F);
        for (int i=0; i<good_matches.size(); i++) {
            error.at<float>(i,0) = A_H.at<float>(2*i, 0) + A_H.at<float>(2*i+1, 0);
        }
        
        // cout << "error are   " << endl << error << endl;
        
        double error_min, error_max;
        minMaxIdx(error.col(0), &error_min, &error_max);
        error = (error <= 1.50367e-03); // satisfied, return 255
        error /= 255;
        
        int inliers = sum(error)[0]; // channel zero
        
        if (inliers > max) {
            max = inliers;
            (*inlier_matches).clear();
            
            for (int i=0; i<good_matches.size(); i++) {
                if (error.at<uchar>(i,0) == 1) {
                    (*inlier_matches).push_back(good_matches[i]);
                }
            }
            
        }
        cout <<"Ransac Inliers  " <<  inliers << "Inliers total " <<(*inlier_matches).size() << endl;
        cout << "Rtio of inliers  " << float((*inlier_matches).size())/(float)good_matches.size() << endl;
    }
}
/**
 * @function buildA
 * @build Matrix A for DLT algorithm
 */

cv::Mat buildA(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> good_matches)
{
    // Creat Coordinate Matrix
    cv::Mat X_norm1;
    cv::Mat X_norm2;
    cv::Mat X_img1(3,(int)good_matches.size(),CV_32FC1);
    cv::Mat X_img2(3,(int)good_matches.size(),CV_32FC1);
    for (int i=0; i<(int)good_matches.size(); i++) {
        int id1 = good_matches[i].queryIdx;
        int id2 = good_matches[i].trainIdx;
        float x[3] = {keypoint_1[id1].pt.x,keypoint_1[id1].pt.y,1};
        float x_prime[3] = {keypoint_2[id2].pt.x,keypoint_2[id2].pt.y,1};
        cv::Mat(3,1,CV_32FC1,x).copyTo(X_img1.col(i));
        cv::Mat(3, 1, CV_32FC1, x_prime).copyTo(X_img2.col(i));
    }
    
    X_norm1 = norm_matrix*X_img1;
    X_norm2 = norm_matrix*X_img2;
    
    // Creat 8*9 Matrix, using 4 points
    cv::Mat A(2*(int)good_matches.size(),9,CV_32FC1,Scalar::all(0));
    for (int i=0; i<good_matches.size(); i++) {
        //cout <<(*X_norm1).col(i).t() << endl;
        A(Range(i*2,i*2+1),Range(3,6)) = -X_norm2.at<float>(2,i) * X_norm1.col(i).t();
        A(Range(i*2,i*2+1),Range(6,9)) = X_norm2.at<float>(1,i) * X_norm1.col(i).t();
        A(Range(i*2+1,i*2+2),Range(0,3)) = X_norm2.at<float>(2,i) * X_norm1.col(i).t();
        A(Range(i*2+1,i*2+2),Range(6,9)) = -X_norm2.at<float>(0,i) * X_norm1.col(i).t();
        
        //cout<< A(Range(i*2,i*2+2),Range(0,9)) << endl;
    }
    return A;
}


void matTransCVMat2Mate(cv::Mat *A, Mat3 *B, bool flag)
{
    if(flag){ //flag=1, CV::MAT -> Mate
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {
                (*B)(i,j) = (double)(*A).at<float>(i,j);
            }
        }
    }
    else{
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {
                (*A).at<float>(i,j) = (*B)(i,j);
            }
        }
    }
}


cv::Mat getTrans(std::vector<KeyPoint> keypoint_1, std::vector<KeyPoint> keypoint_2, std::vector<DMatch> good_matches)
{
    cv::Mat A = buildA(keypoint_1, keypoint_2, good_matches);
    SVD svd(A);
    cv::Mat h = svd.vt.t().col(svd.vt.rows - 1);
    cv::Mat H = h.t();
    H = H.reshape(1, 3);
    return H;
}

void getImgNSrcCoord(cv::Mat H, cv::Mat *img_coord, cv::Mat *out_img_coord)
{
    cv::Mat H_board_coord = coordTransX2Xprime(board_coord, H); // Refered Coordinate
    H_board_coord = coordCalib(H_board_coord,TO_EXTENDED_COORD); //Extended Coordinate
    
    double x_min,x_max,y_min,y_max;
    
    minMaxIdx(H_board_coord.row(0), &x_min, &x_max);
    minMaxIdx(H_board_coord.row(1), &y_min, &y_max);
    
    *out_img_coord = buildCoord(x_min, x_max, y_min,y_max);
    *img_coord = coordCalib(*out_img_coord, TO_REFERED_COORD);
    
    *img_coord = coordTransX2Xprime(*img_coord, H.inv());
    
}

cv::Mat buildCoord(int x_min, int x_max, int y_min, int y_max)
{
    int x_range = x_max - x_min;
    int y_range = y_max - y_min;
    cv::Mat coord = cv::Mat(3, x_range*y_range, CV_32FC1);
    for (int i=0; i<(int)y_range; i++) {
        for (int j=0; j<(int)x_range; j++) {
            coord.at<float>(0,i*x_range + j) = j + x_min;
            coord.at<float>(1,i*x_range + j) = i + y_min;
            coord.at<float>(2,i*x_range + j) = 1;
        }
    }
    return coord;
}

cv::Mat coordTransX2Xprime(cv::Mat x, cv::Mat H)
{
    cv::Mat x_prime(3,x.cols,CV_32FC1);
    x_prime = norm_matrix*x;//x_norm
    x_prime = H*x_prime;//h_x
    x_prime = norm_matrix.inv()*x_prime;
    for (int i=0; i<x_prime.cols; i++) {
        x_prime.col(i) /= x_prime.at<float>(2,i);
    }
    return x_prime;
}

cv::Mat coordCalib(cv::Mat x, bool flag)
{
    cv::Mat x_new;
    x.copyTo(x_new);
    if (flag) {
        x_new.row(0) += offset_cols;
        x_new.row(1) += offset_rows;
    }else
    {
        x_new.row(0) -= offset_cols;
        x_new.row(1) -= offset_rows;
    }
    return x_new;
}

void affineTrans(cv::Mat* s_image, cv::Mat* out_img, cv::Mat coord_source, cv::Mat coord_out)
{
    for (int i=0; i<coord_out.cols; i++) {
        int x = coord_out.at<float>(0,i);
        int y = coord_out.at<float>(1,i);
        int x2 = coord_source.at<float>(0,i);
        int y2 = coord_source.at<float>(1,i);
        if ( x2 >=0 && x2<1600 && y2 >=0 && y2<1200 ) {
            (*out_img).at<Vec3b>(y,x) = (*s_image).at<Vec3b>(y2,x2);
        }
    }
}

cv::Mat linearBlend(cv::Mat img1, cv::Mat img2)
{
    cv::Mat mask(img1.size(),CV_8UC1);
    cv::Mat mask2(img1.size(),CV_8UC1);
    cv::min(img1, img2, mask);
    mask2 = (mask != 0);
    mask2 /= 255;
    mask = (mask == 0);
    mask /= 255;
    mask *= 2;
    mask = mask + mask2;
    cv::Mat out_img(img1.size(),CV_8UC1);
    double alpha = 0.5;
    double beta = 1 - alpha;
    addWeighted(img1, alpha, img2, beta, 0, out_img);
    out_img = out_img.mul(mask);
    
    return out_img;
}

void randomArray(int size, int range, int *array)
{
    for (int i=0; i<size; i++) {
        array[i] = rand()%range;
        for (int y=0; y<i; y++) {
            if (array[i] == array[y]) {
                i--;
                break;
            }
        }
        
    }
}

