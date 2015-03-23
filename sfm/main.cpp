#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d.hpp"


#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ceres_optimization.h"
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;



/// Global variables
cv::Mat src[4], src_gray[4];
float k[3][3] = {{1077.9,0,594.0},{0,1077.9,393.3},{0,0,1}};
cv::Mat K(3,3,CV_32FC1,k); //Camera Matrix
cv::Mat Kd(3,3,CV_64FC1,Scalar::all(0));



const int num_pic = 2;

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
void goodMatches(cv::Mat descriptor1, cv::Mat descriptor2,std::vector<DMatch>* good_matches);
vector<Point2d>  normCoord(vector<cv::KeyPoint> keypoints);
void findMatchingPoint(vector<KeyPoint> K1, vector<KeyPoint> K2,vector<DMatch> good_matches, vector<Point2d> *match_point1, vector<Point2d> *match_point2, bool norm_flag);
cv::Mat findE(vector<cv::KeyPoint> keypoints1,vector<cv::KeyPoint> keypoints2,vector<DMatch> good_matches);
void convertP2DtoMat(vector<Point2d> point, cv::Mat *mat_point);
void convertP2DtoMat(Point2d point, cv::Mat *mat_point);
void convertVec2CrossMat(cv::Mat vec_origin, cv::Mat *cross_mat);

cv::Mat myLinearTriangulation(Point2d vec_point1, Point2d vec_point2,cv::Mat P, cv::Mat P_prime);
void testRT(cv::Mat R1, cv::Mat R2, cv::Mat T, cv::Mat *P2, vector<cv::KeyPoint> K1, vector<cv::KeyPoint> K2, std::vector<DMatch> good_matches);
cv::Mat findF(vector<cv::KeyPoint> keypoints1,vector<cv::KeyPoint> keypoints2,vector<DMatch> good_matches);

void drawEpilines(vector<Point2d> points1, vector<Point2d> points2, cv::Mat F, cv::Mat src1, cv::Mat src2);
void drawEpilinesHelper(vector<Point2d> points1, vector<Point2d> points2, cv::Mat F, cv::Mat *img1, cv::Mat *img2);

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
    cv::Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(1000,3,0.04,10,1.6);
    
    vector<cv::KeyPoint> keypoints[num_pic];
    cv::Mat mat_keypoints[num_pic];
    cv::Mat descriptor[num_pic];
    cv::Mat out_img[num_pic];
    for (int i=0; i<num_pic; i++) {
        f2d->detect(src_gray[i], keypoints[i]);
        f2d->compute(src_gray[i], keypoints[i], descriptor[i]);
    }
    
//    //Draw Sift Keypoints
//    Mat img_keypoints_1;
//    drawKeypoints(src_gray[0], keypoints[0], img_keypoints_1);
//    imshow("Keypoints 1", img_keypoints_1);
    
    
    vector<DMatch> good_matches;
    goodMatches(descriptor[0], descriptor[1],&good_matches);
    cout << "Good Matches  " << good_matches.size() << endl;
    
    //Find Fundamental Matrix
   // cv::Mat F = findF(keypoints[0], keypoints[1], good_matches);
   // cout << F << endl;
    
 //   vector<Point2d> vec_match_point1, vec_match_point2;
 //   findMatchingPoint(keypoints[0], keypoints[1], good_matches, &vec_match_point1, &vec_match_point2,NON_NORM);
    
    
    
    cv::Mat E = findE(keypoints[0], keypoints[1], good_matches);

    //  cv::Mat F = ((Kd.t()).inv())*E*(Kd.inv());
    
  //  cout << "F is " << F << endl;
  //  drawEpilines(vec_match_point1, vec_match_point2, F,src[0],src[1]);
  //  cout << "E  " << endl << E << endl;
    
    cv:Mat R1,R2,T,P2;
    decomposeEssentialMat(E, R1, R2, T);
    testRT(R1, R2, T, &P2, keypoints[0], keypoints[1], good_matches);
    
    
    Mat img_matches;
    drawMatches(src_gray[0], keypoints[0], src_gray[1], keypoints[1], good_matches, img_matches);
    
    resize(img_matches, img_matches, Size(img_matches.cols/2,img_matches.rows/2));
    imshow( "Good Matches", img_matches );
    
}


/**
 * @function goodMatches
 * @find Mathces
 */
void goodMatches(cv::Mat descriptor1, cv::Mat descriptor2,std::vector<DMatch> *good_matches)
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
    { if( matches[i].distance <= max(2*min_dist, 0.02) )
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

void convertP2DtoMat(vector<Point2d> point, cv::Mat *mat_point)
{
    int num = point.size();
    for (int i=0; i<num; i++) {
        (*mat_point).at<double>(0,i) = point.at(i).x;
        (*mat_point).at<double>(1,i) = point.at(i).y;
        (*mat_point).at<double>(2,i) = (double)1.0;
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

cv::Mat myLinearTriangulation(Point2d vec_point1, Point2d vec_point2,cv::Mat P, cv::Mat P_prime) //Using Non-Normalized Coordinate
{
    cv::Mat ho_point1(3,1,CV_64FC1);
    cv::Mat ho_point2(3,1,CV_64FC1);
    
    convertP2DtoMat(vec_point1,&ho_point1); //3*N
    convertP2DtoMat(vec_point2, &ho_point2); //3*N
    cv::Mat A1,A2,A;
    cv::Mat cross_mat1(3,3,CV_64FC1);
    cv::Mat cross_mat2(3,3,CV_64FC1);
    convertVec2CrossMat(ho_point1, &cross_mat1);
    convertVec2CrossMat(ho_point2, &cross_mat2);

    
    A1 = cross_mat1*P;
    A2 = cross_mat2*P;
    
    vconcat(A1, A2, A);
    cv::Mat w,u,vt;
    SVD::compute(A.t()*A, w, u, vt);
    cv::Mat X_hat((vt.t()).col(3));
    cout << "Vt " << vt.t().col(3) << endl;
    return X_hat;
}

void testRT(cv::Mat R1, cv::Mat R2, cv::Mat T, cv::Mat *P2, vector<cv::KeyPoint> K1, vector<cv::KeyPoint> K2, std::vector<DMatch> good_matches){
    
    vector<Point2d> vec_match_point1, vec_match_point2;
    findMatchingPoint(K1, K2, good_matches, &vec_match_point1, &vec_match_point2,NORM);
    
    //   cout << "Point2  " << vec_match_point2 << endl;
    
    double tmp1[2] = {vec_match_point1[0].x,vec_match_point1[0].y};
    double tmp2[2] = {vec_match_point2[0].x,vec_match_point2[0].y};
    cv::Mat mat_point1(1,1,CV_64FC2,tmp1);
    cv::Mat mat_point2(1,1,CV_64FC2,tmp2);

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
        triangulatePoints(P, P_prime[i], mat_point1, mat_point2, t_point);
        if ((t_point.at<double>(2,0)/t_point.at<double>(3, 0))>0) {
            cv::Mat t_point_c2;
            t_point_c2 = P_prime[i]*t_point;
            if (t_point_c2.at<double>(2, 0)>0) {
                (P_prime[i]).copyTo(*P2);
                cout << "P2 is" << *P2 << endl;
              //  cout << "t_point2 " << t_point_c2 <<endl;
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

