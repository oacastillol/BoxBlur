#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
typedef cv::Point3_<uint8_t> Pixel;
using namespace std;
using namespace cv;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}
Pixel averangePixel(Pixel* p1,Pixel* p2){
  //cout << " P1 x " << unsigned(p1->x) <<" y " << unsigned(p1->y) <<" z " << unsigned(p1->z) << endl;
  //cout << " P2 x " << unsigned(p2->x) <<" y " << unsigned(p2->y) <<" z " << unsigned(p2->z) << endl;
    return Pixel((p1->x+p2->x)/2,(p1->x+p2->x)/2,(p1->x+p2->x)/2);
}
Pixel sumKernel(Mat* m){
  if (m->rows != m->cols || m->rows % 2 != 1 ){
    perror ("El kernel debe ser un número impar");
    exit(-1);
  }
  int  center = m->cols /2;
  Pixel suma ;
  for (int r=0;r < m->rows; ++r ){
    for(int c=0;c < m->cols; ++c){
      if(r == 0 && c == 0)
	suma = *m->ptr<Pixel>(r,c);
      if(r == center && c == center)
	continue;
      suma = averangePixel(&suma,m->ptr<Pixel>(r,c));
      //cout<<" r "<<r<<" c "<<c<<endl;
    }
  }
  return suma;
}
int main(int argc, char *argv[])
{
  int kernel = 3;
  int mitad = kernel / 2;
  if ( argc != 2){
    cout<<"Se debe ejecutar como ./blur-effect <imageName.ext>\n";
    return -1;
  }
  char* imageName = argv[1];  
  Mat original = imread(imageName);
  Mat copia = original.clone();
  if (original.empty()){
    cout<<"No se pudo abrir la imagen "<<imageName<<endl;
    return -1;
  }else{
    cout << "cols = " << endl << " " << original.cols << endl << endl; 
    cout << "rows = " << endl << " " << original.rows << endl << endl; 
    for(int i = mitad; i<original.rows-mitad;i++){
      int inicioRK = i-mitad;
      Pixel* ptr = copia.ptr<Pixel>(i, 0);
      ++ptr;
      for(int j = mitad; j<original.cols-mitad;++ptr,j++){
	cout<<"("<<i<<","<<j<<")"<<" ";
	int inicioCK = j-mitad;
	//Se usa para sustraer una submatriz operator()(RowRange,ColRange)
	Mat subImagen = original.operator()(Range(inicioRK,inicioRK+kernel),Range(inicioRK,inicioRK+kernel));
	Pixel final = sumKernel(&subImagen);
	subImagen.release();
	ptr->x = final.x;
	ptr->y = final.y;
	ptr->z = final.z;
      }
      cout<<endl;
    }
    cout <<"salida:"<<endl;
    cout<<copia<<endl<<endl;
    namedWindow( imageName,WINDOW_NORMAL | WINDOW_KEEPRATIO );
    imshow(imageName,original);
    cvWaitKey(0);
    namedWindow( "Blur-effect",WINDOW_NORMAL | WINDOW_KEEPRATIO );
    imshow("Blur-effect",copia);
    cvWaitKey(0);
    return 0;

  }
}
