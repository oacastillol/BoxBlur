#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <cstdlib>
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
  //  cout << " P1 x " << unsigned(p1->x) <<" y " << unsigned(p1->y) <<" z " << unsigned(p1->z) << endl;
  //  cout << " P2 x " << unsigned(p2->x) <<" y " << unsigned(p2->y) <<" z " << unsigned(p2->z) << endl;
  return Pixel((p1->x+p2->x)/2,(p1->y+p2->y)/2,(p1->z+p2->z)/2);
}

Pixel sumKernel(Mat* m){
  // cout << "m (python)  = " << endl << format(*m, Formatter::FMT_PYTHON) << endl << endl;
  if (m->rows != m->cols || m->rows % 2 != 1 ){
    perror ("El kernel debe ser un número impar");
    exit(-1);
  }
  int  center = m->cols /2;
  Pixel suma, *tmp ;
  int x, y, z;
  x = 0;
  y = 0;
  z = 0;
  for (int r=0;r < m->rows; ++r ){
    for(int c=0;c < m->cols; ++c){
      if(r == 0 && c == 0){
	tmp = m->ptr<Pixel>(r,c);
	x = tmp->x;
	y = tmp->y;
	z = tmp->z;
	continue;
      }
      if(r == center && c == center)
	continue;
      tmp = m->ptr<Pixel>(r,c);
      x += tmp->x;
      y += tmp->y;
      z += tmp->z;
      //cout<<" r "<<r<<" c "<<c<<endl;
    }
  }
  int totalx = x/(m->rows * m->rows -1);
  int totaly = y/(m->rows * m->rows -1);
  int totalz = z/(m->rows * m->rows -1);
  // cout<<"total x: "<<totalx<<" total y: "<<totaly<<" total z: "<<totalz<<" divide: "<<(m->rows * m->rows -1);
  return Pixel(totalx,totaly,totalz);
}

Pixel sumTotalMat(Mat* m,int re,int ce){
  Pixel suma, *tmp ;
  int x, y, z;
  x = 0;
  y = 0;
  z = 0;
  for (int r=0;r < m->rows; ++r ){
    for(int c=0;c < m->cols; ++c){
      if(r == 0 && c == 0){
	tmp = m->ptr<Pixel>(r,c);
	x = tmp->x;
	y = tmp->y;
	z = tmp->z;
	continue;
      }
      if(r == re && c == ce)
	continue;
      tmp = m->ptr<Pixel>(r,c);
      x += tmp->x;
      y += tmp->y;
      z += tmp->z;
      //cout<<" r "<<r<<" c "<<c<<endl;
    }
  }
  int totalx = x/(m->rows * m->rows -1);
  int totaly = y/(m->rows * m->rows -1);
  int totalz = z/(m->rows * m->rows -1);
  // cout<<"total x: "<<totalx<<" total y: "<<totaly<<" total z: "<<totalz<<" divide: "<<(m->rows * m->rows -1);
  return Pixel(totalx,totaly,totalz);
}

int main(int argc, char *argv[])
{
  if ( argc != 3){
    cout<<"Se debe ejecutar como ./blur-effect <imageName.ext> NumeroKernel\n Donde NumeroKernel debe ser impar\n";
    return -1;
  }
  int kernel = atoi(argv[2]);;
  if (kernel % 2 != 1 ){
    cout<<"NumeroKernel debe ser un numero impar\n";
    return -1;
  }
  int mitad = kernel / 2;
  char* imageName = argv[1];  
  Mat original = imread(imageName);
  Mat copia = original.clone();
  //cout << "original (python)  = " << endl << format(original, Formatter::FMT_PYTHON) << endl << endl;
  //  cout << "copia (python)  = " << endl << format(copia, Formatter::FMT_PYTHON) << endl << endl;
  if (original.empty()){
    cout<<"No se pudo abrir la imagen "<<imageName<<endl;
    return -1;
  }else{
    cout << "cols = " << endl << " " << original.cols << endl << endl; 
    cout << "rows = " << endl << " " << original.rows << endl << endl; 
    for(int i = 0; i<original.rows;i++){
      int inicioRK = i-mitad;
      for(int j = 0; j<original.cols;j++){
	//cout<<"("<<i<<","<<j<<")"<<" ";
	if(i>=mitad && j >=mitad && i<original.rows-mitad && j<original.cols -mitad){
	  int inicioCK = j-mitad;
	  //Se usa para sustraer una submatriz operator()(RowRange,ColRange)
	  Mat subImagen = original.operator()(Range(inicioRK,inicioRK+kernel),Range(inicioCK,inicioCK+kernel));
	  Pixel final = sumKernel(&subImagen);
	  //	cout << " final x " << unsigned(final.x) <<" y " << unsigned(final.y) <<" z " << unsigned(final.z) << endl;
	  Pixel* ptr = copia.ptr<Pixel>(i, j);
	  subImagen.release();
	  ptr->x = final.x;
	  ptr->y = final.y;
	  ptr->z = final.z;
	}
      }
      //cout<<endl;
    }
    // cout << "copia (python)  = " << endl << format(copia, Formatter::FMT_PYTHON) << endl << endl;
    //cout <<"salida:"<<endl;
    //cout<<copia<<endl<<endl;
    namedWindow( imageName,WINDOW_NORMAL | WINDOW_KEEPRATIO );
    imshow(imageName,original);
    imwrite("copia.png",copia);
    namedWindow( "Blur-effect",WINDOW_NORMAL | WINDOW_KEEPRATIO );
    imshow("Blur-effect",copia);
    cvWaitKey(0);
    return 0;

  }
}
