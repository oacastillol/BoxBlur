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
int main(int argc, char *argv[])
{
  if ( argc != 2){
    printf("Se debe ejecutar como ./blur-effect <imageName.ext>\n");
    return -1;
  }
  char* imageName = argv[1];  
  Mat imagen = imread(imageName);
  Mat copia = imagen.clone();
  if (imagen.empty()){
    printf("No se pudo abrir la imagen %s\n",imageName);
    return -1;
  }else{
    cout << "Type = " << endl << " " << type2str(imagen.type()) << endl << endl;
    cout << "cols = " << endl << " " << imagen.cols << endl << endl;
    cout << "rows = " << endl << " " << imagen.rows << endl << endl;
    cout << " dims = " << endl << " " << imagen.dims << endl << endl;
    cout << "size = " << endl << " " << imagen.size << endl << endl;
    cout << "step = " << endl << " " << imagen.step << endl << endl;
    namedWindow( imageName,WINDOW_NORMAL | WINDOW_KEEPRATIO );
    imshow(imageName,imagen);
    cvWaitKey(0);
    for (int r = 0; r < imagen.rows; ++r) {
      Pixel* ptrOriginal = imagen.ptr<Pixel>(r, 0);
      Pixel* ptrOriginalRowUp;
      ptrOriginalRowUp = imagen.ptr<Pixel>(r, 0);
      if(r-1>=0)
	ptrOriginalRowUp = imagen.ptr<Pixel>(r-1, 0);
      Pixel* ptrOriginalRowDown;
      ptrOriginalRowDown = imagen.ptr<Pixel>(r, 0);
      if(r+1<imagen.rows)
	ptrOriginalRowDown = imagen.ptr<Pixel>(r+1, 0);
      Pixel* ptr = copia.ptr<Pixel>(r, 0);
      const Pixel* ptr_end = ptr + copia.cols;
      Pixel* ptrOriginalRDLeft;
      Pixel* ptrOriginalRDRigth;
      Pixel* ptrOriginalRULeft;
      Pixel* ptrOriginalRURigth;
      Pixel* ptrOriginalLeft;
      Pixel* ptrOriginalRigth;
      for (int  i = 0; ptr != ptr_end;++i, ++ptr,++ptrOriginal, ++ptrOriginalRowUp,++ptrOriginalRowDown) {
	if(i == 0){
	  ptrOriginalLeft=ptrOriginal;
	  ptrOriginalRDLeft=ptrOriginalRowDown;
	  ptrOriginalRULeft=ptrOriginalRowUp;
	}else{
	  ptrOriginalLeft=ptrOriginal-1;
	  ptrOriginalRDLeft=ptrOriginalRowDown-1;
	  ptrOriginalRULeft=ptrOriginalRowUp-1;
	}
	if(i == copia.cols-1){
	  ptrOriginalRigth=ptrOriginal;
	  ptrOriginalRDRigth = ptrOriginalRowDown;
	  ptrOriginalRURigth = ptrOriginalRowUp;
	}else{
	  ptrOriginalRigth=ptrOriginal+1;
	  ptrOriginalRDRigth = ptrOriginalRowDown+1;
	  ptrOriginalRURigth = ptrOriginalRowUp+1;
	}
	ptr->x = (ptrOriginalRowUp->x + ptrOriginalRowDown->x + ptrOriginalRURigth->x + ptrOriginalRULeft->x +ptrOriginalRDRigth->x + ptrOriginalRDLeft->x +  ptrOriginalRigth->x +  ptrOriginalLeft->x)/8;
	//cout << "x = " << endl << " " << ptr->x << " "<< ptrOriginal->x << endl << endl;
	ptr->y = (ptrOriginalRowUp->y + ptrOriginalRowDown->y + ptrOriginalRURigth->y + ptrOriginalRULeft->y +ptrOriginalRDRigth->y + ptrOriginalRDLeft->y +  ptrOriginalRigth->y +  ptrOriginalLeft->y)/8;
	//cout << "y = " << endl << " " << ptr->y << " "<< ptrOriginal->y << endl << endl;
	ptr->z = (ptrOriginalRowUp->z + ptrOriginalRowDown->z + ptrOriginalRURigth->z + ptrOriginalRULeft->z +ptrOriginalRDRigth->z + ptrOriginalRDLeft->z +  ptrOriginalRigth->z +  ptrOriginalLeft->z)/8;
	//cout << "z = " << endl << " " << ptr->z << " "<< ptrOriginal->z << endl << endl;
      }
    }
    namedWindow( imageName,WINDOW_NORMAL | WINDOW_KEEPRATIO );
    imshow(imageName,imagen);
    cvWaitKey(0);
    namedWindow( "copia",WINDOW_NORMAL | WINDOW_KEEPRATIO );
    imshow("copia",copia);
    cvWaitKey(0);
    return 0;
  }
}
