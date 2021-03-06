#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <iostream>
#include <omp.h>

using namespace std;
using namespace cv;

typedef cv::Point3_<uint8_t> Pixel;
//parametros que se enviaran para los hilos
struct thread_data {
  int thread_id;
  int xinicio, xfin;
  int yinicio, yfin;

};
// variables Globales
Mat original, copia;
int mitad, kernel;

Pixel sumKernel(Mat* m){
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
    }
  }
  int totalx = x/(m->rows * m->rows -1);
  int totaly = y/(m->rows * m->rows -1);
  int totalz = z/(m->rows * m->rows -1);
  return Pixel(totalx,totaly,totalz);
}

Pixel sumTotalMat(Mat* m){
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
      tmp = m->ptr<Pixel>(r,c);
      x += tmp->x;
      y += tmp->y;
      z += tmp->z;
    }
  }
  int totalx = x/(m->rows * m->cols);
  int totaly = y/(m->rows * m->cols);
  int totalz = z/(m->rows * m->cols);
  return Pixel(totalx,totaly,totalz);
}

void* blurCalculate(void *threadData){
  struct thread_data *data;
  data = (struct thread_data *) threadData;
  int xinicio, xfin;
  int yinicio, yfin;
  xinicio = data-> xinicio;
  xfin = data->xfin;
  yinicio = data->yinicio;
  yfin = data->yfin;
  for(int i = xinicio; i<xfin;i++){
      int inicioRK = i-mitad;
      int finRK = i+mitad;
      for(int j = yinicio; j<yfin;j++){
	//cout<<"("<<i<<","<<j<<")"<<" ";
	/* Si se encuentra dentro de los limites se calcula el kernel de manera normal  */
	/* excluyendo el centro del kernel y asignandole el promedio de sus bordes */
	/* pero si este se encuentra muy al borde no se puede calcular dentro de un kernel */
	/* cuadrado */
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
	}else{
	  /* Cuando el pixel se encuentra en el borde de la imagen se calcula el promedio de la matriz */
	  /* cuyo tamaño máximo es el tamaño del kernel, pero al no ser cuadrada la matriz se calcula el promedio */
	  /*   incluyendo el pixel de la posicion */
	  int inicioCK = j-mitad;
	  int finCK = j+mitad;
	  if (inicioRK<0)
	    inicioRK = 0;
	  if (inicioCK < 0)
	    inicioCK = 0;
	  if (finRK>=original.rows)
	    finRK = original.rows-1;
	  if (finCK >= original.cols)
	    finCK = original.cols-1;
	  //Se usa para sustraer una submatriz operator()(RowRange,ColRange)
	  Mat subImagen = original.operator()(Range(inicioRK,finRK),Range(inicioCK,finCK));
	  Pixel final = sumTotalMat(&subImagen);
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

}

// Crea bloques coordenados para ser asignados a cada uno de los hilos.
/*  rows: filas de la imágen original
    cols: columnas de la imagen original
    thrds: cantidad de hilos a usar
    thread_data_array: coordenadas x,y iniciales y finales para cada hilo
*/
void block(int rows, int cols, int thrds, thread_data thread_data_array[]){
  int fact[100];    //Almacenar factores de "thrds".
  int i_fact=0;     //Recorrer el arreglo "fact[]".
  int i=2;          //Verificar los factores desde 2.
  int r = 1;        //Cantidad de filas
  int c = 1;        //Cantidad de columnas
  int rs, cs;       //Separacion de filas y columnas

  while(i<=thrds){
    if((thrds%i)==0){ //a%b=0, implica que b es factor de a.
      fact[i_fact]=i; //Añadimos factor al arreglo.
      thrds=thrds/i;  //Procesamos variable "thrds".
      i_fact++;       //Incrementamos indice.
      continue;
    }
    i++;              //Incrementamos indice.
  }

  for(i=0; i<i_fact; i++){ //Calcula 2 factores balanceados r*c=thrds
    if(i%2==0)
      r*=fact[i];
    else
      c*=fact[i];
  }

  rs=rows/r;
  cs=cols/c;
  thrds=0;

  for(int x=0;x<c;x++){
    for(int y=0;y<r;y++){
      thread_data_array[thrds].thread_id = thrds;
      thread_data_array[thrds].xinicio = x*cs;
      thread_data_array[thrds].xfin = (x+1)*cs;
      thread_data_array[thrds].yinicio = y*rs;
      thread_data_array[thrds].yfin = (y+1)*rs;
      thrds++;
    }
  }
}


int main(int argc, char *argv[]){
  if ( argc < 3 || argc > 4){
    cout<<"Se debe ejecutar como ./blur-effect <imageName.ext> NumeroKernel <cantidadHilos>\n Donde NumeroKernel debe ser impar y\ncantidadHilosdebe ser menor a 16, por defecto es 1\n";
    return -1;
  }
  kernel = atoi(argv[2]);
  int nThread = 1;
  if (kernel % 2 != 1 ){
    cout<<"NumeroKernel debe ser un numero impar\n";
    return -1;
  }
  if ( argc == 4){
    int nThread = atoi(argv[3]);
    if (nThread > 16 || nThread<1){
      cout<<"El número de hilos debe estar entre 1 y 16";
      return -1;
    }
  }
  mitad = kernel / 2;
  char* imageName = argv[1];
  original = imread(imageName);
  copia = original.clone();
  //cout << "original (python)  = " << endl << format(original, Formatter::FMT_PYTHON) << endl << endl;
  //  cout << "copia (python)  = " << endl << format(copia, Formatter::FMT_PYTHON) << endl << endl;
  if (original.empty()){
    cout<<"No se pudo abrir la imagen "<<imageName<<endl;
    return -1;
  }else{
    //cout << "cols = " << endl << " " << original.cols << endl << endl;
    //cout << "rows = " << endl << " " << original.rows << endl << endl;
    struct thread_data  thread_data_array[nThread];
    omp_set_num_threads(nThread);
    
    // cout << "copia (python)  = " << endl << format(copia, Formatter::FMT_PYTHON) << endl << endl;
    //cout <<"salida:"<<endl;
    //cout<<copia<<endl<<endl;
    block(original.cols,original.rows,nThread,thread_data_array);
#pragma omp parallel
    {
      int tm = omp_get_thread_num();
      blurCalculate((void *)&thread_data_array[tm]);
    }
    //namedWindow( imageName,WINDOW_NORMAL | WINDOW_KEEPRATIO );
    //imshow(imageName,original);
    char str[100];
    strcpy(str,"blur-");
    strcat(str,imageName);
    imwrite(str,copia);
    //namedWindow( "Blur-effect",WINDOW_NORMAL | WINDOW_KEEPRATIO );
    //imshow("Blur-effect",copia);
    //cvWaitKey(0);
    return 0;

  }
}
