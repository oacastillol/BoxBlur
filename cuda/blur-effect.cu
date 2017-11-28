#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <iostream>

using namespace std;
using namespace cv;

typedef struct {
    int r,c;
}tuple;
Mat original, copia;
int kernel;

__global__ void blurCalculate(unsigned char *dataIn,unsigned char *dataOut,int filas,int columnas,int pasoFila,int tamaElement,int kernel,int r,int c){
    const unsigned int posc = blockDim.x * blockIdx.x + threadIdx.x;
    int mitad = kernel/2;
    int xinicio=0;
    int xfin;
    int yinicio;
    int yfin;
    int pasof=0;
    int rs=filas/r;
    int cs=columnas/c;
    for(int x=0;x<c;x++){
        for(int y=0;y<r;y++){
            if(posc==pasof){
                xinicio = x*cs;
                xfin = (x+1)*cs;
                yinicio = y*rs;
                yfin = (y+1)*rs;
            }
            pasof++;
        }
    }
    int kernelXI,kernelYI;
    int kernelXF,kernelYF;
    int totalPix =0;
    for(int ff = yinicio;ff < yfin;ff++){
        for(int cf = xinicio;cf < xfin;cf++){
            int blue=0,green=0,red=0,pasoi=0;
            pasof = (pasoFila * ff)+(tamaElement * cf);
            kernelXI=cf-mitad;
            if(kernelXI<xinicio)kernelXI=xinicio;
            kernelXF=cf+mitad;
            if(kernelXF>xfin)kernelXF=xfin;
            kernelYI=ff-mitad;
            if(kernelYI<yinicio)kernelYI=yinicio;    
            kernelYF=ff+mitad;
            if(kernelYF>yfin)kernelYF=yfin;
            for(int fi=kernelYI;fi<kernelYF;fi++){
                for(int ci=kernelXI;ci<kernelXF;ci++){
                    pasoi=(pasoFila*fi)+(tamaElement*ci);
                    blue += dataIn[pasoi ] ;
                    green += dataIn[pasoi + 1];
                    red += dataIn[pasoi + 2];
                }
            }
            totalPix=((kernelYF-kernelYI)*(kernelXF-kernelXI));
            blue=blue/totalPix;
            green=green/totalPix;
            red=red/totalPix;
            dataOut[pasof ] = blue;
            dataOut[pasof + 1 ] = green;
            dataOut[pasof + 2 ] = red;
        }
    }
}

// Crea bloques coordenados para ser asignados a cada uno de los hilos.
/*  rows: filas de la imágen original
cols: columnas de la imagen original
thrds: cantidad de hilos a usar
thread_data_array: coordenadas x,y iniciales y finales para cada hilo
 */
tuple block(int rows, int cols, int thrds){
    int fact[100];    //Almacenar factores de "thrds".
    int i_fact=0;     //Recorrer el arreglo "fact[]".
    int i=2;          //Verificar los factores desde 2.
    int r = 1;        //Cantidad de filas
    int c = 1;        //Cantidad de columnas

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

    thrds=0;
    tuple ret={r,c};
    return  ret;
}


int main(int argc, char *argv[]){
    if ( argc < 3 || argc > 4){
        cout<<"Se debe ejecutar como ./blur-effect <imageName.ext> NumeroKernel <cantidadHilos>\n Donde NumeroKernel debe ser impar y\ncantidadHilosdebe ser menor a 16, por defecto es 1\n";
        return -1;
    }
    kernel = atoi(argv[2]);
    int nThread = 1;
    int blocks = 2;
    if (kernel % 2 != 1 ){
        cout<<"NumeroKernel debe ser un numero impar\n";
        return -1;
    }
    if ( argc == 4){
        nThread = atoi(argv[3]);
        if (nThread > 16 || nThread<1){
            cout<<"El número de hilos debe estar entre 1 y 16";
            return -1;
        }
    }
    int totalThread = nThread*blocks;
    char* imageName = argv[1];
    original = imread(imageName);
    copia = original.clone();
    if (original.empty()){
        cout<<"No se pudo abrir la imagen "<<imageName<<endl;
        return -1;
    }else{
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp,0);
        unsigned long sizeTotal = original.total()*original.elemSize();
        unsigned char *dataIn = (unsigned char*)(original.data);
        unsigned char *dataOut;
        unsigned char *dataInDev;
        unsigned char *dataOutDev;
        int columnas = original.cols;
        int filas = original.rows;
        int pasoFila = original.cols * original.elemSize();
        int tamaElement = original.elemSize();
        dataOut = (unsigned char*) malloc(sizeTotal);
        cudaMalloc((void**)&dataInDev,sizeTotal);
        cudaMalloc((void**)&dataOutDev,sizeTotal);
        // cout << "copia (python)  = " << endl << format(copia, Formatter::FMT_PYTHON) << endl << endl;
        //cout <<"salida:"<<endl;
        //cout<<copia<<endl<<endl;
        tuple stepCut=block(original.cols,original.rows,totalThread);
        //printf("%i - %i\n",stepCut.r,stepCut.c);
        cudaMemcpy(dataInDev,dataIn,sizeTotal,cudaMemcpyHostToDevice);
        blurCalculate<<<blocks,nThread>>>(dataInDev,dataOutDev,filas,columnas,pasoFila,tamaElement,kernel,stepCut.r,stepCut.c);
        cudaMemcpy(dataOut,dataOutDev,sizeTotal,cudaMemcpyDeviceToHost);

        copia = Mat(original.rows,original.cols,CV_8UC3,dataOut);
        cudaFree(dataInDev);
        cudaFree(dataOutDev);

        //      int tm = omp_get_thread_num();
        //      blurCalculate<<blocks,thread>>(dataInDev,dataOutDev,(void *)&thread_data_array[tm]);
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
