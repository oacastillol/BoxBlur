# BoxBlur
 Box Blur (desenfoque de cuadro), es la forma mas sencilla de aplicación del efecto borroso, el Box Blur es una aproximación al efecto de desenfoque gaussiano. Para realizar el efecto borroso, se copia la imagen y se divide en filas y columnas, luego se procesa cada punto, tomando en cuenta los datos cercanos, definidos por el  *kernel*, que se ha asignado al momento de ejecutar, entonces se leen estos puntos de la imagen original, se promedian separando cada uno de sus 3 canales (R,G,B), generando así el nuevo punto, el cual se asigna a la posición en la imagen copiada, este proceso se repite para todos los puntos que componen la imagen.

 En este repositorio se encuentra la implementación de este algoritmo usando [posix Threads](https://en.wikipedia.org/wiki/Native_POSIX_Thread_Library), [openMP](https://en.wikipedia.org/wiki/OpenMP) y [CUDA](https://en.wikipedia.org/wiki/CUDA). Para el manejo de imagenes se utilizo [OpenCV](https://en.wikipedia.org/wiki/OpenCV). Trabajo realizado para la materia Computación paralela y distribuida.

## Tabla de contenido

## Instalación 

Estas implementaciones se realizaron usando [ubuntu-16.04](https://www.ubuntu.com/) se necesito instalar los siguientes paquetes:


* **OpenCV** basados en el siguiente [tutorial](http://milq.github.io/install-opencv-ubuntu-debian/) usamos 
	```sudo apt-get install libopencv-dev python-opencv``` 
* **CUDA** nos basamos en la [guía de instalación](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) realizada por nvidia. En la instalación de este paquete se debe tener en cuenta la versión, pues no todas las tarjetas son soportadas por la ultima versión.

## Uso 

Para hacer uso del repositorio dentro de la [wiki](https://github.com/oacastillol/BoxBlur/wiki) del repositorio se va a explicar detalladamente cada una.

## Contribución

## Creditos
Este repositorio se realizo gracias a la contribución de :
	
	
* [Osmar Alejandro Castillo Lancheros](https://github.com/oacastillol)
* [Camilo Andres Pinilla Bocanegra](https://github.com/capinillab)

## Licenciamiento
Este repositorio y lo que contiene puede ser editado, compartido y distribuido. Gracias a que se encuentra bajo a la licencia GNU General Public License v3.0 para mayor información puede revisar el archvio [LICENSE.md](LICENSE.md)
