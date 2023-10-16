///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
//В строках "в районе" 30-й - начальные параметры. В строках около 150-й нужно указать путь к файлам с функциями(либо положить в ту же директорию, где основной, и убрать путь к ним)
//если компилятор не знает M_PI, то вернуть строку (с дефайном MPI) из комментария
//Если у компа меньше 64 гигов оперативы(48, на самом деле, тоже должно подойти), то лучше не запускать с сетками больше 256
//Сетки с размерами, не являющимися степенями двойки, могут работать, но НЕ ТЕСТИРОВАЛИСЬ
//Директория для выходных файлов задается в константах
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_error_hadling.h"
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <ctime>
#include <fstream>
#include <cstring>

//#define M_PI 3.1415926535897932384626433832795
#define Gl 0.01
#define nu 0.025
#define eta 0.025
#define N 256
#define h 2.0*M_PI / double(N)
#define Cfl  0.1
#define Cfl2 0.1 //цфл поглощения, < 1/8
#define Omx 0.0 // омега икс
#define Omy 0.0 // омега игрек
#define Omz 0.0 // омега зет
#define B0x 0.1
#define B0y 0.0
#define B0z 0.0
#define lambda (double(N)*double(N)*double(N))
#define kmin 1  //левая граница начального спектра
#define kmax 10  //правая граница начального спектра
#define EForse 0.0001 //энергия, вливаемая в dtmax во всем промежутке по k(в случае с нелинейным спектром - энергия до свертки)
#define dtmax 0.01
using namespace std;

cudaError_t cuerr;
const char OutputDirectory[80] = "/mnt/data2/256/";//директория
const double Time = 40;//расчетное время
const bool vivodphis = 0;//выводить или нет
const bool vivodspec = 1;//выводить или нет
const bool vivodhelicity = 1;//выводить или нет
const bool useforcing = 0;//форсинг(КОД ПОД ФОРСИНГ ЕСТЬ, НО НЕ ТЕСТИРОВАЛСЯ, могут быть косяки)
const int numberofpreviousforcing = 5000;
double dt = 1;//тут число просто чтобы если что-то не так, увидеть, что он не меняет время
double Nk = 0.1;//шаг времени вывода спектров
double dNk = Nk;
double Nk2 = 1.0;//шаг времени вывода скорости
double dNk2 = Nk2;
int NNk = 10;//это для нумерации выходных файлов
int NNk2 = 10;
int NNN = 0;//количество шагов по времени


cufftDoubleReal *UX, *UY, *UZ;//в физdevice
cufftDoubleReal *h_UX, *h_UY, *h_UZ;//в физhost

cufftDoubleComplex *UXv, *UYv, *UZv;//в фурье пространстве device
cufftDoubleComplex *h_UXv, *h_UYv, *h_UZv;//в фурье пространстве host

cufftDoubleComplex *dUv;//в фурье(временные переменные и для поля и для скоростей)

cufftDoubleReal *d1U, *d2U, *d3U, *d4U, *d5U, *d6U;//в физ(временные переменные и для поля и для скоростей) device
cufftDoubleReal *h_d2U,*h_d3U,*h_d4U,*h_d5U, *h_d6U;//в физ(временные переменные и для поля и для скоростей) host

cufftDoubleReal *AiD;//в физ(временные переменные)

cufftDoubleComplex *AXv, *AYv, *AZv;//в фурье(временные переменные) device
cufftDoubleComplex *h_AXv, *h_AYv, *h_AZv;//в фурье(временные переменные) host

cufftDoubleComplex *P;//давление в фурье device
cufftDoubleComplex *h_P;//давление в фурье host

cufftDoubleComplex *F;//правая часть уравнения device
cufftDoubleComplex *h_F;//правая часть уравнения host

cufftDoubleReal *TCH; //dt,cfl,h в видеопамяти
cufftDoubleReal *tch; //dt,cfl,h в оперативе

cufftDoubleComplex *U1, *U2, *U3; //временные переменные(промежуточные значения) device
cufftDoubleComplex *h_U1, *h_U2, *h_U3; //временные переменные(промежуточные значения) host

cufftDoubleComplex *U1n, *U2n, *U3n; //временные переменные(конечные значения на шаге) device
cufftDoubleComplex *h_U1n, *h_U2n, *h_U3n; //временные переменные(конечные значения на шаге) host

cufftDoubleComplex *bx, *by, *bz;//(в оперативе)

cufftDoubleReal *BX, *BY, *BZ;//в физ device
cufftDoubleReal *h_BX, *h_BY, *h_BZ;//в физ host

cufftDoubleComplex *BXv, *BYv, *BZv;//в фурье пространстве device
cufftDoubleComplex *h_BXv, *h_BYv, *h_BZv;//в фурье пространстве host

cufftDoubleComplex *DXv, *DYv, *DZv;//в фурье(временные переменные) device
cufftDoubleComplex *h_DXv, *h_DYv, *h_DZv;//в фурье(временные переменные) host

cufftDoubleComplex *B1, *B2, *B3; //временные переменные(промежуточные значения)device
cufftDoubleComplex *h_B1, *h_B2, *h_B3; //временные переменные(промежуточные значения)host

cufftDoubleComplex *B1n, *B2n, *B3n; //временные переменные(конечные значения на шаге)device
cufftDoubleComplex *h_B1n, *h_B2n, *h_B3n; //временные переменные(конечные значения на шаге)host
const int csize = N*N*N * sizeof(cufftDoubleReal);
const int csizeIm = N*N*(N/2+1) * sizeof(cufftDoubleComplex);
FILE*	out1;
FILE*	out2;
FILE*	out3;

int *WN;
int *Kf;//оператива
cufftDoubleReal *abc;//оператива
int num1, num2;
cufftDoubleReal phix, phiy, phiz, Estep;
int kx, ky, kz;
double kmod;
cufftDoubleReal ABC1, ABC2, ABC3;

cufftDoubleReal *ex, *ey, *ez;
cufftDoubleReal *ebx, *eby, *ebz;
cufftDoubleReal *Ukx, *Uky, *Ukz;//под энергию(3дспектр) device
cufftDoubleReal *h_Ukx, *h_Uky, *h_Ukz;//host
cufftDoubleReal *ukx, *uky, *ukz;//под энергию(3дспектр) в оперативе
cufftDoubleReal *bkx, *bky, *bkz;//под энергию(3дспектр) в оперативе

int kxyz;


cufftHandle plan;
cufftHandle planinverse;

const int K1 = 8;
const int K2 = K1;
const int K3 = K1;

dim3 dimBlock(K1, K2, K3);//Размер блока
dim3 dimGrid(N / K1, N / K2, N / K3); //количество блоков для действительных ядер
dim3 dimGridZ(N / K1, N / K2, N / (2 * K3));//количество юлоков для комплексных ядер



///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
//ТУТ НАДО УКАЗАТЬ ВЕРНЫЕ ПУТИ К ФАЙЛАМ, в них лежат функции, без них нифига работать не будет
//как вариант, можно кинуть файлы в ту же папку, где лежит основной и убрать пути к ним, вроде как, работает(но зависит от компилятора, насколько я помню)
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

#include "/home/yst/data/FunctionsNelin.h"
#include "/home/yst/data/FunctionsForce.h"
#include "/home/yst/data/Helicity_Functions.h"


int main()
{

				///////////////////////////////////////////////
				// информация о всех CUDA GPU в системе
				///////////////////////////////////////////////
//	setlocale(LC_ALL, "Russian");
	//cudaDeviceProp count;
	int c;
	cudaGetDeviceCount(&c);
	/*cout << "Количество GPU: " << c << endl;
	for (int i = 0; i < c; i++)
	{
		cudaGetDeviceProperties(&count, 0);
		cout << "GPU №" << i + 1 << ": " << count.name << endl;
		cout << "Тактовая частота: " << count.clockRate / 1000 << " ГГц" << endl;
		cout << "Глобальная память(всего): " << count.totalGlobalMem / 1024.0 / 1024.0 << " МБ" << endl;
		cout << "Константная память(всего): " << count.totalConstMem / 1024.0 << " КБ" << endl;
		cout << "Количество мультипроцессоров: " << count.multiProcessorCount << endl;
		cout << "Разделяемая память на один МП: " << count.sharedMemPerBlock / 1024.0 << " КБ" << endl;
		cout << "Нитей в варпе: " << count.warpSize << endl;
		cout << "Макс количество нитей в блоке: " << count.maxThreadsPerBlock << endl;
		cout << "Макс количество нитей по измерениям: " << count.maxThreadsDim[0] << ' ' << count.maxThreadsDim[1] << ' ' << count.maxThreadsDim[2] << endl;
		cout << "Максимальные размеры сетки: " << count.maxGridSize[0] << ' ' << count.maxGridSize[1] << ' ' << count.maxGridSize[2] << endl;
		cout << "Может ли отображать память CPU на пространство CUDA-устройства: " << count.canMapHostMemory << endl;
		cout << "Является ли интегрированным: " << count.integrated << endl;

	}*/
	//cout << "Введите номер выбранного для расчетов GPU (n<=" << c <<")"<< endl;
	//cin >> c;
	c = 0; //пока что по умолчанию гпу=1
		   //выбираем GPU
	cudaSetDevice(c);


				///////////////////////////////////////////////
				//А тут уже основной кусок проги
				///////////////////////////////////////////////
	double T = 0;
	unsigned int start_time = clock()/CLOCKS_PER_SEC;// начальное время ввода данных
	srand(time(0));//включаем рандомизацию

				///////////////////////////////////////////////
				//создаём план
				///////////////////////////////////////////////

	cufftPlan3d(&plan, N, N, N, CUFFT_D2Z);
	cufftPlan3d(&planinverse, N, N, N, CUFFT_Z2D);



	ios_base::sync_with_stdio(0);
	char file[160];
	char dir[10];

	unsigned int end_time = clock()/CLOCKS_PER_SEC;//конечное время ввода данных
	//cout << "download data in " << (end_time - start_time)<< " sek" << endl;


				///////////////////////////////////////////////
				//расчёт всех возможных вариантов волновых чисел(естественно, квадратов волновых чисел)
				///////////////////////////////////////////////
	int Nl = 0; bool dn = 1; int si2;

	WN = (int *)malloc(N*N*N * sizeof(int));

	for (int ix = 0; ix < N / 3; ix += 1)
		for (int iy = 0; iy < N / 3; iy += 1)
			for (int iz = 0; iz < N / 3; iz += 1)
			{
				si2 = ix*ix + iy*iy + iz*iz;
				dn = 1;
				for (int i = 0; i < Nl; i++) if ((WN[i] == si2)||(si2>= (N / 3)*(N / 3))) dn = 0;
				if (dn)
				{
					WN[Nl] = si2;
					Nl++;
				}
			}

	qsort(WN, Nl, sizeof(int), compare);
	
	strcpy(file, OutputDirectory);
	strncat(file, "K.dat", 20);
	out1 = fopen(file, "w+");
	for (int i = 0; i < Nl; i++)
	{
		fprintf(out1, "%i\n", WN[i]);
	}
	fclose(out1);

	cudaHostAlloc((void **)&h_Ukx,
				   N*N*(N / 2 + 1) * sizeof(cufftDoubleReal),
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&Ukx,h_Ukx,0);

	cudaHostAlloc((void **)&h_Uky,
				   N*N*(N / 2 + 1) * sizeof(cufftDoubleReal),
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&Uky,h_Uky,0);

	cudaHostAlloc((void **)&h_Ukz,
				   N*N*(N / 2 + 1) * sizeof(cufftDoubleReal),
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&Ukz,h_Ukz,0);


	ukx = (cufftDoubleReal *)malloc(N*N*(N / 2 + 1) * sizeof(cufftDoubleReal));
	uky = (cufftDoubleReal *)malloc(N*N*(N / 2 + 1) * sizeof(cufftDoubleReal));
	ukz = (cufftDoubleReal *)malloc(N*N*(N / 2 + 1) * sizeof(cufftDoubleReal));
	bkx = (cufftDoubleReal *)malloc(N*N*(N / 2 + 1) * sizeof(cufftDoubleReal));
	bky = (cufftDoubleReal *)malloc(N*N*(N / 2 + 1) * sizeof(cufftDoubleReal));
	bkz = (cufftDoubleReal *)malloc(N*N*(N / 2 + 1) * sizeof(cufftDoubleReal));
	ex = (cufftDoubleReal *)malloc(Nl * sizeof(cufftDoubleReal));
	ey = (cufftDoubleReal *)malloc(Nl * sizeof(cufftDoubleReal));
	ez = (cufftDoubleReal *)malloc(Nl * sizeof(cufftDoubleReal));
	ebx = (cufftDoubleReal *)malloc(Nl * sizeof(cufftDoubleReal));
	eby = (cufftDoubleReal *)malloc(Nl * sizeof(cufftDoubleReal));
	ebz = (cufftDoubleReal *)malloc(Nl * sizeof(cufftDoubleReal));
	int id1, id2, id3, id4;

				///////////////////////////////////////////////
				//расчёт всех возможных вариантов волновых чисел k_x, k_y, k_z для задания начальных условий(в пределах k_x, k_y, k_z)
				///////////////////////////////////////////////

	int Nf = 0;
	double imax = kmax*sqrt(3);
	for (int ix = 1; ix < imax; ix += 1)
		for (int iy = 1; iy < imax; iy += 1)
			for (int iz = 1; iz < imax; iz += 1)
				if ((ix*ix + iy*iy + iz*iz >= kmin*kmin) && (ix*ix + iy*iy + iz*iz <= kmax*kmax)) Nf++;

	Kf = (int *)malloc((3 * Nf) * sizeof(int));
	abc = (cufftDoubleReal *)malloc((3) * sizeof(cufftDoubleReal));

	Nf = 0;
	for (int ix = 1; ix < imax; ix += 1)
		for (int iy = 1; iy < imax; iy += 1)
			for (int iz = 1; iz < imax; iz += 1)
				if ((ix*ix + iy*iy + iz*iz >= kmin*kmin) && (ix*ix + iy*iy + iz*iz <= kmax*kmax))
				{
					Kf[3 * Nf] = ix;
					Kf[3 * Nf + 1] = iy;
					Kf[3 * Nf + 2] = iz;
					Nf++;
				}

	printf("made data for making specters great again\n");


				///////////////////////////////////////////////
				//если N=256, мы храним в памяти видяхи, если больше, используем оперативку
				///////////////////////////////////////////////
	if (N>256) {
	cudaHostAlloc((void **)&h_UX, csize,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&UX,h_UX,0);

	cudaHostAlloc((void **)&h_UY, csize,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&UY,h_UY,0);

	cudaHostAlloc((void **)&h_UZ, csize,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&UZ,h_UZ,0);

	cudaHostAlloc((void **)&h_UXv, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&UXv,h_UXv,0);

	cudaHostAlloc((void **)&h_UYv, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&UYv,h_UYv,0);

	cudaHostAlloc((void **)&h_UZv, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&UZv,h_UZv,0);

	cudaHostAlloc((void **)&h_BX, csize,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&BX,h_BX,0);

	cudaHostAlloc((void **)&h_BY, csize,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&BY,h_BY,0);

	cudaHostAlloc((void **)&h_BZ, csize,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&BZ,h_BZ,0);

	cudaHostAlloc((void **)&h_BXv, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&BXv,h_BXv,0);

	cudaHostAlloc((void **)&h_BYv, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&BYv,h_BYv,0);

	cudaHostAlloc((void **)&h_BZv, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&BZv,h_BZv,0);


	//кусок кода под всякие там разные сервисные переменные
	cudaMalloc((void **)&dUv, csizeIm);

	cudaHostAlloc((void **)&h_AXv, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&AXv,h_AXv,0);

	cudaHostAlloc((void **)&h_AYv, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&AYv,h_AYv,0);

	cudaHostAlloc((void **)&h_AZv, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&AZv,h_AZv,0);

	cudaMalloc((void **)&d1U, csize);

	//вернуть, если памяти хватит
    //cudaMalloc((void **)&d2U, csize);
	cudaHostAlloc((void **)&h_d2U, csize,
		cudaHostAllocWriteCombined |
		cudaHostAllocMapped);
	cudaHostGetDevicePointer(&d2U, h_d2U, 0);

	cudaHostAlloc((void **)&h_d3U, csize,
		cudaHostAllocWriteCombined |
		cudaHostAllocMapped);
	cudaHostGetDevicePointer(&d3U, h_d3U, 0);

	cudaHostAlloc((void **)&h_d4U, csize,
		cudaHostAllocWriteCombined |
		cudaHostAllocMapped);
	cudaHostGetDevicePointer(&d4U, h_d4U, 0);

	cudaHostAlloc((void **)&h_d5U, csize,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&d5U,h_d5U,0);

	cudaHostAlloc((void **)&h_d6U, csize,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&d6U,h_d6U,0);

	cudaMalloc((void **)&AiD, csize);

	cudaHostAlloc((void **)&h_P, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&P,h_P,0);

	cudaHostAlloc((void **)&h_F, csizeIm,
		cudaHostAllocWriteCombined |
		cudaHostAllocMapped);
	cudaHostGetDevicePointer(&F, h_F, 0);

	cudaHostAlloc((void **)&h_U1, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&U1,h_U1,0);

	cudaHostAlloc((void **)&h_U2, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&U2,h_U2,0);

	cudaHostAlloc((void **)&h_U3, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&U3,h_U3,0);

	cudaHostAlloc((void **)&h_U1n, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);

	cudaHostGetDevicePointer(&U1n,h_U1n,0);
	cudaHostAlloc((void **)&h_U2n, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&U2n,h_U2n,0);

	cudaHostAlloc((void **)&h_U3n, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&U3n,h_U3n,0);

	cudaHostAlloc((void **)&h_B1, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&B1,h_B1,0);

	cudaHostAlloc((void **)&h_B2, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&B2,h_B2,0);

	cudaHostAlloc((void **)&h_B3, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&B3,h_B3,0);

	cudaHostAlloc((void **)&h_B1n, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&B1n,h_B1n,0);

	cudaHostAlloc((void **)&h_B2n, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&B2n,h_B2n,0);

	cudaHostAlloc((void **)&h_B3n, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&B3n,h_B3n,0);

	cudaMalloc((void **)&TCH, 3 * sizeof(cufftDoubleReal));
}
else
{

	h_UX=(cufftDoubleReal *)malloc(csize);
	cudaMalloc((void **)&UX,csize);
	h_UY=(cufftDoubleReal *)malloc(csize);
	cudaMalloc((void **)&UY,csize);
	h_UZ=(cufftDoubleReal *)malloc(csize);
	cudaMalloc((void **)&UZ,csize);

	cudaMalloc((void **)&UXv,csizeIm);
	cudaMalloc((void **)&UYv,csizeIm);
	cudaMalloc((void **)&UZv,csizeIm);

	h_BX=(cufftDoubleReal *)malloc(csize);
	cudaMalloc((void **)&BX,csize);
	h_BY=(cufftDoubleReal *)malloc(csize);
	cudaMalloc((void **)&BY,csize);
	h_BZ=(cufftDoubleReal *)malloc(csize);
	cudaMalloc((void **)&BZ,csize);

	cudaMalloc((void **)&BXv,csizeIm);
	cudaMalloc((void **)&BYv,csizeIm);
	cudaMalloc((void **)&BZv,csizeIm);

	//кусок кода под всякие там разные сервисные переменные
	cudaMalloc((void **)&dUv, csizeIm);

	cudaMalloc((void **)&AXv, csizeIm);
	cudaMalloc((void **)&AYv, csizeIm);
	cudaMalloc((void **)&AZv, csizeIm);

	cudaMalloc((void **)&d1U, csize);
    cudaMalloc((void **)&d2U, csize);
    cudaMalloc((void **)&d3U, csize);
    cudaMalloc((void **)&d4U, csize);
    cudaMalloc((void **)&d5U, csize);
    cudaMalloc((void **)&d6U, csize);

	cudaMalloc((void **)&AiD, csize);

	cudaMalloc((void **)&P, csizeIm);
	cudaMalloc((void **)&F, csizeIm);


	cudaMalloc((void **)&U1, csizeIm);
	cudaMalloc((void **)&U2, csizeIm);
	cudaMalloc((void **)&U3, csizeIm);

	cudaMalloc((void **)&U1n, csizeIm);
	cudaMalloc((void **)&U2n, csizeIm);
	cudaMalloc((void **)&U3n, csizeIm);


	cudaMalloc((void **)&B1, csizeIm);
	cudaMalloc((void **)&B2, csizeIm);
	cudaMalloc((void **)&B3, csizeIm);

	cudaMalloc((void **)&B1n, csizeIm);
	cudaMalloc((void **)&B2n, csizeIm);
	cudaMalloc((void **)&B3n, csizeIm);

	cudaMalloc((void **)&TCH, 3 * sizeof(cufftDoubleReal));
}
	tch = (cufftDoubleReal *)malloc(3 * sizeof(cufftDoubleReal));
	dt = 0.;//чтобы вышел сразу же, если не посчитает dt
	tch[0] = dt;
	tch[1] = Cfl;
	tch[2] = h;
	cudaMemcpy(TCH, tch, 3 * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);
	if (N>256) {
	cudaHostAlloc((void **)&h_DXv, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&DXv,h_DXv,0);

	cudaHostAlloc((void **)&h_DYv, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&DYv,h_DYv,0);

	cudaHostAlloc((void **)&h_DZv, csizeIm,
				   cudaHostAllocWriteCombined |
				   cudaHostAllocMapped);
	cudaHostGetDevicePointer(&DZv,h_DZv,0);
}
else
{

	cudaMalloc((void **)&DXv, csizeIm);
	cudaMalloc((void **)&DYv, csizeIm);
	cudaMalloc((void **)&DZv, csizeIm);

}
	NullREAL <<<dimGrid, dimBlock >>> (BX);
	NullREAL <<<dimGrid, dimBlock >>> (BY);
	NullREAL <<<dimGrid, dimBlock >>> (BZ);
	NullREAL <<<dimGrid, dimBlock >>> (UX);
	NullREAL <<<dimGrid, dimBlock >>> (UY);
	NullREAL <<<dimGrid, dimBlock >>> (UZ);

	//задаём в фурье
	cufftExecD2Z(plan, BX, BXv);
	cufftExecD2Z(plan, BY, BYv);
	cufftExecD2Z(plan, BZ, BZv);

	cudaDeviceSynchronize();


				///////////////////////////////////////////////
				//задание начальных условий
				///////////////////////////////////////////////

	if (numberofpreviousforcing>0) for(int i=0;i<numberofpreviousforcing;i++)
	{
		phix = 2 * M_PI*double(rand()) / double(RAND_MAX);
		phiy = 2 * M_PI*double(rand()) / double(RAND_MAX);
		phiz = 2 * M_PI*double(rand()) / double(RAND_MAX);
		num1 = rand() % Nf;//число определяющее группу векторов(4 симметричных вектора), на которых форсируем
		num2 = rand() % 4;//определяем, kx,ky или -kx,ky или kx,-ky  или -kx,-ky
		if (num2 == 0) {
			kx = Kf[3 * num1];
			ky = Kf[3 * num1 + 1];
			kz = Kf[3 * num1 + 2];
		}
		else if (num2 == 1) {
			kx = -Kf[3 * num1];
			ky = Kf[3 * num1 + 1];
			kz = Kf[3 * num1 + 2];
		}
		else if (num2 == 2) {
			kx = Kf[3 * num1];
			ky = -Kf[3 * num1 + 1];
			kz = Kf[3 * num1 + 2];
		}
		else if (num2 == 3) {
			kx = -Kf[3 * num1];
			ky = -Kf[3 * num1 + 1];
			kz = Kf[3 * num1 + 2];
		}
			//гаусс, сигма=0.5, медиана - среднее арифм он Кмакс и Кмин
		//	double sigma=0.7;
		// kmod=sqrt(double(kx)*double(kx) + double(ky)*double(ky) + double(kz)*double(kz));
		// Estep = 100.0 * EForse / kmod * exp(-(kmod-0.5*double(kmax+kmin))*(kmod-0.5*double(kmax+kmin))/(2.0*sigma*sigma))/(sigma*sqrt(2.0*double(M_PI)));



		//плоский спектр
		Estep = 100.0 * EForse / sqrt(double(kx)*double(kx) + double(ky)*double(ky) + double(kz)*double(kz));

		//~k^(-2)
		//kmod=sqrt(double(kx)*double(kx) + double(ky)*double(ky) + double(kz)*double(kz));
		//Estep = 100.0 * (EForse / kmod) / (kmod*kmod);

		//~k^(-3/2?)
		//kmod=sqrt(double(kx)*double(kx) + double(ky)*double(ky) + double(kz)*double(kz));
		//Estep = 100.0 * (EForse / kmod) / (kmod*sqrt(kmod));



		FindABC(kx, ky, kz, phix, phiy, phiz, Estep, abc);//(kx,ky,kz,phi1,phi2,phi3,Eforcing,ABC)

		if (i%50==0) printf("A = %f B = %f C = %f n = %d \n",abc[0],abc[1],abc[2],i);

		/*cout << numofforsing[0] << " " << numofforsing[1] << " " << numofforsing[2] << endl;																															cout << forseandphi[0] << " " << forseandphi[1] << " " << forseandphi[2] << " " << forseandphi[3] << endl;*/
		ABC1 = abc[0];
		ABC2 = abc[1];
		ABC3 = abc[2];

		Forcing << <dimGrid, dimBlock >> > (AiD, UX, UY, UZ, kx, ky, kz, phix, phiy, phiz, ABC1, ABC2, ABC3);

		cudaDeviceSynchronize();

	}

				///////////////////////////////////////////////
				//вывод начальных условий
				///////////////////////////////////////////////

	cufftExecD2Z(plan, UX, UXv);
	cufftExecD2Z(plan, UY, UYv);
	cufftExecD2Z(plan, UZ, UZv);

	cudaDeviceSynchronize();

				///////////////////////////////////////////////
				//вывод в физ пространстве
				///////////////////////////////////////////////
if (vivodphis)
	{
		start_time = clock()/CLOCKS_PER_SEC;

		
		strcpy(file, OutputDirectory);
		strncat(file, "UXout0.dat", 80);
		out1 = fopen(file, "w+");

		strcpy(file, OutputDirectory);
		strncat(file, "UYout0.dat",80);
		out2 = fopen(file, "w+");

		strcpy(file, OutputDirectory);
		strncat(file, "UZout0.dat",80);
		out3 = fopen(file, "w+");

        if (N<=256) {
			cudaMemcpy(h_UX, UX, csize, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_UY, UY, csize, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_UZ, UZ, csize, cudaMemcpyDeviceToHost);
			}
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++) {
				for (int k = 0; k < N; k++)
				{
					fprintf(out1, "%.16e ", h_UX[(i*N + j)*N + k]);
					fprintf(out2, "%.16e ", h_UY[(i*N + j)*N + k]);
					fprintf(out3, "%.16e ", h_UZ[(i*N + j)*N + k]);
				}
				fprintf(out1, "\n");
				fprintf(out2, "\n");
				fprintf(out3, "\n");
			}

		fclose(out1);
		fclose(out2);
		fclose(out3);

		strcpy(file, OutputDirectory);
		strncat(file, "BXout0.dat",80);
		out1 = fopen(file, "w+");

		strcpy(file, OutputDirectory);
		strncat(file, "BYout0.dat",80);
		out2 = fopen(file, "w+");

		strcpy(file, OutputDirectory);
		strncat(file, "BZout0.dat",80);
		out3 = fopen(file, "w+");

         if (N<=256) {
			cudaMemcpy(h_BX, BX, csize, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_BY, BY, csize, cudaMemcpyDeviceToHost);
			cudaMemcpy(h_BZ, BZ, csize, cudaMemcpyDeviceToHost);
			}
		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
			{
				for (int k = 0; k < N; k++)
				{
					fprintf(out1, "%.16e ", h_BX[(i*N + j)*N + k]);
					fprintf(out2, "%.16e ", h_BY[(i*N + j)*N + k]);
					fprintf(out3, "%.16e ", h_BZ[(i*N + j)*N + k]);
				}
				fprintf(out1, "\n");
				fprintf(out2, "\n");
				fprintf(out3, "\n");
			}

		fclose(out1);
		fclose(out2);
		fclose(out3);

		end_time = clock()/CLOCKS_PER_SEC;//конечное время ввода данных

		cout << "Raw data transfer to disk made in " << double(end_time - start_time) << " sek" << endl;
	}


				///////////////////////////////////////////////
				//вывод спектров
				///////////////////////////////////////////////

	if (vivodspec)
	{
		start_time = clock()/CLOCKS_PER_SEC;
		Sumforen << <dimGridZ, dimBlock >> > (UXv, Ukx);
		Sumforen << <dimGridZ, dimBlock >> > (UYv, Uky);
		Sumforen << <dimGridZ, dimBlock >> > (UZv, Ukz);
		cudaDeviceSynchronize();

		cudaMemcpy(ukx, h_Ukx, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);
		cudaMemcpy(uky, h_Uky, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);
		cudaMemcpy(ukz, h_Ukz, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);

		cudaDeviceSynchronize();
		Sumforen << <dimGridZ, dimBlock >> > (BXv, Ukx);
		Sumforen << <dimGridZ, dimBlock >> > (BYv, Uky);
		Sumforen << <dimGridZ, dimBlock >> > (BZv, Ukz);
		cudaDeviceSynchronize();

		cudaMemcpy(bkx, h_Ukx, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);
		cudaMemcpy(bky, h_Uky, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);
		cudaMemcpy(bkz, h_Ukz, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);

		cudaDeviceSynchronize();
		Null1(ex, Nl);
		Null1(ey, Nl);
		Null1(ez, Nl);
		Null1(ebx, Nl);
		Null1(eby, Nl);
		Null1(ebz, Nl);

		for (int i = 0; i < Nl; i++)
			for (int ix = 0; ix < N / 3; ix += 1)
				for (int iy = 0; iy < N / 3; iy += 1)
					for (int iz = 0; iz < N / 3; iz += 1)
						if (WN[i] == ix*ix + iy*iy + iz*iz)
						{
							id1 = (ix*N + iy)*(N / 2 + 1) + iz;
							id2 = ((-ix + N)*N + iy)*(N / 2 + 1) + iz;
							id3 = (ix*N + (-iy + N))*(N / 2 + 1) + iz;
							id4 = ((-ix + N)*N + (-iy + N))*(N / 2 + 1) + iz;
							ex[i] += ukx[id1] + ukx[id2] + ukx[id3] + ukx[id4];
							ey[i] += uky[id1] + uky[id2] + uky[id3] + uky[id4];
							ez[i] += ukz[id1] + ukz[id2] + ukz[id3] + ukz[id4];
							ebx[i] += bkx[id1] + bkx[id2] + bkx[id3] + bkx[id4];
							eby[i] += bky[id1] + bky[id2] + bky[id3] + bky[id4];
							ebz[i] += bkz[id1] + bkz[id2] + bkz[id3] + bkz[id4];
						}

		ex[0] /= 4.0;
		ey[0] /= 4.0;
		ez[0] /= 4.0;
		ebx[0] /= 4.0;
		eby[0] /= 4.0;
		ebz[0] /= 4.0;
		
		strcpy(file, OutputDirectory);
		strncat(file, "UXsp0.dat",80);
		out1 = fopen(file, "w+");

		strcpy(file, OutputDirectory);
		strncat(file, "UYsp0.dat",80);
		out2 = fopen(file, "w+");

		strcpy(file, OutputDirectory);
		strncat(file, "UZsp0.dat",80);
		out3 = fopen(file, "w+");

		for (int i = 0; i < Nl; i++)
		{
			fprintf(out1, "%.8e\n", ex[i]);
		}
		for (int i = 0; i < Nl; i++)
		{
			fprintf(out2, "%.8e\n", ey[i]);
		}
		for (int i = 0; i < Nl; i++)
		{
			fprintf(out3, "%.8e\n", ez[i]);
		}
		fclose(out1);
		fclose(out2);
		fclose(out3);

		strcpy(file, OutputDirectory);
		strncat(file, "BXsp0.dat",80);
		out1 = fopen(file, "w+");

		strcpy(file, OutputDirectory);
		strncat(file, "BYsp0.dat",80);
		out2 = fopen(file, "w+");

		strcpy(file, OutputDirectory);
		strncat(file, "BZsp0.dat",80);
		out3 = fopen(file, "w+");

		for (int i = 0; i < Nl; i++)
		{
			fprintf(out1, "%.8e\n", ebx[i]);
		}
		for (int i = 0; i < Nl; i++)
		{
			fprintf(out2, "%.8e\n", eby[i]);
		}
		for (int i = 0; i < Nl; i++)
		{
			fprintf(out3, "%.8e\n", ebz[i]);
		}
		fclose(out1);
		fclose(out2);
		fclose(out3);


		end_time = clock()/CLOCKS_PER_SEC;//конечное время ввода данных
		cout << "Calculation of spectres and data transfer to disk made in " << double(end_time - start_time) << " sek" << endl;
	}
				///////////////////////////////////////////////
				//вывод спиральностей
				///////////////////////////////////////////////


	if (vivodhelicity)
	{
		start_time = clock()/CLOCKS_PER_SEC;
				///////////////////////////////////////////////
				// расчет перекрестной спиральности
				///////////////////////////////////////////////
		CutN3 << <dimGridZ, dimBlock >> > (UXv);
		CutN3 << <dimGridZ, dimBlock >> > (UYv);
		CutN3 << <dimGridZ, dimBlock >> > (UZv);
				cudaDeviceSynchronize();

		cufftExecZ2D(planinverse, UXv, UX);
		cufftExecZ2D(planinverse, UYv, UY);
		cufftExecZ2D(planinverse, UZv, UZ);
				cudaDeviceSynchronize();

		CutN3 << <dimGridZ, dimBlock >> > (BXv);
		CutN3 << <dimGridZ, dimBlock >> > (BYv);
		CutN3 << <dimGridZ, dimBlock >> > (BZv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, BXv, BX);
		cufftExecZ2D(planinverse, BYv, BY);
		cufftExecZ2D(planinverse, BZv, BZ);
				cudaDeviceSynchronize();


	CrossHelicity <<< dimGrid, dimBlock >>> (UX, UY, UZ, BX, BY, BZ, AiD);
				cudaDeviceSynchronize();
	cufftExecD2Z(plan, AiD, DXv);
				cudaDeviceSynchronize();

				///////////////////////////////////////////////
				//расчет спиральност
				///////////////////////////////////////////////

				///////////////////////////////////////////////
				//иксовая компонента(производные, умножаемые на соответствующую компоненту Ux)
				///////////////////////////////////////////////
		Ddy << < dimGridZ, dimBlock >> > (UZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (UYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();

				///////////////////////////////////////////////
				//игрековая компонента(производные, умножаемые на соответствующую компоненту Ux)
				///////////////////////////////////////////////
		Ddz << < dimGridZ, dimBlock >> > (UXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (UZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();

				///////////////////////////////////////////////
				//зетовая компонента(производные, умножаемые на соответствующую компоненту Ux)
				///////////////////////////////////////////////
		Ddx << < dimGridZ, dimBlock >> > (UYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d5U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (UXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d6U);
				cudaDeviceSynchronize();

		Helicity	<<< dimGrid, dimBlock >>> (UX, UY, UZ, d1U, d2U, d3U, d4U, d5U, d6U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, AXv);
				cudaDeviceSynchronize();


				///////////////////////////////////////////////
				//вывод и расчет спектров
				///////////////////////////////////////////////


		Sumforen << <dimGridZ, dimBlock >> > (AXv, Ukx);
		Sumforen << <dimGridZ, dimBlock >> > (DXv, Uky);
		cudaDeviceSynchronize();

		cudaMemcpy(ukx, h_Ukx, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);
		cudaMemcpy(uky, h_Uky, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);

		cudaDeviceSynchronize();

		Null1(ex, Nl);
		Null1(ey, Nl);


		for (int i = 0; i < Nl; i++)
			for (int ix = 0; ix < N / 3; ix += 1)
				for (int iy = 0; iy < N / 3; iy += 1)
					for (int iz = 0; iz < N / 3; iz += 1)
						if (WN[i] == ix*ix + iy*iy + iz*iz)
						{
							id1 = (ix*N + iy)*(N / 2 + 1) + iz;
							id2 = ((-ix + N)*N + iy)*(N / 2 + 1) + iz;
							id3 = (ix*N + (-iy + N))*(N / 2 + 1) + iz;
							id4 = ((-ix + N)*N + (-iy + N))*(N / 2 + 1) + iz;
							ex[i] += ukx[id1] + ukx[id2] + ukx[id3] + ukx[id4];
							ey[i] += uky[id1] + uky[id2] + uky[id3] + uky[id4];
						}

		ex[0] /= 4.0;
		ey[0] /= 4.0;

		strcpy(file, OutputDirectory);
		strncat(file, "HelicitySP0.dat",80);
		out1 = fopen(file, "w+");

		strcpy(file, OutputDirectory);
		strncat(file, "CrossHelicitySP0.dat",80);
		out2 = fopen(file, "w+");


		for (int i = 0; i < Nl; i++)
		{
			fprintf(out1, "%.8e\n", ex[i]);
		}
		for (int i = 0; i < Nl; i++)
		{
			fprintf(out2, "%.8e\n", ey[i]);
		}

		fclose(out1);
		fclose(out2);


		end_time = clock()/CLOCKS_PER_SEC;//конечное время ввода данных
		cout << "Calculation of helicity spectra and data transfer to disk made in " << double(end_time - start_time) << " sek" << endl;
	}

	int start_time2 = clock()/CLOCKS_PER_SEC;
	int end_time2;
	while (T <= Time)
	{
		//Считаем нелинейную матрицу A
		//компоненты UX,UY,UZ

		cufftExecZ2D(planinverse, UXv, UX);
		cufftExecZ2D(planinverse, UYv, UY);
		cufftExecZ2D(planinverse, UZv, UZ);
				cudaDeviceSynchronize();

		// считаем производные UX, UY, UZ, необходимые для AX и перекидываем в физ пр-во
		Ddx << < dimGridZ, dimBlock >> > (UYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (UXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (UXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (UZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();


		//собственно AX
		nelinAx << < dimGrid, dimBlock >> > (UY, UZ, d1U, d2U, d3U, d4U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, AXv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (AXv);

		//аналогично для AY
		Ddy << < dimGridZ, dimBlock >> > (UZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (UYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (UYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (UXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();

		nelinAy << < dimGrid, dimBlock >> > (UZ, UX, d1U, d2U, d3U, d4U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, AYv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (AYv);
				cudaDeviceSynchronize();
		//аналогично для AZ
		Ddz << < dimGridZ, dimBlock >> > (UXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (UZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (UZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (UYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();

		nelinAz << < dimGrid, dimBlock >> > (UX, UY, d1U, d2U, d3U, d4U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, AZv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (AZv);
				cudaDeviceSynchronize();



		//Аналогичные расчёты нелинейного члена D(чисто магнитного)
		//DX
		CutN3 << <dimGridZ, dimBlock >> > (BXv);
		CutN3 << <dimGridZ, dimBlock >> > (BYv);
		CutN3 << <dimGridZ, dimBlock >> > (BZv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, BXv, BX);
		cufftExecZ2D(planinverse, BYv, BY);
		cufftExecZ2D(planinverse, BZv, BZ);
				cudaDeviceSynchronize();

		Ddx << < dimGridZ, dimBlock >> > (BYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (BXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (BXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (BZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();


		nelinDx << < dimGrid, dimBlock >> > (BY, BZ, d1U, d2U, d3U, d4U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, DXv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (DXv);
				cudaDeviceSynchronize();

		////////////////////////////////////////////////////////////////////////////////////////////////////////
		//DY
		Ddy << < dimGridZ, dimBlock >> > (BZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (BYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (BYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (BXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();

		nelinDy << < dimGrid, dimBlock >> > (BZ, BX, d1U, d2U, d3U, d4U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, DYv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (DYv);
				cudaDeviceSynchronize();

		////////////////////////////////////////////////////////////////////////////////////////////////////////
		// DZ
		Ddz << < dimGridZ, dimBlock >> > (BXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (BZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (BZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (BYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();

		nelinDz << < dimGrid, dimBlock >> > (BX, BY, d1U, d2U, d3U, d4U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, DZv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (DZv);
				cudaDeviceSynchronize();

		//дальше надо посчитать давление ч/з лапласа

		lapP << <dimGridZ, dimBlock >> > (AXv, AYv, AZv, DXv, DYv, DZv, P);

				cudaDeviceSynchronize();

		AbsZn << <dimGrid, dimBlock >> > (UX, UY, UZ); //работаем с модулями, ищем их в этой процедурe
				cudaDeviceSynchronize();

		DT << <dimGrid, dimBlock >> > (UX, UY, UZ, TCH); //алгоритм поиска максимального значения на куде(редукция) и высчитывание dt
				cudaDeviceSynchronize();


		cudaMemcpy(tch, TCH, 3 * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
		if (tch[0] > dtmax) tch[0] = dtmax;
		if (tch[0] > Cfl2*h*h / nu) tch[0] = Cfl2*h*h / nu;
		dt = tch[0];
		AbsZnB << <dimGrid, dimBlock >> > (BX, BY, BZ);
				cudaDeviceSynchronize();
		DT << <dimGrid, dimBlock >> > (BX, BY, BZ, TCH);
				cudaDeviceSynchronize();
		cudaMemcpy(tch, TCH, 3 * sizeof(cufftDoubleReal), cudaMemcpyDeviceToHost);
		if (tch[0] > dt) tch[0] = dt;
		printf("%lf %lf \n", tch[0], T + tch[0]);
		cudaMemcpy(TCH, tch, 3 * sizeof(cufftDoubleReal), cudaMemcpyHostToDevice);


		//осталось только посчитать F(правая часть уравнения)

		Fx << <dimGridZ, dimBlock >> > (AXv, UXv, UYv, UZv, P, F, DXv);
				cudaDeviceSynchronize();
		// тут считается мацуно для Fx

		Mazuno << <dimGridZ, dimBlock >> > (UXv, U1, F, TCH);
				cudaDeviceSynchronize();

				Multiple << <dimGridZ, dimBlock >> > (U1, U1n);
				cudaDeviceSynchronize();

		Mazuno << <dimGridZ, dimBlock >> > (U1n, U1, F, TCH);
				cudaDeviceSynchronize();

		//теперь считаем Fy и мацуно для Fy

		Fy << <dimGridZ, dimBlock >> > (AYv, UXv, UYv, UZv, P, F, DYv);
				cudaDeviceSynchronize();

		// мацуно Fy

		Mazuno << <dimGridZ, dimBlock >> > (UYv, U2, F, TCH);
				cudaDeviceSynchronize();

				Multiple << <dimGridZ, dimBlock >> > (U2, U2n);

				cudaDeviceSynchronize();
		Mazuno << <dimGridZ, dimBlock >> > (U2n, U2, F, TCH);
				cudaDeviceSynchronize();

		// теперь считаем Fz и мацуно для Fz

		Fz << <dimGridZ, dimBlock >> > (AZv, UXv, UYv, UZv, P, F, DZv);
				cudaDeviceSynchronize();

		// мацуно Fz

		Mazuno << <dimGridZ, dimBlock >> > (UZv, U3, F, TCH);
				cudaDeviceSynchronize();

				Multiple << <dimGridZ, dimBlock >> > (U3, U3n);
				cudaDeviceSynchronize();
		Mazuno << <dimGridZ, dimBlock >> > (U3n, U3, F, TCH);
				cudaDeviceSynchronize();

		cufftExecZ2D(planinverse, UXv, UX);
		cufftExecZ2D(planinverse, UYv, UY);
		cufftExecZ2D(planinverse, UZv, UZ);
		cufftExecZ2D(planinverse, BXv, BX);
		cufftExecZ2D(planinverse, BYv, BY);
		cufftExecZ2D(planinverse, BZv, BZ);
				cudaDeviceSynchronize();
		//Поиск Fb(сначала посчитать D со скоростями)//поменять расчётную функцию

		Ddx << < dimGridZ, dimBlock >> > (BXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (BXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (BXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (UXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (UXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d5U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (UXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d6U);
				cudaDeviceSynchronize();

		//DX
		nelinDb << < dimGrid, dimBlock >> > (UX, UY, UZ, BX, BY, BZ, d1U, d2U, d3U, d4U, d5U, d6U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, DXv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (DXv);
				cudaDeviceSynchronize();
		////////////////////////////////////////////////////////////////
		Ddx << < dimGridZ, dimBlock >> > (BYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (BYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (BYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (UYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (UYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d5U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (UYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d6U);
				cudaDeviceSynchronize();
		//DY

		nelinDb << < dimGrid, dimBlock >> > (UX, UY, UZ, BX, BY, BZ, d1U, d2U, d3U, d4U, d5U, d6U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, DYv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (DYv);
				cudaDeviceSynchronize();

		////////////////////////////////////////////////////////////////
		Ddx << < dimGridZ, dimBlock >> > (BZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (BZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (BZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (UZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (UZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d5U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (UZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d6U);
				cudaDeviceSynchronize();
		////////////////////////////////////////////////////////////////////////////////////////////////////////

		// DZ
		nelinDb << < dimGrid, dimBlock >> > (UX, UY, UZ, BX, BY, BZ, d1U, d2U, d3U, d4U, d5U, d6U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, DZv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (DZv);
				cudaDeviceSynchronize();

		//Fbx
		Fbx << <dimGridZ, dimBlock >> > (DXv, BXv, BYv, BZv, F);
				cudaDeviceSynchronize();

		//мацуно Fbx
		Mazuno << <dimGridZ, dimBlock >> > (BXv, B1, F, TCH);
				cudaDeviceSynchronize();

				Multiple << <dimGridZ, dimBlock >> > (B1, B1n);

				cudaDeviceSynchronize();
		Mazuno << <dimGridZ, dimBlock >> > (B1n, B1, F, TCH);
				cudaDeviceSynchronize();

		//Fby
		Fby << <dimGridZ, dimBlock >> > (DYv, BXv, BYv, BZv, F);
				cudaDeviceSynchronize();

		//мацуно Fby
		Mazuno << <dimGridZ, dimBlock >> > (BYv, B2, F, TCH);
				cudaDeviceSynchronize();

				Multiple << <dimGridZ, dimBlock >> > (B2, B2n);

				cudaDeviceSynchronize();
		Mazuno << <dimGridZ, dimBlock >> > (B2n, B2, F, TCH);
				cudaDeviceSynchronize();
		//Fbz
		Fbz << <dimGridZ, dimBlock >> > (DZv, BXv, BYv, BZv, F);
				cudaDeviceSynchronize();

		//мацуно Fbz
		Mazuno << <dimGridZ, dimBlock >> > (BZv, B3, F, TCH);
				cudaDeviceSynchronize();

				Multiple << <dimGridZ, dimBlock >> > (B3, B3n);

				cudaDeviceSynchronize();
		Mazuno << <dimGridZ, dimBlock >> > (B3n, B3, F, TCH);
				cudaDeviceSynchronize();






		//повторяем расчёт, чтобы найти F*, зависящее от промежуточного значения U*
		///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		//Считаем нелинейную матрицу A*
		//компоненты UX,UY,UZ в физ(нужны для А*)
		CutN3 << <dimGridZ, dimBlock >> > (U1);
		CutN3 << <dimGridZ, dimBlock >> > (U2);
		CutN3 << <dimGridZ, dimBlock >> > (U3);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, U1, UX);
		cufftExecZ2D(planinverse, U2, UY);
		cufftExecZ2D(planinverse, U3, UZ);
				cudaDeviceSynchronize();

		// считаем производные UX*, UY*, UZ*, необходимые для AX и перекидываем в физ пр-во
		Ddx << < dimGridZ, dimBlock >> > (U2, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (U1, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (U1, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (U3, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();




		//собственно AX*
		nelinAx << < dimGrid, dimBlock >> > (UY, UZ, d1U, d2U, d3U, d4U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, AXv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (AXv);
				cudaDeviceSynchronize();

		//аналогично для AY
		Ddy << < dimGridZ, dimBlock >> > (U3, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (U2, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (U2, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (U1, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();

		nelinAy << < dimGrid, dimBlock >> > (UZ, UX, d1U, d2U, d3U, d4U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, AYv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (AYv);
				cudaDeviceSynchronize();

		//аналогично для AZ*
		Ddz << < dimGridZ, dimBlock >> > (U1, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (U3, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (U3, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (U2, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();

		nelinAz << < dimGrid, dimBlock >> > (UX, UY, d1U, d2U, d3U, d4U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, AZv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (AZv);
				cudaDeviceSynchronize();

		//Аналогичные расчёты нелинейного члена D*(чисто магнитного)
		CutN3 << <dimGridZ, dimBlock >> > (B1);
		CutN3 << <dimGridZ, dimBlock >> > (B2);
		CutN3 << <dimGridZ, dimBlock >> > (B3);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, B1, BX);
		cufftExecZ2D(planinverse, B2, BY);
		cufftExecZ2D(planinverse, B3, BZ);
				cudaDeviceSynchronize();

		Ddx << < dimGridZ, dimBlock >> > (B2, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (B1, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (B1, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (B3, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();

		//DX*
		nelinDx << < dimGrid, dimBlock >> > (BY, BZ, d1U, d2U, d3U, d4U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, DXv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (DXv);
				cudaDeviceSynchronize();

		////////////////////////////////////////////////////////////////////////////////////////////////////////
		Ddy << < dimGridZ, dimBlock >> > (B3, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (B2, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (B2, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (B1, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();

		//DY*

		nelinDy << < dimGrid, dimBlock >> > (BZ, BX, d1U, d2U, d3U, d4U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, DYv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (DYv);
				cudaDeviceSynchronize();

		////////////////////////////////////////////////////////////////////////////////////////////////////////

		Ddz << < dimGridZ, dimBlock >> > (B1, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (B3, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (B3, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (B2, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();

		// DZ
		nelinDz << < dimGrid, dimBlock >> > (BX, BY, d1U, d2U, d3U, d4U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, DZv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (DZv);
				cudaDeviceSynchronize();
		//дальше надо посчитать давление* ч/з лапласа

		lapP << <dimGridZ, dimBlock >> > (AXv, AYv, AZv, DXv, DYv, DZv, P);
				cudaDeviceSynchronize();

		//осталось только посчитать F*(правая часть уравнения)
		//Fx*
		Fx << <dimGridZ, dimBlock >> > (AXv, U1, U2, U3, P, F, DXv);
				cudaDeviceSynchronize();
		//мацуно Fx*
		Mazuno << <dimGridZ, dimBlock >> > (U1n, UXv, F, TCH);
				cudaDeviceSynchronize();

		//теперь считаем Fy*
		Fy << <dimGridZ, dimBlock >> > (AYv, U1, U2, U3, P, F, DYv);
				cudaDeviceSynchronize();
		// мацуно Fy*
		Mazuno << <dimGridZ, dimBlock >> > (U2n, UYv, F, TCH);
				cudaDeviceSynchronize();

		//теперь считаем Fz*
		Fz << <dimGridZ, dimBlock >> > (AZv, U1, U2, U3, P, F, DZv);
				cudaDeviceSynchronize();
		// мацуно Fz*
		Mazuno << <dimGridZ, dimBlock >> > (U3n, UZv, F, TCH);
				cudaDeviceSynchronize();


		//DX*(со скоростью)
		Ddx << < dimGridZ, dimBlock >> > (B1, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (B1, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (B1, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (U1, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (U1, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d5U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (U1, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d6U);
				cudaDeviceSynchronize();

		nelinDb << < dimGrid, dimBlock >> > (UX, UY, UZ, BX, BY, BZ, d1U, d2U, d3U, d4U, d5U, d6U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, DXv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (DXv);
				cudaDeviceSynchronize();
		///////////////////////////////////////
		//DY*(cо скоростью)
		Ddx << < dimGridZ, dimBlock >> > (B2, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (B2, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (B2, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (U2, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (U2, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d5U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (U2, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d6U);
				cudaDeviceSynchronize();

		nelinDb << < dimGrid, dimBlock >> > (UX, UY, UZ, BX, BY, BZ, d1U, d2U, d3U, d4U, d5U, d6U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, DYv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (DYv);
				cudaDeviceSynchronize();
		////////////////////////////////////////////
		//DZ*(cо скоростью)
		Ddx << < dimGridZ, dimBlock >> > (B3, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (B3, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (B3, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (U3, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (U3, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d5U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (U3, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d6U);
				cudaDeviceSynchronize();

		nelinDb << < dimGrid, dimBlock >> > (UX, UY, UZ, BX, BY, BZ, d1U, d2U, d3U, d4U, d5U, d6U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, DZv);
				cudaDeviceSynchronize();
		CutN3 << <dimGridZ, dimBlock >> > (DZv);
				cudaDeviceSynchronize();

	//нахрена это тут??????????????????!!!!!!!!!!!!!!!!!!!!!!!(это было для отключения магнитных полей)
	//	Null << <dimGridZ, dimBlock >> > (B1n);
	//	Null << <dimGridZ, dimBlock >> > (B2n);
	//	Null << <dimGridZ, dimBlock >> > (B3n);
	//			cudaDeviceSynchronize();
	//видимо, чтобы жизнь мёдом не казалась!!
		//Fbx*
		Fbx << <dimGridZ, dimBlock >> > (DXv, B1, B2, B3, F);
				cudaDeviceSynchronize();

		//мацуно Fbx*
		Mazuno << <dimGridZ, dimBlock >> > (B1n, BXv, F, TCH);
				cudaDeviceSynchronize();

		//Fby*
		Fby << <dimGridZ, dimBlock >> > (DYv, B1, B2, B3, F);
				cudaDeviceSynchronize();

		//мацуно Fby*
		Mazuno << <dimGridZ, dimBlock >> > (B2n, BYv, F, TCH);
				cudaDeviceSynchronize();

		//Fbz*
		Fbz << <dimGridZ, dimBlock >> > (DZv, B1, B2, B3, F);
				cudaDeviceSynchronize();

		//мацуно Fbz*
		Mazuno << <dimGridZ, dimBlock >> > (B3n, BZv, F, TCH);
				cudaDeviceSynchronize();





//Uxv, Uyv,Uzv вместо копирования копируем с добавлением форсинга после вывода данных
		T += tch[0];
		CutN3 << <dimGridZ, dimBlock >> > (UXv);
		CutN3 << <dimGridZ, dimBlock >> > (UYv);
		CutN3 << <dimGridZ, dimBlock >> > (UZv);
		CutN3 << <dimGridZ, dimBlock >> > (BXv);
		CutN3 << <dimGridZ, dimBlock >> > (BYv);
		CutN3 << <dimGridZ, dimBlock >> > (BZv);

				cudaDeviceSynchronize();

			if ((vivodphis)&&(T>Nk2))
			{
				start_time = clock()/CLOCKS_PER_SEC;
				cufftExecZ2D(planinverse, UXv, UX);
				cufftExecZ2D(planinverse, UYv, UY);
				cufftExecZ2D(planinverse, UZv, UZ);
				cudaDeviceSynchronize();
				DNNN << <dimGrid, dimBlock >> > (UX);
				DNNN << <dimGrid, dimBlock >> > (UY);
				DNNN << <dimGrid, dimBlock >> > (UZ);
				cudaDeviceSynchronize();

				strcpy(file, OutputDirectory);
				strncat(file, "UXout",80);
				sprintf(dir,"%d",NNk2);
				strncat(file, dir, 10);
				strncat(file, ".dat", 80);
				out1 = fopen(file, "w+");


				strcpy(file, OutputDirectory);
				strncat(file, "UYout",80);
				sprintf(dir,"%d",NNk2);
				strncat(file, dir, 10);
				strncat(file, ".dat", 80);
				out2 = fopen(file, "w+");


				strcpy(file, OutputDirectory);
				strncat(file, "UZout",80);
				sprintf(dir,"%d",NNk2);
				strncat(file, dir, 10);
				strncat(file, ".dat", 80);
				out3 = fopen(file, "w+");

				 if (N<=256) {
					cudaMemcpy(h_UX, UX, csize, cudaMemcpyDeviceToHost);
					cudaMemcpy(h_UY, UY, csize, cudaMemcpyDeviceToHost);
					cudaMemcpy(h_UZ, UZ, csize, cudaMemcpyDeviceToHost);
					}

				for (int i = 0; i < N; i++)
					for (int j = 0; j < N; j++) {
						for (int k = 0; k < N; k++)
						{
							fprintf(out1, "%.16e ", h_UX[(i*N + j)*N + k]);
							fprintf(out2, "%.16e ", h_UY[(i*N + j)*N + k]);
							fprintf(out3, "%.16e ", h_UZ[(i*N + j)*N + k]);
						}
						fprintf(out1, "\n");
						fprintf(out2, "\n");
						fprintf(out3, "\n");
					}

				fclose(out1);
				fclose(out2);
				fclose(out3);


				cufftExecZ2D(planinverse, BXv, UX);
				cufftExecZ2D(planinverse, BYv, UY);
				cufftExecZ2D(planinverse, BZv, UZ);
				cudaDeviceSynchronize();
				DNNN << <dimGrid, dimBlock >> > (UX);
				DNNN << <dimGrid, dimBlock >> > (UY);
				DNNN << <dimGrid, dimBlock >> > (UZ);
				cudaDeviceSynchronize();


				strcpy(file, OutputDirectory);
				strncat(file, "BXout",80);
				sprintf(dir,"%d",NNk2);
				strncat(file, dir, 10);
				strncat(file, ".dat", 80);
				out1 = fopen(file, "w+");


				strcpy(file, OutputDirectory);
				strncat(file, "BYout",80);
				sprintf(dir,"%d",NNk2);
				strncat(file, dir, 10);
				strncat(file, ".dat", 80);
				out2 = fopen(file, "w+");


				strcpy(file, OutputDirectory);
				strncat(file, "BZout",80);
				sprintf(dir,"%d",NNk2);
				strncat(file, dir, 10);
				strncat(file, ".dat", 80);
				out3 = fopen(file, "w+");

				 if (N<=256) {
					cudaMemcpy(h_BX, BX, csize, cudaMemcpyDeviceToHost);
					cudaMemcpy(h_BY, BY, csize, cudaMemcpyDeviceToHost);
					cudaMemcpy(h_BZ, BZ, csize, cudaMemcpyDeviceToHost);
					}

				for (int i = 0; i < N; i++)
					for (int j = 0; j < N; j++)
					{
						for (int k = 0; k < N; k++)
						{
							fprintf(out1, "%.16e ", h_UX[(i*N + j)*N + k]);
							fprintf(out2, "%.16e ", h_UY[(i*N + j)*N + k]);
							fprintf(out3, "%.16e ", h_UZ[(i*N + j)*N + k]);
						}
						fprintf(out1, "\n");
						fprintf(out2, "\n");
						fprintf(out3, "\n");
					}
				fclose(out1);
				fclose(out2);
				fclose(out3);

				Nk2 += dNk2;
				NNk2 += 10;

				end_time = clock()/CLOCKS_PER_SEC;//конечное время ввода данных

				cout << "Raw data transfer to disk made in " << double(end_time - start_time) << " sek" << endl;
			}

			//specters
			if ((vivodspec) && (T>Nk))
			{
				start_time = clock()/CLOCKS_PER_SEC;
				Sumforen << <dimGridZ, dimBlock >> > (UXv, Ukx);
				Sumforen << <dimGridZ, dimBlock >> > (UYv, Uky);
				Sumforen << <dimGridZ, dimBlock >> > (UZv, Ukz);
				cudaDeviceSynchronize();


				cudaMemcpy(ukx, h_Ukx, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);
				cudaMemcpy(uky, h_Uky, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);
				cudaMemcpy(ukz, h_Ukz, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);

				cudaDeviceSynchronize();
				Sumforen << <dimGridZ, dimBlock >> > (BXv, Ukx);
				Sumforen << <dimGridZ, dimBlock >> > (BYv, Uky);
				Sumforen << <dimGridZ, dimBlock >> > (BZv, Ukz);
				cudaDeviceSynchronize();

				cudaMemcpy(bkx, h_Ukx, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);
				cudaMemcpy(bky, h_Uky, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);
				cudaMemcpy(bkz, h_Ukz, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);

				cudaDeviceSynchronize();
				Null1(ex, Nl);
				Null1(ey, Nl);
				Null1(ez, Nl);
				Null1(ebx, Nl);
				Null1(eby, Nl);
				Null1(ebz, Nl);


				for (int i = 0; i < Nl; i++)
					for (int ix = 0; ix < N / 3; ix += 1)
						for (int iy = 0; iy < N / 3; iy += 1)
							for (int iz = 1; iz < N / 3; iz += 1)
								if (WN[i] == ix*ix + iy*iy + iz*iz)
								{
									id1 = (ix*N + iy)*(N / 2 + 1) + iz;
									id2 = ((-ix + N)*N + iy)*(N / 2 + 1) + iz;
									id3 = (ix*N + (-iy + N))*(N / 2 + 1) + iz;
									id4 = ((-ix + N)*N + (-iy + N))*(N / 2 + 1) + iz;
									ex[i] += ukx[id1] + ukx[id2] + ukx[id3] + ukx[id4];
									ey[i] += uky[id1] + uky[id2] + uky[id3] + uky[id4];
									ez[i] += ukz[id1] + ukz[id2] + ukz[id3] + ukz[id4];
									ebx[i] += bkx[id1] + bkx[id2] + bkx[id3] + bkx[id4];
									eby[i] += bky[id1] + bky[id2] + bky[id3] + bky[id4];
									ebz[i] += bkz[id1] + bkz[id2] + bkz[id3] + bkz[id4];
								}
//убираем повторяющиеся нули
				ex[0] /= 4.0;
				ey[0] /= 4.0;
				ez[0] /= 4.0;
				ebx[0] /= 4.0;
				eby[0] /= 4.0;
				ebz[0] /= 4.0;

				strcpy(file, OutputDirectory);
				strncat(file, "UXsp",80);
				sprintf(dir,"%d",NNk);
				strncat(file, dir, 10);
				strncat(file, ".dat", 80);
				out1 = fopen(file, "w+");


				strcpy(file, OutputDirectory);
				strncat(file, "UYsp",80);
				sprintf(dir,"%d",NNk);
				strncat(file, dir, 10);
				strncat(file, ".dat", 80);
				out2 = fopen(file, "w+");


				strcpy(file, OutputDirectory);
				strncat(file, "UZsp",80);
				sprintf(dir,"%d",NNk);
				strncat(file, dir, 10);
				strncat(file, ".dat", 80);
				out3 = fopen(file, "w+");

				for (int i = 0; i < Nl; i++)
				{
					fprintf(out1, "%.8e\n", ex[i]);
				}
				for (int i = 0; i < Nl; i++)
				{
					fprintf(out2, "%.8e\n", ey[i]);
				}
				for (int i = 0; i < Nl; i++)
				{
					fprintf(out3, "%.8e\n", ez[i]);
				}
				fclose(out1);
				fclose(out2);
				fclose(out3);


				strcpy(file, OutputDirectory);
				strncat(file, "BXsp",80);
				sprintf(dir,"%d",NNk);
				strncat(file, dir, 10);
				strncat(file, ".dat", 80);
				out1 = fopen(file, "w+");


				strcpy(file, OutputDirectory);
				strncat(file, "BYsp",80);
				sprintf(dir,"%d",NNk);
				strncat(file, dir, 10);
				strncat(file, ".dat", 80);
				out2 = fopen(file, "w+");


				strcpy(file, OutputDirectory);
				strncat(file, "BZsp",80);
				sprintf(dir,"%d",NNk);
				strncat(file, dir, 10);
				strncat(file, ".dat", 80);
				out3 = fopen(file, "w+");

				for (int i = 0; i < Nl; i++)
				{
					fprintf(out1, "%.8e\n", ebx[i]);
				}
				for (int i = 0; i < Nl; i++)
				{
					fprintf(out2, "%.8e\n", eby[i]);
				}
				for (int i = 0; i < Nl; i++)
				{
					fprintf(out3, "%.8e\n", ebz[i]);
				}
				fclose(out1);
				fclose(out2);
				fclose(out3);
				Nk += dNk;
				NNk += 10;

				end_time = clock()/CLOCKS_PER_SEC;//конечное время ввода данных
				cout << "Calculation of spectres and data transfer to disk made in " << double(end_time - start_time) << " sek" << endl;
			}


		if ((vivodhelicity) && (T>Nk))
	{
		start_time = clock()/CLOCKS_PER_SEC;
				///////////////////////////////////////////////
				// расчет перекрестной спиральности
				///////////////////////////////////////////////
		CutN3 << <dimGridZ, dimBlock >> > (UXv);
		CutN3 << <dimGridZ, dimBlock >> > (UYv);
		CutN3 << <dimGridZ, dimBlock >> > (UZv);
				cudaDeviceSynchronize();

		cufftExecZ2D(planinverse, UXv, UX);
		cufftExecZ2D(planinverse, UYv, UY);
		cufftExecZ2D(planinverse, UZv, UZ);
				cudaDeviceSynchronize();

		CutN3 << <dimGridZ, dimBlock >> > (BXv);
		CutN3 << <dimGridZ, dimBlock >> > (BYv);
		CutN3 << <dimGridZ, dimBlock >> > (BZv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, BXv, BX);
		cufftExecZ2D(planinverse, BYv, BY);
		cufftExecZ2D(planinverse, BZv, BZ);
				cudaDeviceSynchronize();


	CrossHelicity <<< dimGrid, dimBlock >>> (UX, UY, UZ, BX, BY, BZ, AiD);
				cudaDeviceSynchronize();
	cufftExecD2Z(plan, AiD, DXv);
				cudaDeviceSynchronize();

				///////////////////////////////////////////////
				//расчет спиральност
				///////////////////////////////////////////////

				///////////////////////////////////////////////
				//иксовая компонента(производные, умножаемые на соответствующую компоненту Ux)
				///////////////////////////////////////////////
		Ddy << < dimGridZ, dimBlock >> > (UZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d1U);
				cudaDeviceSynchronize();
		Ddz << < dimGridZ, dimBlock >> > (UYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d2U);
				cudaDeviceSynchronize();

				///////////////////////////////////////////////
				//игрековая компонента(производные, умножаемые на соответствующую компоненту Ux)
				///////////////////////////////////////////////
		Ddz << < dimGridZ, dimBlock >> > (UXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d3U);
				cudaDeviceSynchronize();
		Ddx << < dimGridZ, dimBlock >> > (UZv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d4U);
				cudaDeviceSynchronize();

				///////////////////////////////////////////////
				//зетовая компонента(производные, умножаемые на соответствующую компоненту Ux)
				///////////////////////////////////////////////
		Ddx << < dimGridZ, dimBlock >> > (UYv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d5U);
				cudaDeviceSynchronize();
		Ddy << < dimGridZ, dimBlock >> > (UXv, dUv);
				cudaDeviceSynchronize();
		cufftExecZ2D(planinverse, dUv, d6U);
				cudaDeviceSynchronize();

		Helicity	<<< dimGrid, dimBlock >>> (UX, UY, UZ, d1U, d2U, d3U, d4U, d5U, d6U, AiD);
				cudaDeviceSynchronize();
		cufftExecD2Z(plan, AiD, AXv);
				cudaDeviceSynchronize();


				///////////////////////////////////////////////
				//вывод и расчет спектров
				///////////////////////////////////////////////


		Sumforen << <dimGridZ, dimBlock >> > (AXv, Ukx);
		Sumforen << <dimGridZ, dimBlock >> > (DXv, Uky);
		cudaDeviceSynchronize();

		cudaMemcpy(ukx, h_Ukx, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);
		cudaMemcpy(uky, h_Uky, N*N*(N / 2 + 1) * sizeof(cufftDoubleReal), cudaMemcpyHostToHost);

		cudaDeviceSynchronize();

		Null1(ex, Nl);
		Null1(ey, Nl);


		for (int i = 0; i < Nl; i++)
			for (int ix = 0; ix < N / 3; ix += 1)
				for (int iy = 0; iy < N / 3; iy += 1)
					for (int iz = 0; iz < N / 3; iz += 1)
						if (WN[i] == ix*ix + iy*iy + iz*iz)
						{
							id1 = (ix*N + iy)*(N / 2 + 1) + iz;
							id2 = ((-ix + N)*N + iy)*(N / 2 + 1) + iz;
							id3 = (ix*N + (-iy + N))*(N / 2 + 1) + iz;
							id4 = ((-ix + N)*N + (-iy + N))*(N / 2 + 1) + iz;
							ex[i] += ukx[id1] + ukx[id2] + ukx[id3] + ukx[id4];
							ey[i] += uky[id1] + uky[id2] + uky[id3] + uky[id4];
						}

		ex[0] /= 4.0;
		ey[0] /= 4.0;

		strcpy(file, OutputDirectory);
		strncat(file, "HelicitySP",80);
		sprintf(dir,"%d",NNk);
		strncat(file, dir, 10);
		strncat(file, ".dat", 80);
		out1 = fopen(file, "w+");

		strcpy(file, OutputDirectory);
		strncat(file, "CrossHelicitySP",80);
		sprintf(dir,"%d",NNk);
		strncat(file, dir, 10);
		strncat(file, ".dat", 80);
		out2 = fopen(file, "w+");


		for (int i = 0; i < Nl; i++)
		{
			fprintf(out1, "%.8e\n", ex[i]);
		}
		for (int i = 0; i < Nl; i++)
		{
			fprintf(out2, "%.8e\n", ey[i]);
		}

		fclose(out1);
		fclose(out2);


		end_time = clock()/CLOCKS_PER_SEC;//конечное время ввода данных
		cout << "Calculation of helicity spectra and data transfer to disk made in " << double(end_time - start_time) << " sek" << endl;
	}


		NNN++;

		if (useforcing)
		{
			cufftExecZ2D(planinverse, UXv, UX);
			cufftExecZ2D(planinverse, UYv, UY);
			cufftExecZ2D(planinverse, UZv, UZ);
				cudaDeviceSynchronize();
			DNNN << <dimGrid, dimBlock >> > (UX);
			DNNN << <dimGrid, dimBlock >> > (UY);
			DNNN << <dimGrid, dimBlock >> > (UZ);
				cudaDeviceSynchronize();

			//start_time = clock()/CLOCKS_PER_SEC;
			phix = 2 * M_PI*double(rand()) / double(RAND_MAX);
			phiy = 2 * M_PI*double(rand()) / double(RAND_MAX);
			phiz = 2 * M_PI*double(rand()) / double(RAND_MAX);
			num1 = rand() % Nf;//число определяющее группу векторов(4 симметричных вектора), на которых форсируем
			num2 = rand() % 4;//определяем, kx,ky или -kx,ky или kx,-ky  или -kx,-ky
			if (num2 == 0) {
				kx = Kf[3 * num1];
				ky = Kf[3 * num1 + 1];
				kz = Kf[3 * num1 + 2];
			}
			else if (num2 == 1) {
				kx = -Kf[3 * num1];
				ky = Kf[3 * num1 + 1];
				kz = Kf[3 * num1 + 2];
			}
			else if (num2 == 2) {
				kx = Kf[3 * num1];
				ky = -Kf[3 * num1 + 1];
				kz = Kf[3 * num1 + 2];
			}
			else if (num2 == 3) {
				kx = -Kf[3 * num1];
				ky = -Kf[3 * num1 + 1];
				kz = Kf[3 * num1 + 2];
			}

		Estep = EForse / dtmax*tch[0] / sqrt(double(kx)*double(kx) + double(ky)*double(ky) + double(kz)*double(kz));
		FindABC(kx, ky, kz, phix, phiy, phiz, Estep, abc);//(kx,ky,kz,phi1,phi2,phi3,Eforcing,ABC)
		//cout << abc[0] << " " << abc[1] << " " << abc[2] << " " << NNN << endl;
		//cout << numofforsing[0] << " " << numofforsing[1] << " " << numofforsing[2] << endl;																															cout << forseandphi[0] << " " << forseandphi[1] << " " << forseandphi[2] << " " << forseandphi[3] << endl;*/
		ABC1 = abc[0];
		ABC2 = abc[1];
		ABC3 = abc[2];

		Forcing << <dimGrid, dimBlock >> > (AiD, UX, UY, UZ, kx, ky, kz, phix, phiy, phiz, ABC1, ABC2, ABC3);




		//конечное время ввода данных

		cufftExecD2Z(plan, UX, UXv);
		cufftExecD2Z(plan, UY, UYv);
		cufftExecD2Z(plan, UZ, UZv);

	//	cout << "Forcing made in " << (end_time - start_time) << " msek" << endl;
		}
				cudaDeviceSynchronize();

		end_time2 = clock()/CLOCKS_PER_SEC;

		printf("step N %d time is %lf s\n", NNN, (double(end_time2 - start_time2)));
		//cout << "step N" << NNN << " time is " << double(end_time2 - start_time2) << " s\n";
	}


	//cout << "Конец вычислений." << endl;
	//end_time = clock()/CLOCKS_PER_SEC;//конечное время расчёта
	/*out.open("D:\\Files\\6\\data.dat");
	out << "Расчёт занял " << (end_time - start_time)  << " секунд" <<" на "<< NNN << " шагов"<<endl;
	out.close();*/



	cufftDestroy(plan);
	cufftDestroy(planinverse);
	cudaFree(TCH);
	cudaFreeHost(h_F);
	cudaFreeHost(h_P);
	cudaFreeHost(h_UX);
	cudaFreeHost(h_UY);
	cudaFreeHost(h_UZ);
	cudaFreeHost(h_UXv);
	cudaFreeHost(h_UYv);
	cudaFreeHost(h_UZv);
	cudaFreeHost(h_AXv);
	cudaFreeHost(h_AYv);
	cudaFreeHost(h_AZv);
	cudaFree(dUv);
	cudaFree(d1U);
	cudaFreeHost(h_d2U);
	cudaFreeHost(h_d3U);
	cudaFreeHost(h_d4U);
	cudaFreeHost(h_d5U);
	cudaFreeHost(h_d6U);

	cudaFreeHost(h_Ukx);
	cudaFreeHost(h_UYv);
	cudaFreeHost(h_UZv);

	cudaFree(AiD);

	cudaFreeHost(h_BX);
	cudaFreeHost(h_BY);
	cudaFreeHost(h_BZ);
	cudaFreeHost(h_BXv);
	cudaFreeHost(h_BYv);
	cudaFreeHost(h_BZv);
	cudaFreeHost(h_DXv);
	cudaFreeHost(h_DYv);
	cudaFreeHost(h_DZv);

	cudaFreeHost(h_U1);
	cudaFreeHost(h_U2);
	cudaFreeHost(h_U3);
	cudaFreeHost(h_U1n);
	cudaFreeHost(h_U2n);
	cudaFreeHost(h_U3n);
	cudaFreeHost(h_B1);
	cudaFreeHost(h_B2);
	cudaFreeHost(h_B3);
	cudaFreeHost(h_B1n);
	cudaFreeHost(h_B2n);
	cudaFreeHost(h_B3n);
	free(tch);
	free(ukx);
	free(uky);
	free(ukz);
	free(bkx);
	free(bky);
	free(bkz);
	free(ex);
	free(ey);
	free(ez);
	free(ebx);
	free(eby);
	free(ebz);
	free(WN);
	free(Kf);
	free(abc);

	return 0;
}

