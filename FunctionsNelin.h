__global__
void NullREAL(cufftDoubleReal *a)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*N + idz;
	a[id] = 0;
}

__global__
void Null(cufftDoubleComplex *a)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N/2+1) + idz;
	a[id].x = 0;
	a[id].y = 0;
}

/*
__global__
void CutN3(cufftDoubleComplex *a)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;

	if (idx>N / 2) idx = idx - N;
	if (idy>N / 2) idy = idy - N;
	if (idz>N / 2) idz = idz - N;

	if ((idx >= (N / 3)) || (idy >= (N / 3)) || (idz >= (N / 3)) || (idx <= (-N / 3)) || (idy <= (-N / 3)) || (idz <= (-N / 3)))
	{
		a[id].x = 0;
		a[id].y = 0;
	}
	if ((blockIdx.z==gridDim.z-1)&&(threadIdx.z==blockDim.z-1))
	{
		id = id + 1;
	a[id].x = 0.0;
	a[id].y = 0.0;
	}
}
*/

__global__
void CutN3(cufftDoubleComplex *a)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;
	int localedge=(N/3)*(N/3);
	
	if (idx>N / 2) idx = idx - N;
	if (idy>N / 2) idy = idy - N;
	if (idz>N / 2) idz = idz - N;

	if (idx*idx+idy*idy+idz*idz >= localedge)
	{
		a[id].x = 0;
		a[id].y = 0;
	}
	if ((blockIdx.z==gridDim.z-1)&&(threadIdx.z==blockDim.z-1))
	{
		id = id + 1;
	a[id].x = 0.0;
	a[id].y = 0.0;
	}
}

__global__
void DNNN(cufftDoubleReal *a)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*N + idz;

	a[id] = a[id] / lambda;
}

__global__
void Ddx(cufftDoubleComplex *a, cufftDoubleComplex *b)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;


	if (idx>N / 2) idx = idx - N;

	b[id].x = -(double)idx*a[id].y;
	b[id].y = (double)idx*a[id].x;

	if (idx == N/2) {
		b[id].x = 0.;
		b[id].y = 0.;
	}

}

__global__
void Ddy(cufftDoubleComplex *a, cufftDoubleComplex *b)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;


	if (idy>N / 2) idy = idy - N;

	b[id].x = -(double)idy*a[id].y;
	b[id].y = (double)idy*a[id].x;

	if (idy == N/2) {
		b[id].x = 0.;
		b[id].y = 0.;
	}


}

__global__
void Ddz(cufftDoubleComplex *a, cufftDoubleComplex *b)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;

	if (idz>N / 2) idz = idz - N;

	b[id].x = -(double)idz*a[id].y;
	b[id].y = (double)idz*a[id].x;

	if (idz == N/2) {
		b[id].x = 0.;
		b[id].y = 0.;
	}
}

__global__
void nelinAx(cufftDoubleReal *a1, cufftDoubleReal *a2, cufftDoubleReal *b1, cufftDoubleReal *b2, cufftDoubleReal *b3, cufftDoubleReal *b4, cufftDoubleReal *c) //a - функции, b - производные, c -выходное значение
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*N + idz;

	c[id] = a1[id]*(b1[id] - b2[id] + 2.0*Omz*lambda)/ (lambda*lambda) - a2[id]*(b3[id] - b4[id] + 2.0*Omy*lambda) / (lambda*lambda);
	
}


__global__
void nelinAy(cufftDoubleReal *a1, cufftDoubleReal *a2, cufftDoubleReal *b1, cufftDoubleReal *b2, cufftDoubleReal *b3, cufftDoubleReal *b4, cufftDoubleReal *c) //a - функции, b - производные, c -выходное значение
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*N + idz;

	c[id] = a1[id]*(b1[id] - b2[id] + 2.0*Omx*lambda) / (lambda*lambda) - a2[id]*(b3[id] - b4[id]+2.0*Omz*lambda) / (lambda*lambda);
}


__global__
void nelinAz(cufftDoubleReal *a1, cufftDoubleReal *a2, cufftDoubleReal *b1, cufftDoubleReal *b2, cufftDoubleReal *b3, cufftDoubleReal *b4, cufftDoubleReal *c) //a - функции, b - производные, c -выходное значение
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*N + idz;

	c[id] = a1[id]*(b1[id] - b2[id] + 2.0*Omy*lambda) / (lambda*lambda) - a2[id]*(b3[id] - b4[id]+2.0*Omx*lambda) / (lambda*lambda);
}


__global__
void nelinDx(cufftDoubleReal *a1, cufftDoubleReal *a2, cufftDoubleReal *b1, cufftDoubleReal *b2, cufftDoubleReal *b3, cufftDoubleReal *b4, cufftDoubleReal *c) //a - функции, b - производные, c -выходное значение
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*N + idz;

	c[id] = -(a1[id]+B0y*lambda)*(b1[id] - b2[id]) / (lambda*lambda) + (a2[id]+B0z*lambda)*(b3[id] - b4[id]) / (lambda*lambda);
}

__global__
void nelinDy(cufftDoubleReal *a1, cufftDoubleReal *a2, cufftDoubleReal *b1, cufftDoubleReal *b2, cufftDoubleReal *b3, cufftDoubleReal *b4, cufftDoubleReal *c) //a - функции, b - производные, c -выходное значение
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*N + idz;

	c[id] = -(a1[id]+B0z*lambda)*(b1[id] - b2[id]) / (lambda*lambda) + (a2[id]+B0x*lambda)*(b3[id] - b4[id]) / (lambda*lambda);
}

__global__
void nelinDz(cufftDoubleReal *a1, cufftDoubleReal *a2, cufftDoubleReal *b1, cufftDoubleReal *b2, cufftDoubleReal *b3, cufftDoubleReal *b4, cufftDoubleReal *c) //a - функции, b - производные, c -выходное значение
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*N + idz;

	c[id] = -(a1[id]+B0x*lambda)*(b1[id] - b2[id]) / (lambda*lambda) + (a2[id]+B0y*lambda)*(b3[id] - b4[id]) / (lambda*lambda);
}

__global__
void lapP(cufftDoubleComplex *b1, cufftDoubleComplex *b2, cufftDoubleComplex *b3, cufftDoubleComplex *d1, cufftDoubleComplex *d2, cufftDoubleComplex *d3, cufftDoubleComplex *a)//a - давление, b - нелинейные члены AX,AY,AZ, d - нелинейные члены DX,DY,DZ
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;

	if (idx>N / 2) idx = idx - N;
	if (idy>N / 2) idy = idy - N;
	if (idz>N / 2) idz = idz - N;

	a[id].x = ((double)idx*(b1[id].y + d1[id].y) + (double)idy*(b2[id].y + d2[id].y) + (double)idz*(b3[id].y + d3[id].y)) / ((double)(idx*idx + idy*idy + idz*idz));
	a[id].y = -((double)idx*(b1[id].x + d1[id].x) + (double)idy*(b2[id].x + d2[id].x) + (double)idz*(b3[id].x + d3[id].x)) / ((double)(idx*idx + idy*idy + idz*idz));

	if ((idx == 0) && (idy == 0) && (idz == 0))
	{
		a[id].x = 0.;
		a[id].y = 0.;
	}

}

__global__
void Fx(cufftDoubleComplex *a, cufftDoubleComplex *ux, cufftDoubleComplex *uy, cufftDoubleComplex *uz, cufftDoubleComplex *p, cufftDoubleComplex *f, cufftDoubleComplex *d)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;
	
	if (idx>N / 2) idx = idx - N;
	if (idy>N / 2) idy = idy - N;
	if (idz>N / 2) idz = idz - N;
	

	f[id].x = a[id].x + d[id].x + (double)idx*p[id].y - nu*((double)idx*idx*ux[id].x + (double)idx*idy*uy[id].x + (double)idx*idz*uz[id].x);
	f[id].y = a[id].y + d[id].y - (double)idx*p[id].x - nu*((double)idx*idx*ux[id].y + (double)idx*idy*uy[id].y + (double)idx*idz*uz[id].y);

}

__global__
void Fy(cufftDoubleComplex *a, cufftDoubleComplex *ux, cufftDoubleComplex *uy, cufftDoubleComplex *uz, cufftDoubleComplex *p, cufftDoubleComplex *f, cufftDoubleComplex *d)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;
	
	
	if (idx>N / 2) idx = idx - N;
	if (idy>N / 2) idy = idy - N;
	if (idz>N / 2) idz = idz - N;
	
	
	f[id].x = a[id].x + d[id].x + (double)idy*p[id].y - nu*((double)idy*idx*ux[id].x + (double)idy*idy*uy[id].x + (double)idy*idz*uz[id].x);
	f[id].y = a[id].y + d[id].y - (double)idy*p[id].x - nu*((double)idy*idx*ux[id].y + (double)idy*idy*uy[id].y + (double)idy*idz*uz[id].y);

}

__global__
void Fz(cufftDoubleComplex *a, cufftDoubleComplex *ux, cufftDoubleComplex *uy, cufftDoubleComplex *uz, cufftDoubleComplex *p, cufftDoubleComplex *f, cufftDoubleComplex *d)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;
	

	if (idx>N / 2) idx = idx - N;
	if (idy>N / 2) idy = idy - N;
	if (idz>N / 2) idz = idz - N;
	
	
	f[id].x = a[id].x + d[id].x + (double)idz*p[id].y - nu*((double)idz*idx*ux[id].x + (double)idz*idy*uy[id].x + (double)idz*idz*uz[id].x);
	f[id].y = a[id].y + d[id].y - (double)idz*p[id].x - nu*((double)idz*idx*ux[id].y + (double)idz*idy*uy[id].y + (double)idz*idz*uz[id].y);

}

__global__
void nelinDb(cufftDoubleReal *ux, cufftDoubleReal *uy, cufftDoubleReal *uz, cufftDoubleReal *bx, cufftDoubleReal *by, cufftDoubleReal *bz, cufftDoubleReal *d1b, cufftDoubleReal *d2b, cufftDoubleReal *d3b, cufftDoubleReal *d1u, cufftDoubleReal *d2u, cufftDoubleReal *d3u, cufftDoubleReal *c) //a - функции, b - производные, c -выходное значение
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*N + idz;

	c[id] = ((bx[id] + B0x*lambda)*d1u[id] + (by[id] + B0y*lambda)* d2u[id] + (bz[id] + B0z*lambda)* d3u[id] - ux[id] * d1b[id] - uy[id] * d2b[id] - uz[id] * d3b[id]) / (lambda*lambda);
}


__global__
void Fbx(cufftDoubleComplex *d, cufftDoubleComplex *bx, cufftDoubleComplex *by, cufftDoubleComplex *bz, cufftDoubleComplex *f)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;
	
	
	if (idx>N / 2) idx = idx - N;
	if (idy>N / 2) idy = idy - N;
	if (idz>N / 2) idz = idz - N;
	
	
	f[id].x = d[id].x - eta*((double)idx*idx*bx[id].x + (double)idx*idy*by[id].x + (double)idx*idz*bz[id].x);
	f[id].y = d[id].y - eta*((double)idx*idx*bx[id].y + (double)idx*idy*by[id].y + (double)idx*idz*bz[id].y);

}

__global__
void Fby(cufftDoubleComplex *d, cufftDoubleComplex *bx, cufftDoubleComplex *by, cufftDoubleComplex *bz, cufftDoubleComplex *f)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;
	

	if (idx>N / 2) idx = idx - N;
	if (idy>N / 2) idy = idy - N;
	if (idz>N / 2) idz = idz - N;
	
	
	
	f[id].x = d[id].x - eta*((double)idy*idx*bx[id].x + (double)idy*idy*by[id].x + (double)idy*idz*bz[id].x);
	f[id].y = d[id].y - eta*((double)idy*idx*bx[id].y + (double)idy*idy*by[id].y + (double)idy*idz*bz[id].y);

}

__global__
void Fbz(cufftDoubleComplex *d, cufftDoubleComplex *bx, cufftDoubleComplex *by, cufftDoubleComplex *bz, cufftDoubleComplex *f)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;
	

	if (idx>N / 2) idx = idx - N;
	if (idy>N / 2) idy = idy - N;
	if (idz>N / 2) idz = idz - N;
	
	
	
	f[id].x = d[id].x - eta*((double)idz*idx*bx[id].x + (double)idz*idy*by[id].x + (double)idz*idz*bz[id].x);
	f[id].y = d[id].y - eta*((double)idz*idx*bx[id].y + (double)idz*idy*by[id].y + (double)idz*idz*bz[id].y);


}


__global__
void AbsZn(cufftDoubleReal *a, cufftDoubleReal *c, cufftDoubleReal *d)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*N + idz;

	a[id] = fabs(a[id]);
	c[id] = fabs(c[id]);
	d[id] = fabs(d[id]);
}

__global__
void AbsZnB(cufftDoubleReal *a, cufftDoubleReal *c, cufftDoubleReal *d)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*N + idz;

	a[id] = fabs(B0x+a[id]);
	c[id] = fabs(B0y+c[id]);
	d[id] = fabs(B0z+d[id]);
}


__global__
void DT(cufftDoubleReal *a, cufftDoubleReal *c, cufftDoubleReal *d, cufftDoubleReal *TCH)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*N + idz;
	int s = N*N*N / 2;
	while (s > 0)
	{
		if (id < s)
		{
			if (a[id]<a[id + s]) a[id] = a[id + s];
			if (c[id]<c[id + s]) c[id] = c[id + s];
			if (d[id]<d[id + s]) d[id] = d[id + s];
		}
		__syncthreads();
		s = s / 2;
	}
	__syncthreads();
	TCH[0] = lambda*TCH[1] * TCH[2] / (a[0] + c[0] + d[0]);
	
}


__global__
void Mazuno(cufftDoubleComplex *u, cufftDoubleComplex *up, cufftDoubleComplex *f, cufftDoubleReal *TCH)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;

	up[id].x = u[id].x + 0.5 * TCH[0] * f[id].x;
	up[id].y = u[id].y + 0.5 * TCH[0] * f[id].y;
}

__global__
void Multiple(cufftDoubleComplex *a, cufftDoubleComplex *b)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;

	b[id].x = a[id].x;
	b[id].y = a[id].y;
}

int compare(const void * x1, const void * x2)   // функция сравнения элементов массива
{
	return (*(int*)x1 - *(int*)x2);              // если результат вычитания равен 0, то числа равны, < 0: x1 < x2; > 0: x1 > x2
}

__global__
void Sumforen(cufftDoubleComplex *a,  cufftDoubleReal *c)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*(N / 2 + 1) + idz;

	c[id]=a[id].x * a[id].x + a[id].y * a[id].y;

}

void Null1(cufftDoubleReal *a, int Nk)
{

	for (int i = 0; i < Nk; i++)
		a[i] = 0;
}
