__global__
void Helicity(cufftDoubleReal *ax, cufftDoubleReal *ay, cufftDoubleReal *az, cufftDoubleReal *ZdY, cufftDoubleReal *YdZ, cufftDoubleReal *XdZ, cufftDoubleReal *ZdX, cufftDoubleReal *YdX, cufftDoubleReal *XdY, cufftDoubleReal *c)
{
int idx = blockIdx.x*blockDim.x + threadIdx.x;
int idy = blockIdx.y*blockDim.y + threadIdx.y;
int idz = blockIdx.z*blockDim.z + threadIdx.z;
int	id = (N*idx + idy)*N + idz;

c[id] = ax[id]*(ZdY[id]-YdZ[id])+ay[id]*(XdZ[id]-ZdX[id])+az[id]*(YdX[id]-YdX[id]);
}



__global__
void CrossHelicity(cufftDoubleReal *ax, cufftDoubleReal *ay, cufftDoubleReal *az, cufftDoubleReal *bx, cufftDoubleReal *by, cufftDoubleReal *bz, cufftDoubleReal *c)
{
int idx = blockIdx.x*blockDim.x + threadIdx.x;
int idy = blockIdx.y*blockDim.y + threadIdx.y;
int idz = blockIdx.z*blockDim.z + threadIdx.z;
int	id = (N*idx + idy)*N + idz;

c[id] = ax[id]*bx[id] + ay[id]*by[id] + az[id]*bz[id];
}
