cufftDoubleReal Falpha(int kx, int ky, int kz, cufftDoubleReal phi1, cufftDoubleReal phi2, cufftDoubleReal phi3)
{
	return double(2*M_PI*kx+sin(2*M_PI*kx)*cos(2.0*phi1))*double(2*M_PI*ky-sin(2*M_PI*ky)*cos(2.0*phi2))*double(2*M_PI*kz-sin(2*M_PI*kz)*cos(2.0*phi3))/(8.0*double(kx)*double(ky)*double(kz));
}
cufftDoubleReal Fbeta(int kx, int ky, int kz, cufftDoubleReal phi1, cufftDoubleReal phi2, cufftDoubleReal phi3)
{
	return double(2*M_PI*kx-sin(2*M_PI*kx)*cos(2.0*phi1))*double(2*M_PI*ky+sin(2*M_PI*ky)*cos(2.0*phi2))*double(2*M_PI*kz-sin(2*M_PI*kz)*cos(2.0*phi3))/(8.0*double(kx)*double(ky)*double(kz));
}
cufftDoubleReal Fgamma(int kx, int ky, int kz, cufftDoubleReal phi1, cufftDoubleReal phi2, cufftDoubleReal phi3)
{
	return double(2*M_PI*kx-sin(2*M_PI*kx)*cos(2.0*phi1))*double(2*M_PI*ky-sin(2*M_PI*ky)*cos(2.0*phi2))*double(2*M_PI*kz+sin(2*M_PI*kz)*cos(2.0*phi3))/(8.0*double(kx)*double(ky)*double(kz));
}

void FindABC(int kx, int ky, int kz, cufftDoubleReal phi1, cufftDoubleReal phi2, cufftDoubleReal phi3, cufftDoubleReal potok, cufftDoubleReal *ABC)
{
	cufftDoubleReal alpha=Falpha(kx,ky,kz,phi1,phi2,phi3);
	cufftDoubleReal beta=Fbeta(kx,ky,kz,phi1,phi2,phi3);
	cufftDoubleReal gamma=Fgamma(kx,ky,kz,phi1,phi2,phi3);
	
	ABC[0]=sqrt(potok/(alpha+beta*(double(kx)*double(kx))/(double(ky)*double(ky))))*(-1.0+double(rand())*2.0/double(RAND_MAX));
	
	ABC[2]=sqrt((potok-ABC[0]*ABC[0]*(alpha+beta*(double(kx)*double(kx))/(double(ky)*double(ky))))/(alpha+gamma*(double(kz)*double(kz))/(double(ky)*double(ky))));
	
	ABC[1]=-ABC[0]*double(kx)/double(ky)-ABC[2]*double(kz)/double(ky);
}

__global__
void Forcing(cufftDoubleReal *vrem, cufftDoubleReal *ux, cufftDoubleReal *uy, cufftDoubleReal *uz, int kx, int ky, int kz, cufftDoubleReal phi1, cufftDoubleReal phi2, cufftDoubleReal phi3, cufftDoubleReal A, cufftDoubleReal B, cufftDoubleReal C)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;
	int idz = blockIdx.z*blockDim.z + threadIdx.z;
	int	id = (N*idx + idy)*N + idz;
	
	if (idx>N / 2) idx = idx - N;
	if (idy>N / 2) idy = idy - N;
	if (idz>N / 2) idz = idz - N;
	
	vrem[id]=ux[id];
	ux[id]=vrem[id]+A*cos(kx*idx*h+phi1)*sin(ky*idy*h+phi2)*sin(kz*idz*h+phi3);
	vrem[id]=uy[id];
	uy[id]=vrem[id]+B*sin(kx*idx*h+phi1)*cos(ky*idy*h+phi2)*sin(kz*idz*h+phi3);
	vrem[id]=uz[id];
	uz[id]=vrem[id]+C*sin(kx*idx*h+phi1)*sin(ky*idy*h+phi2)*cos(kz*idz*h+phi3);	
	
}