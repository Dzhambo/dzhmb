#include"matrix.h"


Matrix::Matrix(): _l(0),_mat(0){}

Matrix::Matrix(size_t l,double*a)
{
	_l=l;
	_mat=(double*)malloc(l*l*sizeof(double));
	if(!_mat) throw 1;
	for(size_t i=0;i<l*l;i++) _mat[i]=a[i];
}

Matrix::Matrix(const size_t&l)
{
	_l=l;
	_mat=(double*)malloc(l*l*sizeof(double));
	if(!_mat) throw 1;
	for(size_t i=0;i<l;i++){
		for(size_t j=0;j<l;j++)
		{
			if(i==j){_mat[i*l+j]=1;}
			else _mat[i*l+j]=0;
		}
	}
}

Matrix::Matrix(const Matrix&a)
{
	_l=a._l;
	_mat=(double*)malloc(_l*_l*sizeof(double));
	if(!_mat) throw 1;
	for(size_t i=0;i<_l*_l;i++) _mat[i]=a._mat[i];
}

Matrix::~Matrix()
{
	free(_mat);
}

Matrix Matrix:: operator+(const Matrix&a)
{
	if(_l!=a._l) throw 1;
	Matrix b(*this);
	for(size_t i=0;i<_l*_l;i++) b._mat[i]+=a._mat[i];	
	return b;
}
		
Matrix Matrix:: operator-(const Matrix&a)
{
	if(_l!=a._l) throw 1;
	Matrix b(*this);
	for(size_t i=0;i<_l*_l;i++) b._mat[i]-=a._mat[i];
	return b;
}

Matrix Matrix:: operator*(Matrix&a)
{
	if(_l!=a._l) throw 1;
	Matrix b(_l);
	double sum=0;
	for(size_t i=0;i<_l;i++){
		for(size_t j=0;j<_l;j++){
			for(size_t k=0;k<_l;k++){
				sum+=_mat[i*_l+k]*a._mat[k*_l+j];
			}
			b._mat[i*_l+j]=sum;
			sum=0;
		}
	}
	return b;
}


Matrix & Matrix:: operator=(const Matrix&a)
{
	if(_mat!=0) free(_mat);
	_l=a._l;
	_mat=(double*)malloc(_l*_l*sizeof(double));
	if(!_mat) throw 1;
	for(size_t i=0;i<_l*_l;i++) _mat[i]=a._mat[i];
}
	
Matrix & Matrix::c1(const size_t&n,const size_t&m,const double&f)
{
	if(n>_l||m>_l) throw 1;
	for(size_t i=0;i<_l;i++){
		_mat[n*_l+i]-=f*_mat[m*_l+i];
	}
}
Matrix & Matrix::c2(const size_t&n,const size_t&m)
{
	double*a;
	if(n>_l||m>_l) throw 1;
	a=(double*)malloc(_l*sizeof(double));
	if(!a) throw 1;
	for(size_t i=0;i<_l;i++){
		a[i]=_mat[n*_l+i];
	}
	for(size_t i=0;i<_l;i++){
		_mat[n*_l+i]=_mat[m*_l+i];
	}
	for(size_t i=0;i<_l;i++){
		_mat[m*_l+i]=a[i];
	}	
	free(a);

}
Matrix & Matrix::c3(const size_t&n,const double&f)
{
	if(n>_l) throw 1;
	for(size_t i=0;i<_l;i++){
		_mat[n*_l+i]=f*_mat[n*_l+i];

	}
}

Matrix & Matrix::c4(const size_t&n,const size_t&m)
{
	double*a;
	if(n>_l||m>_l) throw 1;
	a=(double*)malloc(_l*sizeof(double));
	if(!a) throw 1;
	for(size_t i=0;i<_l;i++){
		a[i]=_mat[i*_l+n];
	}
	for(size_t i=0;i<_l;i++){
		_mat[i*_l+n]=_mat[i*_l+m];
	}
	for(size_t i=0;i<_l;i++){
		_mat[i*_l+m]=a[i];
	}	
	free(a);

}

double Matrix::  get_norma()
{
	double sum=0;
	for(size_t i=0;i<_l;i++){
		for(size_t j=0;j<_l;j++){
			sum+=_mat[i*_l+j]*_mat[i*_l+j];
		}
	}
	return sqrt(sum);
}

double Matrix:: get_elem(const size_t i,const size_t j)
{
	return _mat[i*_l+j];
}

void print(const Matrix&a)
{
	int i,min=5;
	if(min>a._l) min=a._l;
	for(i=0;i<min;i++){
		printf("|	");
		for(size_t j=0;j<min;j++){
			if (j!=a._l-1) {printf("%f	", a._mat[i*a._l+j]);}
				else printf("%f	", a._mat[i*a._l+j]);
		}
		if(min<a._l) printf("...	");
		printf("|\n");
	}
	if(i==min && i!=a._l)
			{
				printf("|	");
		     	for(i=0;i<min;i++) printf("...		");
				printf("...	|\n");
			}
	printf("\n\n\n\n");
}
void todo(Matrix&A,int*&a)
{
	Matrix B(A._l);
	for(size_t i=0;i<A._l;i++)
	{
		size_t j=a[i];
		for(size_t k=0;k<A._l;k++)
		{
			B._mat[j*A._l+k]=A._mat[i*A._l+k];
		}
	}
	A=B;
}
	
	
int inverse(const Matrix&c,Matrix&b)
{
	double max,n;
	Matrix a(c);
	size_t num1,num2,l;
	int*d=(int*)malloc(a._l*sizeof(int));
	if(!d) throw 1;
	for(int i=0;i<a._l;i++){d[i]=i;}	
	for(size_t i=0;i<a._l;i++)
	{
		max=fabs(a._mat[i*a._l+i]);
		num1=i;
		num2=i;
		for(size_t k=i;k<a._l;k++){
			for(size_t j=i;j<a._l;j++)
		    {
			     if(fabs(a._mat[j*a._l+k])>max)
			     {
					 num1=j;	
					 num2=k;
				     max=fabs(a._mat[j*a._l+k]);
				 }
			}
		}
		if(max>0)
		{
			a.c2(i,num1);
			a.c4(i,num2);
			l=d[num2];
			d[num2]=d[i];
			d[i]=l;
			b.c2(i,num1);
			for(size_t j=0;j<a._l;j++)
			{
				if(j==i) continue;
				n=a._mat[j*a._l+i]/a._mat[i*a._l+i];
				//a.c1(j,i,n);
				for(size_t k=i;k<a._l;k++) a._mat[j*a._l+k]-=n*a._mat[i*a._l+k];
				b.c1(j,i,n);
			}
			n=a._mat[i*a._l+i];
			a.c3(i,1/n);
			b.c3(i,1/n);
			
		}
		else 
		{
			free(d);return -1;
		}
	}
	todo(b,d);
	free(d);
	return 0;
}

////////////Parallel/////////////

void*fd(void*potocs)
{
	double sum,dob=0;
	arg*flow=(arg*)potocs;
	if(flow->i > flow->n) return NULL;
	flow->cpu_time=get_cpu_time();
	flow->abs_time=get_abs_time();
	for(int j=0;(flow->i)+j*(flow->p) < flow->n;j++)
	{
		for(int k=0;k<flow->n;k++) 
		{
			sum=0;dob=0;
			for(int m=0;m<flow->n;m++)
				sum+=(flow->A->_mat[((flow->i)+j*(flow->p))* (flow->n) +m])*(flow->B->_mat[m * (flow->n) +k]);
			if(flow->i+j*flow->p==k) dob=1.;
			flow->c[((flow->i)+j*(flow->p) )*(flow->n)+k]=(sum-dob)*(sum-dob);
		}
	} 
	flow->cpu_time=get_cpu_time()-flow->cpu_time;
	flow->abs_time=get_abs_time()-flow->abs_time;
	return NULL;
}
			
	

double acc(Matrix&A,Matrix&B)
{
	if(A._l==1) return fabs(A._mat[0]);
	double sum=0;
	int n=A._l;
	double*c=new double[n*n];
	arg*potocs=new arg[NUM];
	pthread_t*tid=new pthread_t[NUM];
	for(int i=0;i<NUM;i++)
	{
		potocs[i].A=&A;
		potocs[i].B=&B;
		potocs[i].i=i;
		potocs[i].n=n;
		potocs[i].p=NUM;
		potocs[i].c=c;
		potocs[i].cpu_time=0.;
		potocs[i].abs_time=0.;
	}
	for(int i=1;i<NUM;i++)
	{
		if(pthread_create(tid+i,0,&fd,potocs+i))
		{
			delete[]potocs;
			delete[]c;
			delete[]tid;
			throw("Flow was not created");
		}
	}
	fd(potocs);
	void *q;
	for(int i=1;i<NUM;i++)
	pthread_join(tid[i],&q);
   for(int i=0;i<n*n;i++) sum+=c[i];
	double time=potocs[0].abs_time;
	for(int i=0;i<NUM;i++)
	{
		if(time<potocs[i].abs_time) time=potocs[i].abs_time;
	}
	printf("Time = %f\n",time);
	delete[]tid;
	delete[]potocs;
	delete[]c;
	return sqrt(sum);
}

double get_cpu_time()
{
    struct rusage u;
    getrusage(RUSAGE_THREAD,&u);
    return (double)u.ru_utime.tv_sec+(double)u.ru_utime.tv_usec/1e6;
}

double get_abs_time()
{
    struct timeval tm;
    gettimeofday(&tm,0);
    return (double)tm.tv_sec+(double)tm.tv_usec/1e6;
}
	
	
	
		
		
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	