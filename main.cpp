#include "matrix.h"
#include "MyException.h"

int main(int argc,char*argv[])
{
	try{
	int n;
	if (argc==1)
	{
		printf("Input size n=");
		if(scanf("%d",&n)!=1) return -1;
		double*a=(double*)malloc(n*n*sizeof(double));
		for(size_t i=0;i<n;i++)
		{
			for(size_t j=0;j<n;j++)
			{
				if(i==j)
				{
					if(i==n-1)
					{
						a[i*n+i]=-(double)(n-1)/n;
						continue;
					}
					if(i==0)
					{
						a[0]=-1;
						continue;
					}
					else a[i*n+j]=-2;
					continue;
				}
				if(i==j-1) 
				{
					a[i*n+j]=1;
					continue;
				}
				
				if(i==j+1){
					a[i*n+j]=1;
					continue;
				}
				a[i*n+j]=0;					
			}
		}
		Matrix A(n,a), B(n);
		printf("Matrix:\n");
		print(A);
		int k=inverse(A,B);
		if(k==-1) throw MyException("This matrix don't have inverse\n");
		printf("Inverse matrix:\n");
		print(B);
		printf("acc=%.16e\n\n\n",acc(A,B));
		free(a);
		return 0;
	}
	FILE*f;
	f=fopen(argv[1],"r");
	if(!f)throw MyException("Can't open file.txt\n");
	if(fscanf(f,"%d",&n)!=1)
	{
		fclose(f);
		throw MyException("Can't read file.txt\n");
	}
	double*a=(double*)malloc(n*n*sizeof(double));
	if(!a)
	{
		fclose(f);
		throw MyException("Error with memmory allocation\n");
	}
	for(int i=0;i<n*n;i++)
	{
		if(fscanf(f,"%lf",&a[i])!=1)
		{
			fclose(f);
			 throw MyException("Can't read file.txt\n");
		} 
	}
	Matrix A(n,a),B(n);
	printf("Matrix:\n");
	print(A);
	int k=inverse(A,B);
	if(k==-1) throw MyException("This matrix don't have inverse\n");
	printf("Inverse matrix:\n");
	print(B);
	fclose(f);
	free(a);
	printf("acc=%.16e \n",acc(A,B));	
	return 0;
	}
	catch(MyException e)
	{
		e.print();
	}
}