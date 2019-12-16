#ifndef _MATRIX_H_
#define _MATRIX_H_

#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>
#include<errno.h>
#include<math.h>
#include <sys/time.h>
#include <sys/resource.h>

#define NUM 4

class Matrix
{
	size_t _l;
	double*_mat;
public:
	Matrix();
	Matrix(size_t l,double*a);
	Matrix(const size_t&l);
	Matrix(const Matrix&a);
	~Matrix();
	Matrix operator+(const Matrix&a);
	Matrix operator-(const Matrix&a);
	Matrix operator*(Matrix&a);
	Matrix &c4(const size_t&n,const size_t&m);
	Matrix&operator=(const Matrix&a);
	Matrix&c1(const size_t&n,const size_t&m,const double&f);
	Matrix&c2(const size_t&n,const size_t&m);
	Matrix&c3(const size_t&n,const double&f);
	double get_norma();
	double get_elem(const size_t i,const size_t j);
	friend void print(const Matrix&a);
	friend int inverse(const Matrix&a,Matrix&b);
	friend void todo(Matrix&A,int*&a);
	friend double acc(Matrix&A,Matrix&B);
	friend void*fd(void*potocs);
};


struct arg
{
	Matrix*A;
	Matrix*B;
	int n;
	int i;
	int p;
	double*c;
	double cpu_time;
	double abs_time;
};

double get_cpu_time();
double get_abs_time();

#endif