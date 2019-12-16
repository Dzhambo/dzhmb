#ifndef _MYEXCEPTION_H_
#define _MYEXCEPTION_H_

#include <stdio.h>
#include <string.h>

class MyException
{
	char str[128];
	public:
	MyException(const char*m);
    MyException(const MyException &a);
	void print();
};

#endif
