#include "MyException.h"

MyException::MyException(const MyException &a)
{
    strncpy(str,a.str,127);
}


MyException :: MyException(const char*m)
{
	strncpy(str,m,127);
}

void MyException:: print()
{
	printf("Error! %s\n",str);
}
	