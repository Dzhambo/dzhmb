#include <iostream>
#include <sstream>
#include <numeric>
#include <map>
#include <set>
#include <vector>
using namespace std;



class Rational{
public:
	Rational(){
		numerator = 0;
		denominator = 1;
	}
	Rational(int num, int den){
		if (den == 0){
			throw invalid_argument("Invalid argument");
		}
		if ((num>0 && den<0) || (num<0 && den<0)){
			num*=-1;
			den*=-1;
		}
		if (num==0){
			den = 1;
		}
		int gcd_value = gcd(num, den);
		numerator = num/gcd_value;
		denominator = den/gcd_value;
	}
	int Numerator() const{
		return numerator;
	}
	int Denominator() const{
		return denominator;
	}
private:
	int numerator;
	int denominator;
};


bool operator==(const Rational& a, const Rational& b){
	return (a.Numerator() == b.Numerator() && a.Denominator() == b.Denominator());
	}

bool operator<(const Rational& a, const Rational& b){
	return (a.Numerator()*b.Denominator() < a.Denominator()*b.Numerator());
}

bool operator>(const Rational& a, const Rational& b){
	return (a.Numerator()*b.Denominator() > a.Denominator()*b.Numerator());
}

Rational operator-(const Rational& a, const Rational& b){
			Rational res = Rational(a.Numerator()*b.Denominator() - b.Numerator()*a.Denominator(), b.Denominator()*a.Denominator());
			return res;
		}

Rational operator+(const Rational& a, const Rational& b){
		Rational res = Rational(a.Numerator()*b.Denominator() + b.Numerator()*a.Denominator(), b.Denominator()*a.Denominator());
		return res;
	}

Rational operator*(const Rational& a, const Rational& b){
	Rational res = Rational(a.Numerator()*b.Numerator(), a.Denominator()*b.Denominator());
	return res;
}

Rational operator/(const Rational& a, const Rational& b){
	if (b.Numerator()==0){
		throw domain_error("Division by zero");
	}
	Rational res = Rational(a.Numerator()*b.Denominator(), b.Numerator()* a.Denominator());
		return res;
}


istream& operator>>(istream& stream, Rational& r){
	int num, den;
	char sep;
	if (stream && stream >> num && stream >> sep && stream >> den && sep=='/'){
		r = Rational(num,den);
	}else{
		throw invalid_argument("Invalid argument");
	}

	return stream;
}

ostream& operator<<(ostream& stream, const Rational& r){
	stream << r.Numerator() << "/" << r.Denominator();
	return stream;
}


