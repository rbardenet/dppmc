#include<stdio.h>

double jacobi(int n, double a, double b, double x)
{
  if (n==0)
    {
	return 1.0;
    }
    else if (n==1)
    {
	return  0.5 * (a - b + (a + b + 2.0)*x);
    }
    else
    {

  double p0, p1, a1, a2, a3, a4, p2=0.0;
	int i;
	p0 = 1.0;
	p1 = 0.5 * (a - b + (a + b + 2)*x);

	for(i=1; i<n; ++i){
	    a1 = 2.0*(i+1.0)*(i+a+b+1.0)*(2.0*i+a+b);
	    a2 = (2.0*i+a+b+1.0)*(a*a-b*b);
	    a3 = (2.0*i+a+b)*(2.0*i+a+b+1.0)*(2.0*i+a+b+2.0);
	    a4 = 2.0*(i+a)*(i+b)*(2.0*i+a+b+2.0);
	    p2 = 1.0/a1*( (a2 + a3*x)*p1 - a4*p0);

	    p0 = p1;
	    p1 = p2;
	}
  return p2;
    }
}

