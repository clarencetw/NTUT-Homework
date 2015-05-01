#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
using namespace std;

int main()
{
        timeval m_liPerfStart, liPerfNow;
        double m_liPer;

        // get start timer
        gettimeofday(&m_liPerfStart, NULL);

        // do something
        int a=0;
        #pragma omp parallel for
        for(int i=0; i<10000000; i++) {
          a+=i;
        }

        // get new time
        gettimeofday(&liPerfNow, NULL);

        // count Total Need time
        m_liPer = (liPerfNow.tv_sec - m_liPerfStart.tv_sec) * 1000.0;
        m_liPer += (liPerfNow.tv_usec - m_liPerfStart.tv_usec) / 1000.0;
        printf("It took me clicks (%lf ms).\n", m_liPer);

        return 0;
}
