#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
using namespace std;

#define MAX 11

void printArray(int array[], int max)
{
        for(int i=1; i<max; i++)
        {
                printf("%d ", array[i]);
        }
        printf("\n");
}

int main()
{
        timeval m_liPerfStart, liPerfNow;
        double m_liPer;
        int array[MAX] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
        int final[MAX] = {0};
        int i;

        // get start timer
        gettimeofday(&m_liPerfStart, NULL);

        // do something
        printArray(array, 10);
        #pragma omp parallel for
        for(i=1; i<MAX-1; i++)
        {
                final[i] = array[i-1] + array[i] + array[i+1];
        }
        printArray(final, 10);

        // get new time
        gettimeofday(&liPerfNow, NULL);

        // count Total Need time
        m_liPer = (liPerfNow.tv_sec - m_liPerfStart.tv_sec) * 1000.0;
        m_liPer += (liPerfNow.tv_usec - m_liPerfStart.tv_usec) / 1000.0;
        printf("It took me clicks (%lf ms).\n", m_liPer);

        return 0;
}
