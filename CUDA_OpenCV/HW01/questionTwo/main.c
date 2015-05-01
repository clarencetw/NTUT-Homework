#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/time.h>
using namespace std;

#define ROW 5
#define COL 5

void printArray(int array[ROW][COL], int row, int col)
{
        for(int i=1; i<row-1; i++)
        {
                for(int j=1; j<col-1; j++)
                {
                        printf("%2d ", array[i][j]);
                }
                printf("\n");
        }
        printf("\n");
}

int main()
{
        timeval m_liPerfStart, liPerfNow;
        double m_liPer;
        int array[ROW][COL] =  {{0, 0, 0, 0, 0},
                                {0, 1, 2, 3, 0},
                                {0, 4, 5, 6, 0},
                                {0, 7, 8, 9, 0},
                                {0, 0, 0, 0, 0}};
        int final[ROW][COL] = {0};
        int i, j;

        // get start timer
        gettimeofday(&m_liPerfStart, NULL);

        // do something
        printf("Inital Array\n");
        printArray(array, ROW, COL);
        #pragma omp parallel for
        for(i=1; i<ROW-1; i++)
        {
                #pragma omp parallel for
                for(j=1; j<COL-1; j++)
                {
                        final[i][j] = array[i][j] + array[i-1][j] + array[i][j-1] + array[i+1][j] + array[i][j+1];
                }
        }
        printf("Final Array\n");
        printArray(final, ROW, COL);

        // get new time
        gettimeofday(&liPerfNow, NULL);

        // count Total Need time
        m_liPer = (liPerfNow.tv_sec - m_liPerfStart.tv_sec) * 1000.0;
        m_liPer += (liPerfNow.tv_usec - m_liPerfStart.tv_usec) / 1000.0;
        printf("It took me clicks (%lf ms).\n", m_liPer);

        return 0;
}
