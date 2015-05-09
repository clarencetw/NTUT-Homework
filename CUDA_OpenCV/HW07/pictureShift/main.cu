#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv.cpp>

__device__ double change(int x, int y, int z, uchar *d_frame_in, int height, int weight, int shift)
{
	int widthStep = weight*3;
	
	switch(shift)
	{
	case 0: //lower right shift
		if(x>100 && y>100)
			return d_frame_in[(y-100)*widthStep+3*(x-100)+z];
		break;
	case 1: //lower left shift
		if((x+100)<weight && y>100)
			return d_frame_in[(y-100)*widthStep+3*(x+100)+z];
		break;
	case 2: //upper left shift
		if((x+100)<weight && (y+100)<height)
			return d_frame_in[(y+100)*widthStep+3*(x+100)+z];
		break;
	case 3: //upper right shift
		if(x>100 && (y+100)<height)
			return d_frame_in[(y+100)*widthStep+3*(x-100)+z];
		break;
	}
}

__global__ void Displacement(uchar *d_frame_in, uchar *d_frame_out, int height, int weight)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	double total;
	int shift = 0;

	for(int y=0; y<height; y++)
	{
		for(int z=0; z<3; z++)
		{
			total=change(x,y,z,d_frame_in, height, weight, shift);
			d_frame_out[y*weight*3+3*x+z] = (uchar)total;
		}
	}
} 

int main()  
{
	IplImage *Image1=cvLoadImage("Koala.jpg", 1);
	IplImage *Image2= cvCreateImage(cvSize(Image1->widthStep, Image1->height), IPL_DEPTH_8U,3);
	
	cvNamedWindow("Result", CV_WINDOW_AUTOSIZE);
	
	uchar *frame=(uchar*)calloc(Image1->imageSize, sizeof(uchar));
	uchar *dis=(uchar*)calloc(Image1->imageSize, sizeof(uchar));
	
	for(int y=0; y<(Image1->height)-1; y++)
	{
		for(int x=0; x<Image1->widthStep; x++)
		{
			for(int z=0; z<3; z++)
			{ 
				frame[y*(Image1->widthStep)+3*x+z] = Image1->imageData[y*Image1->widthStep+3*x+z];
			}
		} 
	}

	uchar *d_frame_in;
	uchar *d_frame_out;
	cudaMalloc((void**)&d_frame_in,sizeof(uchar)*(Image1->imageSize));
	cudaMalloc((void**)&d_frame_out,sizeof(uchar)*(Image1->imageSize));
	cudaMemcpy(d_frame_in,frame,sizeof(uchar)*(Image1->imageSize),cudaMemcpyHostToDevice);
	Displacement<<<16,64>>>(d_frame_in,d_frame_out,Image1->height,Image1->width);
	cudaMemcpy(dis,d_frame_out,sizeof(uchar)*(Image1->imageSize),cudaMemcpyDeviceToHost); 
	
	for(int y=0; y<(Image1->height)-1;y++)
	{
		for(int x=0; x<Image1->widthStep; x++)
		{
			for(int z=0; z<3; z++)
			{ 
				Image1->imageData[y*(Image1->widthStep)+3*x+z] = dis[y*(Image1->widthStep)+3*x+z];
			} 
		} 
	}  

	cvShowImage("Result",Image1);
	cvWaitKey(0); 
	free(frame);
	cvDestroyWindow("Result");
}
