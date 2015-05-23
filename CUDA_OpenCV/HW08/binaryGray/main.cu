#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2/gpu/gpu.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv.cpp>

__global__ void gray(uchar *d_frame_in, uchar *d_frame_out, int height, int width)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	double total;
	for(int y=0; y<height; y++)
	{
		total = 0.299*d_frame_in[(y)*width+3*(x)] + 0.587*d_frame_in[(y)*width+3*(x)+1] + 0.114*d_frame_in[(y)*width+3*(x)+2]; 
		if(total > 128) {
			total = 255;
		} else {
			total = 0;
		}
		d_frame_out[y*(width/3)+x] = (uchar)total;
	}
}

int main()
{
	IplImage *Image1 = cvLoadImage("Koala.jpg", 1);
	IplImage *Image2 = cvCreateImage(cvSize(Image1->width, Image1->height), IPL_DEPTH_8U, 1);
	
	cvNamedWindow("Result", CV_WINDOW_AUTOSIZE);
	
	uchar *frame=(uchar*)calloc(Image1->imageSize,sizeof(uchar));
	uchar *dis=(uchar*)calloc(Image2->imageSize,sizeof(uchar));
	
	for(int y=0; y<Image1->height; y++)
	{
		for(int x=0; x<Image1->width; x++)
		{
			for(int z=0; z<3; z++){	
				frame[y*Image1->widthStep+3*x+z]=Image1->imageData[y*Image1->widthStep+3*x+z];
			}
		} 
	}
	
	uchar *d_frame_in;
	uchar *d_frame_out;
	cudaMalloc((void**)&d_frame_in,sizeof(uchar)*(Image1->imageSize));
	cudaMalloc((void**)&d_frame_out,sizeof(uchar)*(Image2->imageSize));
	cudaMemcpy(d_frame_in, frame, sizeof(uchar)*(Image1->imageSize), cudaMemcpyHostToDevice);
	gray<<<16,64>>>(d_frame_in, d_frame_out, Image1->height, Image1->widthStep);
	cudaMemcpy(dis, d_frame_out, sizeof(uchar)*(Image2->imageSize), cudaMemcpyDeviceToHost);	
	
	for(int y=0; y<Image2->height; y++)
	{
		for(int x=0; x<Image2->width; x++)
		{
			Image2->imageData[y*Image2->width+3*x]=dis[y*Image2->width+3*x];
		} 
	}  
	
	cvShowImage("Result",Image2);
	cvWaitKey(0); 
	free(frame);
	free(dis);   
	cudaFree(d_frame_in);
	cudaFree(d_frame_out);
	cvDestroyWindow("Result");
}
