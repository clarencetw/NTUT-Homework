#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv.hpp>

#define grid 50

__global__ void Grid(uchar *d_frame_in,uchar *d_frame_out,int height,int width)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
  
	for(int y=0; y<height; y++)
	{
		for(int z=0; z<3; z++){
			if((x + y) % grid == 0 || (x-y) % grid == 0)
			{
				d_frame_out[y*width+3*x+z] = 255;  
			}
			else
			{
				d_frame_out[y*width+3*x+z] = d_frame_in[y*width+3*x+z];      
			}  
		}
		
	}
}

int main()
{
	IplImage *Image1=cvLoadImage("Koala.jpg", 1);

	cvNamedWindow("Result",CV_WINDOW_AUTOSIZE);
	
	uchar *frame=(uchar*)calloc(Image1->imageSize,sizeof(uchar));
	uchar *dis=(uchar*)calloc(Image1->imageSize,sizeof(uchar));
	
	for(int y=0;y<Image1->height;y++)
	{
		for(int x=0;x<Image1->width;x++)
		{
			for(int z=0;z<3;z++){ 
				frame[y*Image1->widthStep+3*x+z]=Image1->imageData[y*Image1->widthStep+3*x+z];
			}
		} 
	}

	uchar *d_frame_in;
	uchar *d_frame_out;
	cudaMalloc((void**)&d_frame_in,sizeof(uchar)*(Image1->imageSize));
	cudaMalloc((void**)&d_frame_out,sizeof(uchar)*(Image1->imageSize));
	cudaMemcpy(d_frame_in,frame,sizeof(uchar)*(Image1->imageSize),cudaMemcpyHostToDevice);
	Grid<<<16,64>>>(d_frame_in,d_frame_out,Image1->height,Image1->widthStep);
	cudaMemcpy(dis,d_frame_out,sizeof(uchar)*(Image1->imageSize),cudaMemcpyDeviceToHost);

	for(int y=0;y<Image1->height;y++)
	{
		for(int x=0;x<Image1->width;x++)
		{
			for(int z=0;z<3;z++){ 
				Image1->imageData[y*Image1->widthStep+3*x+z]=dis[y*(Image1->widthStep)+3*x+z];
			}

		} 
	}

	cvShowImage("Result",Image1);
	cvWaitKey(0); 
	free(frame);
	free(dis);   
	cudaFree(d_frame_in);
	cudaFree(d_frame_out);
	cvDestroyWindow("Result");
}
