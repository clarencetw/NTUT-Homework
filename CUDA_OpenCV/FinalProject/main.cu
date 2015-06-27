#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp> 
#include <opencv2/gpu/gpu.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv.cpp>

__global__ void embed(uchar *d_frame_paint, uchar *d_frame_frame, uchar *d_frame_out, int paint_height, int paint_width, int frame_height, int frame_width)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int space_height = frame_height/2 - paint_height/2;
	int space_width = frame_width/3/2 - paint_width/3/2;

	for(int y = 0; y<frame_height; y++)
	{
		if(y <= space_height || y >= paint_height + space_height || x <= space_width || x >= paint_width/3 + space_width) {
			d_frame_out[y*frame_width+3*x] = d_frame_frame[y*frame_width+3*x];
			d_frame_out[y*frame_width+3*x+1] = d_frame_frame[y*frame_width+3*x+1];
			d_frame_out[y*frame_width+3*x+2] = d_frame_frame[y*frame_width+3*x+2];
		}
		else
		{
			d_frame_out[y*frame_width+3*x] = d_frame_paint[(y-space_height)*paint_width+3*(x-space_width)];
			d_frame_out[y*frame_width+3*x+1] = d_frame_paint[(y-space_height)*paint_width+3*(x-space_width)+1];
			d_frame_out[y*frame_width+3*x+2] = d_frame_paint[(y-space_height)*paint_width+3*(x-space_width)+2];
		}
	}
}

void loadImage(uchar *out, IplImage *in)
{
	for(int y=0; y<in->height; y++)
	{
		for(int x=0; x<in->width; x++)
		{
			for(int z=0; z<3; z++)
			{	
				out[y*in->widthStep+3*x+z] = in->imageData[y*in->widthStep+3*x+z];
			}
		} 
	}
}

int main()
{
	IplImage *Image1 = cvLoadImage("painting.jpg", 1);
	IplImage *Image2 = cvLoadImage("frames.jpg", 1);
	IplImage *Image3 = cvCreateImage(cvSize(Image2->width, Image2->height), IPL_DEPTH_8U, 3);
	
	cvNamedWindow("Result", CV_WINDOW_AUTOSIZE);
	
	uchar *paint = (uchar*)calloc(Image1->imageSize, sizeof(uchar));
	uchar *frame = (uchar*)calloc(Image2->imageSize, sizeof(uchar));
	uchar *dis = (uchar*)calloc(Image3->imageSize, sizeof(uchar));

	loadImage(paint, Image1);
	loadImage(frame, Image2);
	
	uchar *d_frame_paint, *d_frame_frame;
	uchar *d_frame_out;
	cudaMalloc((void**)&d_frame_paint, sizeof(uchar)*(Image1->imageSize));
	cudaMalloc((void**)&d_frame_frame, sizeof(uchar)*(Image2->imageSize));
	cudaMalloc((void**)&d_frame_out, sizeof(uchar)*(Image3->imageSize));
	cudaMemcpy(d_frame_paint, paint, sizeof(uchar)*(Image1->imageSize), cudaMemcpyHostToDevice);
	cudaMemcpy(d_frame_frame, frame, sizeof(uchar)*(Image2->imageSize), cudaMemcpyHostToDevice);
	embed<<<13,64>>>(d_frame_paint, d_frame_frame, d_frame_out, Image1->height, Image1->widthStep, Image2->height, Image2->widthStep);
	cudaMemcpy(dis, d_frame_out, sizeof(uchar)*(Image3->imageSize), cudaMemcpyDeviceToHost);	
	
	for(int y=0; y<Image3->height; y++)
	{
		for(int x=0; x<Image3->width; x++)
		{
			for(int z=0; z<3; z++)
			{
				Image3->imageData[y*Image3->widthStep+3*x+z] = dis[y*Image3->widthStep+3*x+z];
			}
		} 
	}  
	
	cvShowImage("Result", Image3);
	cvSaveImage("result.jpg", Image3);
	cvWaitKey(0); 
	free(frame);
	free(dis);   
	cudaFree(d_frame_paint);
	cudaFree(d_frame_frame);
	cudaFree(d_frame_out);
	cvDestroyWindow("Result");
}