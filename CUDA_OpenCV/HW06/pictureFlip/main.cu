#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv.hpp>

__global__ void flip(char *d_frame_in, char *d_frame_out, int in_height, int in_width, int out_height, int out_width)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;

	for(int y=0; y<in_height; y++)
	{
		for(int z=0; z<3; z++){
			d_frame_out[x*out_width+3*y+z] = d_frame_in[y*in_width+3*x+z];
		}
	}
}

int main()
{
	IplImage *Image1 = cvLoadImage("lena.jpg", 1);
	IplImage *Image2 = cvCreateImage(cvSize(Image1->height, Image1->width), IPL_DEPTH_8U, 3);

	if(Image1 == NULL) return 0;

	cvNamedWindow("readImage", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("newImage", CV_WINDOW_AUTOSIZE);

	char *frame = (char*)calloc(Image1->imageSize,sizeof(char));
	char *dis = (char*)calloc(Image1->imageSize,sizeof(char));

	for(int y=0; y<Image1->height; y++)
	{
		for(int x=0; x<Image1->width; x++)
		{
			for(int z=0; z<3; z++) {
				frame[y*Image1->widthStep+3*x+z] = Image1->imageData[y*Image1->widthStep+3*x+z];
			}
		}
	}

	char *d_frame_in;
	char *d_frame_out;
	cudaMalloc((void**)&d_frame_in, sizeof(char)*(Image1->imageSize));
	cudaMalloc((void**)&d_frame_out, sizeof(char)*(Image1->imageSize));
	cudaMemcpy(d_frame_in, frame, sizeof(char)*(Image1->imageSize), cudaMemcpyHostToDevice);
	flip<<<16,64>>>(d_frame_in, d_frame_out, Image1->height, Image1->widthStep, Image2->height, Image2->widthStep);
	cudaMemcpy(dis, d_frame_out, sizeof(char)*(Image1->imageSize), cudaMemcpyDeviceToHost);

	for(int y=0; y<Image1->height; y++)
	{
		for(int x=0; x<Image1->width; x++)
		{
  			for(int z=0;z<3;z++){
   				Image2->imageData[y*Image1->widthStep+3*x+z] = dis[y*Image1->widthStep+3*x+z];
			}
		}
	}
	cvShowImage("readImage", Image1);
	cvShowImage("newImage", Image2);
	cvWaitKey(0);
	free(frame);
	free(dis);
	//cudaFree(d_frame_in);
	//cudaFree(d_frame_out);
	cvDestroyWindow("readImage");
	cvDestroyWindow("newImage");
}
