/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Simple 3D volume renderer

#ifndef _VOLUMERENDER_KERNEL_CU_
#define _VOLUMERENDER_KERNEL_CU_

#include <helper_cuda.h>
#include <helper_math.h>

#include "parameters.h"
#include "planckmapping.h"
#include "macros.h"

typedef unsigned int  uint;
typedef unsigned char uchar;

cudaArray *d_volumeArray = 0;
cudaArray *d_transferFuncArray;



texture<VolumeType, 3, cudaReadModeElementType> tex;         // 3D texture
texture<float4, 1, cudaReadModeElementType>         transferTex; // 1D transfer function texture

typedef struct
{
    float4 m[3];
} float3x4;




__constant__ float3x4 c_invViewMatrix;  // inverse view matrix
__constant__ float3 c_nearFar[8];

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

extern "C"
void copyNearFar(float* nearFar, size_t sizeofNearFar)
{
	checkCudaErrors(cudaMemcpyToSymbol(c_nearFar, nearFar, sizeofNearFar));
}


// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}
__device__ uint4 intTorgba(uint v)
{
	uint4 res;
	res.x = (v & 0x000000ff);
	res.y = (v & 0x0000ff00) >> 8;
	res.z = (v & 0x00ff0000) >> 16;
	res.w = (v & 0xff000000) >> 24;
	return res;

}
__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__ uint rgbaToInt(uchar4 rgba)
{
	return (uint(rgba.w) << 24) | (uint(rgba.z) << 16) | (uint(rgba.y) << 8) | uint(rgba.x);
}



__global__ void
d_render(uint *d_output, uint imageW, uint imageH,
         float density, float brightness,
         float transferOffset, float transferScale)
{
    const int maxSteps = 500;
    const float tstep = 0.01f;
    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;

    for (int i=0; i<maxSteps; i++)
    {
        // read from 3D texture
        // remap position to [0, 1] coordinates
        float sample = tex3D(tex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
        //sample *= 64.0f;    // scale for 10-bit data

        // lookup in transfer function texture
        float4 col = tex1D(transferTex, (sample-transferOffset)*transferScale);
        col.w *= density;

        // "under" operator for back-to-front blending
        //sum = lerp(sum, col, col.w);

        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;
        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);

        // exit early if opaque
        if (sum.w > opacityThreshold)
            break;

        t += tstep;

        if (t > tfar) break;

        pos += step;
    }

    sum *= brightness;
    // write output color
    d_output[y*imageW + x] = rgbaFloatToInt(sum);
}

extern "C"
void setTextureFilterMode(bool bLinearFilter)
{
    tex.filterMode = bLinearFilter ? cudaFilterModeLinear : cudaFilterModePoint;
}

extern "C"
void initCuda(void *h_volume, cudaExtent volumeSize)
{
    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize));

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, volumeSize.width*sizeof(VolumeType), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyHostToDevice;
    checkCudaErrors(cudaMemcpy3D(&copyParams));

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

    // create transfer function texture
    float4 transferFunc[] =
    {
        {  0.0, 0.0, 0.0, 0.0, },
        {  1.0, 0.0, 0.0, 1.0, },
        {  1.0, 0.5, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 0.0, 0.0, },
    };

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray *d_transferFuncArray;
    checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc)/sizeof(float4), 1));
    checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc2));
}

extern "C"
void freeCudaBuffers()
{
    checkCudaErrors(cudaFreeArray(d_volumeArray));
    checkCudaErrors(cudaFreeArray(d_transferFuncArray));
}


extern "C"
void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
                   float density, float brightness, float transferOffset, float transferScale)
{
    d_render<<<gridSize, blockSize>>>(d_output, imageW, imageH, density,
                                      brightness, transferOffset, transferScale);
}

extern "C"
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

__device__
void d_isotropic_scattering(float& color, float& voxelLength, float3& pos,
	float3& boxMin, float3& boxMax, cudaExtent& extent)
{
	color += 1.f / 4 / PI * tex3D(tex,
		(0.5f + ((pos.x - voxelLength) - boxMin.x) / (boxMax.x - boxMin.x) * (extent.width - 1)) / extent.width,
		(0.5f + ((pos.y) - boxMin.y) / (boxMax.y - boxMin.y) * (extent.height - 1)) / extent.height,
		(0.5f + ((pos.z) - boxMin.z) / (boxMax.z - boxMin.z) * (extent.depth - 1)) / extent.depth);

	color += 1.f / 4 / PI * tex3D(tex,
		(0.5f + ((pos.x + voxelLength) - boxMin.x) / (boxMax.x - boxMin.x) * (extent.width - 1)) / extent.width,
		(0.5f + ((pos.y) - boxMin.y) / (boxMax.y - boxMin.y) * (extent.height - 1)) / extent.height,
		(0.5f + ((pos.z) - boxMin.z) / (boxMax.z - boxMin.z) * (extent.depth - 1)) / extent.depth);
	color += 1.f / 4 / PI * tex3D(tex,
		(0.5f + ((pos.x) - boxMin.x) / (boxMax.x - boxMin.x) * (extent.width - 1)) / extent.width,
		(0.5f + ((pos.y - voxelLength) - boxMin.y) / (boxMax.y - boxMin.y) * (extent.height - 1)) / extent.height,
		(0.5f + ((pos.z) - boxMin.z) / (boxMax.z - boxMin.z) * (extent.depth - 1)) / extent.depth);
	color += 1.f / 4 / PI * tex3D(tex,
		(0.5f + ((pos.x) - boxMin.x) / (boxMax.x - boxMin.x) * (extent.width - 1)) / extent.width,
		(0.5f + ((pos.y + voxelLength) - boxMin.y) / (boxMax.y - boxMin.y) * (extent.height - 1)) / extent.height,
		(0.5f + ((pos.z) - boxMin.z) / (boxMax.z - boxMin.z) * (extent.depth - 1)) / extent.depth);
	color += 1.f / 4 / PI * tex3D(tex,
		(0.5f + ((pos.x) - boxMin.x) / (boxMax.x - boxMin.x) * (extent.width - 1)) / extent.width,
		(0.5f + ((pos.y) - boxMin.y) / (boxMax.y - boxMin.y) * (extent.height - 1)) / extent.height,
		(0.5f + ((pos.z - voxelLength) - boxMin.z) / (boxMax.z - boxMin.z) * (extent.depth - 1)) / extent.depth);
	color += 1.f / 4 / PI * tex3D(tex,
		(0.5f + ((pos.x) - boxMin.x) / (boxMax.x - boxMin.x) * (extent.width - 1)) / extent.width,
		(0.5f + ((pos.y) - boxMin.y) / (boxMax.y - boxMin.y) * (extent.height - 1)) / extent.height,
		(0.5f + ((pos.z + voxelLength) - boxMin.z) / (boxMax.z - boxMin.z) * (extent.depth - 1)) / extent.depth);

	color += 1.f / 4 / PI * tex3D(tex,
		(0.5f + ((pos.x + voxelLength) - boxMin.x) / (boxMax.x - boxMin.x) * (extent.width - 1)) / extent.width,
		(0.5f + ((pos.y + voxelLength) - boxMin.y) / (boxMax.y - boxMin.y) * (extent.height - 1)) / extent.height,
		(0.5f + ((pos.z + voxelLength) - boxMin.z) / (boxMax.z - boxMin.z) * (extent.depth - 1)) / extent.depth);
	color += 1.f / 4 / PI * tex3D(tex,
		(0.5f + ((pos.x - voxelLength) - boxMin.x) / (boxMax.x - boxMin.x) * (extent.width - 1)) / extent.width,
		(0.5f + ((pos.y + voxelLength) - boxMin.y) / (boxMax.y - boxMin.y) * (extent.height - 1)) / extent.height,
		(0.5f + ((pos.z + voxelLength) - boxMin.z) / (boxMax.z - boxMin.z) * (extent.depth - 1)) / extent.depth);
	color += 1.f / 4 / PI * tex3D(tex,
		(0.5f + ((pos.x + voxelLength) - boxMin.x) / (boxMax.x - boxMin.x) * (extent.width - 1)) / extent.width,
		(0.5f + ((pos.y - voxelLength) - boxMin.y) / (boxMax.y - boxMin.y) * (extent.height - 1)) / extent.height,
		(0.5f + ((pos.z + voxelLength) - boxMin.z) / (boxMax.z - boxMin.z) * (extent.depth - 1)) / extent.depth);
	color += 1.f / 4 / PI * tex3D(tex,
		(0.5f + ((pos.x + voxelLength) - boxMin.x) / (boxMax.x - boxMin.x) * (extent.width - 1)) / extent.width,
		(0.5f + ((pos.y + voxelLength) - boxMin.y) / (boxMax.y - boxMin.y) * (extent.height - 1)) / extent.height,
		(0.5f + ((pos.z - voxelLength) - boxMin.z) / (boxMax.z - boxMin.z) * (extent.depth - 1)) / extent.depth);
	color += 1.f / 4 / PI * tex3D(tex,
		(0.5f + ((pos.x - voxelLength) - boxMin.x) / (boxMax.x - boxMin.x) * (extent.width - 1)) / extent.width,
		(0.5f + ((pos.y - voxelLength) - boxMin.y) / (boxMax.y - boxMin.y) * (extent.height - 1)) / extent.height,
		(0.5f + ((pos.z + voxelLength) - boxMin.z) / (boxMax.z - boxMin.z) * (extent.depth - 1)) / extent.depth);
	color += 1.f / 4 / PI * tex3D(tex,
		(0.5f + ((pos.x + voxelLength) - boxMin.x) / (boxMax.x - boxMin.x) * (extent.width - 1)) / extent.width,
		(0.5f + ((pos.y - voxelLength) - boxMin.y) / (boxMax.y - boxMin.y) * (extent.height - 1)) / extent.height,
		(0.5f + ((pos.z - voxelLength) - boxMin.z) / (boxMax.z - boxMin.z) * (extent.depth - 1)) / extent.depth);
	color += 1.f / 4 / PI * tex3D(tex,
		(0.5f + ((pos.x - voxelLength) - boxMin.x) / (boxMax.x - boxMin.x) * (extent.width - 1)) / extent.width,
		(0.5f + ((pos.y + voxelLength) - boxMin.y) / (boxMax.y - boxMin.y) * (extent.height - 1)) / extent.height,
		(0.5f + ((pos.z - voxelLength) - boxMin.z) / (boxMax.z - boxMin.z) * (extent.depth - 1)) / extent.depth);
	color += 1.f / 4 / PI * tex3D(tex,
		(0.5f + ((pos.x - voxelLength) - boxMin.x) / (boxMax.x - boxMin.x) * (extent.width - 1)) / extent.width,
		(0.5f + ((pos.y - voxelLength) - boxMin.y) / (boxMax.y - boxMin.y) * (extent.height - 1)) / extent.height,
		(0.5f + ((pos.z - voxelLength) - boxMin.z) / (boxMax.z - boxMin.z) * (extent.depth - 1)) / extent.depth);
}

__global__ void
d_render_color_temperature(uint *d_output, uint imageW, uint imageH,
	float density, float brightness, int maxSteps, float tstep,
	float3 boxMin, float3 boxMax, cudaExtent volumeSize, float voxelLength)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= imageW) || (y >= imageH)) return;

	// calculate eye ray in world space
	Ray eyeRay;
	eyeRay.o = c_nearFar[0] + x * (c_nearFar[3] - c_nearFar[0]) / imageW + y * (c_nearFar[1] - c_nearFar[0]) / imageH;
	eyeRay.d = c_nearFar[4] + x * (c_nearFar[7] - c_nearFar[4]) / imageW + y * (c_nearFar[5] - c_nearFar[4]) / imageH;
	eyeRay.d = normalize(eyeRay.d - eyeRay.o);

	// find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

	float4 sum = make_float4(0.0f, 0.f, 0.f, 0.5f);

	if (!hit)
	{
		//d_output[y*imageW + x] = rgbaFloatToInt(sum);
		return;
	}

	//if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

	// march along ray from front to back, accumulating color
	
	float t = tnear;
	float3 pos = eyeRay.o + eyeRay.d*tnear;
	float3 step = eyeRay.d*tstep;
	float3 normPos;
	float sample = 0;

	for (int i = 0; i<maxSteps; i++)
	{
		// read from 3D texture
		// remap position to [0, 1] coordinates
		// +0.5是由于cuda 3dtex的存储方式
		//float sample = tex3D(tex, (0.5 + (pos.x - boxMin.x) / (boxMax.x - boxMin.x) * (volumeSize.width - 1)) / volumeSize.width, 
		//	(0.5 + (pos.y - boxMin.y) / (boxMax.y - boxMin.y) * (volumeSize.height - 1)) / volumeSize.height,
		//	(0.5 + (pos.z - boxMin.z) / (boxMax.z - boxMin.z) * (volumeSize.depth - 1)) / volumeSize.depth);


		normPos.x = (0.5f + ((pos.x) - boxMin.x) / (boxMax.x - boxMin.x) * (volumeSize.width - 1)) / volumeSize.width;
		normPos.y = (0.5f + ((pos.y) - boxMin.y) / (boxMax.y - boxMin.y) * (volumeSize.height - 1)) / volumeSize.height;
		normPos.z = (0.5f + ((pos.z) - boxMin.z) / (boxMax.z - boxMin.z) * (volumeSize.depth - 1)) / volumeSize.depth;

		sample = tex3D(tex, normPos.x, normPos.y, normPos.z);

		if (sample > CLAMP_TEMPERATURE_LOW)
		{

			d_isotropic_scattering(sample, voxelLength, pos, boxMin, boxMax, volumeSize);

			float4 col = tex1D(transferTex,
				(sample - CLAMP_TEMPERATURE_LOW) / (CLAMP_TEMPERATURE_HIGH - CLAMP_TEMPERATURE_LOW));

			// under operator for front-to-back blending
			sum.x = sum.w * col.w * col.x + sum.x;
			sum.y = sum.w * col.w * col.y + sum.y;
			sum.z = sum.w * col.w * col.z + sum.z;
			sum.w = (1 - col.w) * sum.w;
		}

		t += tstep;

		if (t > tfar) break;

		pos += step;
	}

	sum *= brightness;

	// write output color
	uint4 src = intTorgba(d_output[y * imageW + x]);
	sum.x = src.x / 255.f + sum.x * sum.w;
	sum.y = src.y / 255.f + sum.y * sum.w;
	sum.z = src.z / 255.f + sum.z * sum.w;
	sum.w = 1;
	d_output[y*imageW + x] = rgbaFloatToInt(sum);
}



extern "C"
void render_color_temperature_kernel(dim3 gridsize, dim3 blocksize, uint *d_output, uint imagew, uint imageh,
	float density, float brightness,
	int maxsteps, float step,
	float3 boxmin, float3 boxmax, cudaExtent volumeextent, float voxellength)
{
	d_render_color_temperature << <gridsize, blocksize >> >(d_output, imagew, imageh,
		density, brightness, maxsteps, step,
		boxmin, boxmax, volumeextent, voxellength);
	cudaThreadSynchronize();
}

extern "C"
void initTransferTex()
{
	PlanckMapping mp;
	size_t transferFuncSize = CLAMP_TEMPERATURE_HIGH - CLAMP_TEMPERATURE_LOW + 1;
	float4* transferFunc = (float4*)malloc(transferFuncSize * sizeof(float4));
	for (int i = 0; i < transferFuncSize; ++i)
	{
		transferFunc[i].x = PlanckMapping::spectrum_[i].c[0];
		transferFunc[i].y = PlanckMapping::spectrum_[i].c[1];
		transferFunc[i].z = PlanckMapping::spectrum_[i].c[2];
		// to do
		transferFunc[i].w = 0.05f;

	}

	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
	cudaArray *d_transferFuncArray;
	checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, transferFuncSize, 1));
	checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, transferFuncSize * sizeof(float4), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMallocArray(&d_transferFuncArray, &channelDesc2, sizeof(transferFunc) / sizeof(float4), 1));
	//checkCudaErrors(cudaMemcpyToArray(d_transferFuncArray, 0, 0, transferFunc, sizeof(transferFunc), cudaMemcpyHostToDevice));

	transferTex.filterMode = cudaFilterModeLinear;
	transferTex.normalized = true;    // access with normalized texture coordinates
	transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

														 // Bind the array to the texture
	checkCudaErrors(cudaBindTextureToArray(transferTex, d_transferFuncArray, channelDesc2));
}

extern "C"
void bindVolumeTexture(void *h_volume, cudaExtent volumeextent)
{
	// create 3D array
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<VolumeType>();
	if (0 == d_volumeArray)
	{
		checkCudaErrors(cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeextent));
	}

	// copy data to 3D array
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(h_volume, volumeextent.width*sizeof(VolumeType), volumeextent.width, volumeextent.height);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent = volumeextent;
	copyParams.kind = cudaMemcpyHostToDevice;
	checkCudaErrors(cudaMemcpy3D(&copyParams));

	// set texture parameters
	tex.normalized = true;                      // access with normalized texture coordinates
	tex.filterMode = cudaFilterModeLinear;      // linear interpolation
	tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
	tex.addressMode[1] = cudaAddressModeClamp;

	// bind array to 3D texture
	checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

}

__global__ void
d_align_channels(uchar* src, uint* dst, int width, int height)
{
	uint x = blockIdx.x*blockDim.x + threadIdx.x;
	uint y = blockIdx.y*blockDim.y + threadIdx.y;

	if ((x >= width) || (y >= height)) return;

	uchar4 res = make_uchar4(0, 0, 0, 0);
	res.x = src[(y * width + x) * 3 + 2];
	res.y = src[(y * width + x) * 3 + 1];
	res.z = src[(y * width + x) * 3 + 0];
	res.w = 255;
	dst[(height - 1 - y) * width + x] = rgbaToInt(res);
}

extern "C"
void align_channels(dim3 gridsize, dim3 blocksize, uchar* src, uint* dst, int width, int height)
{
	d_align_channels << <gridsize, blocksize >> >(src, dst, width, height);
	cudaThreadSynchronize();
}

#endif // #ifndef _VOLUMERENDER_KERNEL_CU_
