
extern "C"
{
	void setTextureFilterMode(bool bLinearFilter);
	void initCuda(void *h_volume, cudaExtent volumeSize);
	void freeCudaBuffers();
	void render_kernel(dim3 gridSize, dim3 blockSize, uint *d_output, uint imageW, uint imageH,
		float density, float brightness, float transferOffset, float transferScale);
	void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
	void copyNearFar(float* nearFar, size_t sizeofNearFar);
	void initTransferTex();
	void bindVolumeTexture(void *h_volume, cudaExtent volumeSize);
	void render_color_temperature_kernel(dim3 gridsize, dim3 blocksize, uint *d_output, uint imagew, uint imageh,
		float density, float brightness, int maxsteps, float step,
		float3 boxmin, float3 boxmax, cudaExtent volumeextent,float voxellength);
	void align_channels(dim3 gridsize, dim3 blocksize, uchar* src, uint* dst, int width, int height);
}