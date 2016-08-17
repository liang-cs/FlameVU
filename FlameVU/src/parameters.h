#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include "headers.h"
#include "macros.h"


typedef float VolumeType;

extern GLfloat  g_near_far[];


extern GLuint g_pbo;     // OpenGL pixel buffer object
extern GLuint g_tex;     // OpenGL texture object
extern struct cudaGraphicsResource *g_cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)


extern int g_sampling_max_step;
extern float g_sampling_step;
extern cudaExtent g_volume_extent;

extern dim3 g_block_size;
extern dim3 g_grid_size;
extern float g_density;
extern float g_brightness;
extern float3 g_box_min, g_box_max;
extern float g_voxel_length;

extern size_t g_volume_size;
extern const char *g_volume_filename;

// volume data
extern float* g_volume_data;


extern std::vector<VolumeType*> g_volume_data_list;
extern const int g_num_frames;
extern int g_current_frame;


#endif //  __PARAMETERS_H__