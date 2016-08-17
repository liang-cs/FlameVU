#include "parameters.h"

GLfloat  g_near_far[24];

GLuint g_pbo = 0;     // OpenGL pixel buffer object
GLuint g_tex = 0;     // OpenGL texture object
struct cudaGraphicsResource *g_cuda_pbo_resource = 0; // CUDA Graphics Resource (to transfer PBO)


int g_sampling_max_step = 50000;
float g_sampling_step = 1.5;
cudaExtent g_volume_extent = make_cudaExtent(128, 128, 128);

dim3 g_block_size(16, 16);
dim3 g_grid_size;

float g_density = 0.05f;
float g_brightness = 5.0f;

float3 g_box_min = make_float3(-1.0f, -1.0f, -1.0f);
//float3 g_box_min = make_float3(.0f, .0f, .0f);
float3 g_box_max = make_float3(1.0f, 1.0f, 1.0f);
float g_voxel_length = 1.f;

const char *g_volume_filename = "bin/data/frame.data";
size_t g_volume_size = 0;

// volume data
float* g_volume_data = nullptr;

std::vector<VolumeType*> g_volume_data_list;
const int g_num_frames = 4;
int g_current_frame = 0;

