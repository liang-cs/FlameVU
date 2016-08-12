#ifndef __SCENE_WIDGET_H__
#define __SCENE_WIDGET_H__

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif


#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QVector3D>
#include <QMatrix4x4>
#include <QTime>
#include <QVector>
#include <QPushButton>


// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>


#include "armanager.h"

class FlameVU;

QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)
QT_FORWARD_DECLARE_CLASS(QOpenGLShader)
QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)


typedef unsigned char VolumeType;


class SceneWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT
public:
	SceneWidget(FlameVU *mw, bool button, const QColor &background);
	~SceneWidget();

protected:
	void resizeGL(int w, int h) Q_DECL_OVERRIDE;
	void paintGL() Q_DECL_OVERRIDE;
	void initializeGL() Q_DECL_OVERRIDE;

	void initAR();
	void updateAR();
	void drawAR();

	// cuda
	void initPixelBuffer();
	void initData();
	void *loadRawFile(char *filename, size_t size);
	int iDivUp(int a, int b);
	void render();

private:

	FlameVU* m_mainWindow;



	// cuda
	GLuint pbo;     // OpenGL pixel buffer object
	GLuint tex;     // OpenGL texture object
	struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

	const char *volumeFilename;
	cudaExtent volumeSize;


	dim3 blockSize;
	dim3 gridSize;
	float invViewMatrix[12];

	float density;
	float brightness;
	float transferOffset;
	float transferScale;
	bool linearFiltering;

	std::ofstream debugInfo_;

	ARManager* ar_mananger_;
	double last_time_stamp_;

	GLuint tex_id_;
	uchar* capture_image_data_;

};






#endif	// __SCENE_WIDGET_H__
