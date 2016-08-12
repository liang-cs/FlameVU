#include "scenewidget.h"

#include <QPainter>
#include <QPaintEngine>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QCoreApplication>
#include <cmath>
#include <opencv/cv.h>
#include <easyar/base.hpp>

#include "functions.h"
#include "macros.h"
#include "flamevu.h"



SceneWidget::SceneWidget(FlameVU *mw, bool button, const QColor &background)
	: m_mainWindow(mw),
	pbo(0),
	tex(0)
{
	//setMinimumSize(300, 250);
	//setFixedSize(640, 480);

	volumeFilename = "Bucky.raw";
	volumeSize = make_cudaExtent(32, 32, 32);


	blockSize.x = 16;
	blockSize.y = 16;
	capture_image_data_ = 0;
	debugInfo_.open("debug.txt");
}

SceneWidget::~SceneWidget()
{
	debugInfo_.close();
	SAFE_DELETE_ARRAY(capture_image_data_);

	// And now release all OpenGL resources.
	makeCurrent();
	//delete program;
	doneCurrent();
}

void SceneWidget::initAR()
{
	ar_mananger_.setup();
	ar_mananger_.initCamera();
	bool status = ar_mananger_.loadFromJsonFile("data/target.json", "flame");
	if (!status)
	{
		std::cout << "failed while loading json file" << std::endl;
	}
	ar_mananger_.start();
	last_time_stamp_ = 0.0;
	capture_image_data_ = new uchar[ar_mananger_.imageSize()[0] * ar_mananger_.imageSize()[1] * 3];

}

void SceneWidget::initializeGL()
{
	initAR();

	initializeOpenGLFunctions();
	glewInit();

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);

	glGenTextures(1, &tex_id_);
	glBindTexture(GL_TEXTURE_2D, tex_id_);        
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	//#define PROGRAM_VERTEX_ATTRIBUTE 0
	//#define PROGRAM_TEXCOORD_ATTRIBUTE 1
	//
	//	QOpenGLShader *vshader = new QOpenGLShader(QOpenGLShader::Vertex, this);
	//	const char *vsrc =
	//		"void main()\n"
	//		"{\n"
	//		"	gl_TexCoord[0] = gl_MultiTexCoord0;\n"
	//		"	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
	//		"}\n";
	//	vshader->compileSourceCode(vsrc);
	//
	//	QOpenGLShader *fshader = new QOpenGLShader(QOpenGLShader::Fragment, this);
	//	const char *fsrc =
	//		"uniform sampler2D color_texture;\n"
	//		"void main(void)\n"
	//		"{\n"
	//		"    gl_FragColor = texture2D(color_texture, gl_TexCoord[0].st);\n"
	//		"}\n";
	//
	//	fshader->compileSourceCode(fsrc);
	//
	//	program = new QOpenGLShaderProgram;
	//	program->addShader(vshader);
	//	program->addShader(fshader);
	//	program->link();
	//
	//	program->bind();

	//initPixelBuffer();
	//initData();
}

void SceneWidget::paintGL()
{



	glClearColor(0.0, 0.0, 0.0, 0.5);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if 0 // cuda render test
	GLfloat modelView[16];
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glRotatef(0., 1.0, 0.0, 0.0);
	glRotatef(0., 0.0, 1.0, 0.0);
	glTranslatef(0.f, 0.f, 0.f);
	glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
	glPopMatrix();

	invViewMatrix[0] = modelView[0];
	invViewMatrix[1] = modelView[4];
	invViewMatrix[2] = modelView[8];
	invViewMatrix[3] = modelView[12];
	invViewMatrix[4] = modelView[1];
	invViewMatrix[5] = modelView[5];
	invViewMatrix[6] = modelView[9];
	invViewMatrix[7] = modelView[13];
	invViewMatrix[8] = modelView[2];
	invViewMatrix[9] = modelView[6];
	invViewMatrix[10] = modelView[10];
	invViewMatrix[11] = modelView[14];

	render();

	// display results
	glClear(GL_COLOR_BUFFER_BIT);

	// draw image from PBO
	glDisable(GL_DEPTH_TEST);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	// copy from pbo to texture
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->width(), this->height(), GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// draw textured quad
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0);
	glVertex2f(0, 0);
	glTexCoord2f(1, 0);
	glVertex2f(1, 0);
	glTexCoord2f(1, 1);
	glVertex2f(1, 1);
	glTexCoord2f(0, 1);
	glVertex2f(0, 1);
	glEnd();

	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

#endif // cuda render test

	updateAR();
	drawAR();

	update();
}




void SceneWidget::resizeGL(int width, int height)
{
	initPixelBuffer();

	// calculate new grid size
	gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

	glViewport(0, 0, width, height);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, this->width(), 0, this->height(), -100, 100);
}

void SceneWidget::updateAR()
{
	EasyAR::Frame frame = ar_mananger_.augmenter_.newFrame();
	EasyAR::Image image = frame.images()[0];
	glDisable(GL_CULL_FACE);
	glDisable(GL_LIGHTING);
	glViewport(0, 0, this->width(), this->height());
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//gluOrtho2D(0, this->width(), 0, this->height());
	glOrtho(0, this->width(), 0, this->height(), -100, 100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glBindTexture(GL_TEXTURE_2D, tex_id_);
	glEnable(GL_TEXTURE_2D);

	

	for (int y = 0; y < image.height(); ++y)
	{
		for (int x = 0; x < image.width(); ++x)
		{
			capture_image_data_[3 * (y * image.width() + x) + 0]
				= *((uchar*)image.data() + 3 * (y * image.width() + x) + 2);
			capture_image_data_[3 * (y * image.width() + x) + 1]
				= *((uchar*)image.data() + 3 * (y * image.width() + x) + 1);
			capture_image_data_[3 * (y * image.width() + x) + 2]
				= *((uchar*)image.data() + 3 * (y * image.width() + x) + 0);

		}
	}
	if (last_time_stamp_ != frame.timeStamp())
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width(), image.height(), 
			0, GL_RGB, GL_UNSIGNED_BYTE, capture_image_data_);

		last_time_stamp_ = frame.timeStamp();
	}

	glColor3d(1, 1, 1);

	// make a rectangle
	glBegin(GL_QUADS);

	// top left
	glTexCoord2i(0, 0);
	glVertex3f(0, this->height(), 0);
	// bottom left
	glTexCoord2i(0, 1);
	glVertex3f(0, 0, 0);
	// bottom right
	glTexCoord2i(1, 1);
	glVertex3f(this->width(), 0, 0);
	// top right
	glTexCoord2i(1, 0);
	glVertex3f(this->width(), this->height(), 0);

	glEnd();

	glDisable(GL_TEXTURE_2D);

}

void SceneWidget::drawAR()
{
	EasyAR::Frame frame = ar_mananger_.augmenter_.newFrame();
	EasyAR::AugmentedTarget::Status status = frame.targets()[0].status();


	if (status == EasyAR::AugmentedTarget::kTargetStatusTracked)
	{


		EasyAR::Matrix44F projectionMatrix = EasyAR::getProjectionGL(ar_mananger_.camera_.cameraCalibration(), 0.2f, 500.f);
		EasyAR::Matrix44F cameraview = EasyAR::getPoseGL(frame.targets()[0].pose());
		EasyAR::ImageTarget target = frame.targets()[0].target().cast_dynamic<EasyAR::ImageTarget>();




		glPushMatrix();
		glViewport(0, 0, this->width(), this->height());
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glLoadMatrixf(&projectionMatrix.data[0]);
		
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glLoadMatrixf(&cameraview.data[0]);

		glClear(GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH);
		//glEnable(GL_CULL_FACE);

		glBegin(GL_TRIANGLES);

		glColor3f(1.0f, 0.0f, 0.0f);          // Red
		glVertex3f(0.0f, .0f, 1.0f);          // Top Of Triangle (Front)
		glColor3f(0.0f, 1.0f, 0.0f);          // Green
		glVertex3f(-1.0f, -1.0f, -1.0f);          // Left Of Triangle (Front)
		glColor3f(0.0f, 0.0f, 1.0f);          // Blue
		glVertex3f(1.0f, -1.0f, -1.0f);          // Right Of Triangle (Front)

		glColor3f(1.0f, 0.0f, 0.0f);          // Red
		glVertex3f(0.0f, 0.0f, 1.0f);          // Top Of Triangle (Right)
		glColor3f(0.0f, 0.0f, 1.0f);          // Blue
		glVertex3f(-1.0f, 1.0f, -1.0f);          // Left Of Triangle (Right)
		glColor3f(0.0f, 1.0f, 0.0f);          // Green
		glVertex3f(1.0f, 1.0f, -1.0f);         // Right Of Triangle (Right)

		glColor3f(1.0f, 0.0f, 0.0f);          // Red
		glVertex3f(0.0f, 0.0f, 1.0f);          // Top Of Triangle (Back)
		glColor3f(0.0f, 1.0f, 0.0f);          // Green
		glVertex3f(1.0f, -1.0f, -1.0f);         // Left Of Triangle (Back)
		glColor3f(0.0f, 0.0f, 1.0f);          // Blue
		glVertex3f(1.0f, 1.0f, -1.0f);         // Right Of Triangle (Back)


		glColor3f(1.0f, 0.0f, 0.0f);          // Red
		glVertex3f(0.0f, 0.0f, 1.0f);          // Top Of Triangle (Left)
		glColor3f(0.0f, 0.0f, 1.0f);          // Blue
		glVertex3f(-1.0f, -1.0f, -1.0f);          // Left Of Triangle (Left)
		glColor3f(0.0f, 1.0f, 0.0f);          // Green
		glVertex3f(-1.0f, 1.0f, -1.0f);          // Right Of Triangle (Left)



		//glColor3f(1.0f, 0.0f, 0.0f);          // Red
		//glVertex3f(0.0f, 1.0f, 0.0f);          // Top Of Triangle (Front)
		//glColor3f(0.0f, 1.0f, 0.0f);          // Green
		//glVertex3f(-1.0f, -1.0f, 1.0f);          // Left Of Triangle (Front)
		//glColor3f(0.0f, 0.0f, 1.0f);          // Blue
		//glVertex3f(1.0f, -1.0f, 1.0f);          // Right Of Triangle (Front)

		//glColor3f(1.0f, 0.0f, 0.0f);          // Red
		//glVertex3f(0.0f, 1.0f, 0.0f);          // Top Of Triangle (Right)
		//glColor3f(0.0f, 0.0f, 1.0f);          // Blue
		//glVertex3f(1.0f, -1.0f, 1.0f);          // Left Of Triangle (Right)
		//glColor3f(0.0f, 1.0f, 0.0f);          // Green
		//glVertex3f(1.0f, -1.0f, -1.0f);         // Right Of Triangle (Right)

		//glColor3f(1.0f, 0.0f, 0.0f);          // Red
		//glVertex3f(0.0f, 1.0f, 0.0f);          // Top Of Triangle (Back)
		//glColor3f(0.0f, 1.0f, 0.0f);          // Green
		//glVertex3f(1.0f, -1.0f, -1.0f);         // Left Of Triangle (Back)
		//glColor3f(0.0f, 0.0f, 1.0f);          // Blue
		//glVertex3f(-1.0f, -1.0f, -1.0f);         // Right Of Triangle (Back)


		//glColor3f(1.0f, 0.0f, 0.0f);          // Red
		//glVertex3f(0.0f, 1.0f, 0.0f);          // Top Of Triangle (Left)
		//glColor3f(0.0f, 0.0f, 1.0f);          // Blue
		//glVertex3f(-1.0f, -1.0f, -1.0f);          // Left Of Triangle (Left)
		//glColor3f(0.0f, 1.0f, 0.0f);          // Green
		//glVertex3f(-1.0f, -1.0f, 1.0f);          // Right Of Triangle (Left)
		glEnd();                        // Done Drawing The Pyramid



#if 0
		GLdouble x, y, z; // target 2d coords (z is 'unused')
		GLdouble coords0[] = { .0, .0, 10.0 }; // current 3d coords
		GLdouble coords1[] = { 1.0, .0, 10.0 }; // current 3d coords
		GLdouble coords2[] = { 1.0, 1.0, 10.0 }; // current 3d coords
		GLdouble coords3[] = { .0, 1.0, 10.0 }; // current 3d coords

		GLdouble model_view[16];
		glGetDoublev(GL_MODELVIEW_MATRIX, model_view);

		GLdouble projection[16];
		glGetDoublev(GL_PROJECTION_MATRIX, projection);

		GLint viewport[4];
		glGetIntegerv(GL_VIEWPORT, viewport);

		// get window coords based on 3D coordinates
		gluProject(0., 0., 10.,
			model_view, projection, viewport,
			coords0, coords0 + 1 , coords0 + 2);
		gluProject(1., 0., 10.,
			model_view, projection, viewport,
			coords1, coords1 + 1 , coords1 + 2);
		gluProject(1., 1., 10.,
			model_view, projection, viewport,
			coords2, coords2 + 1 , coords2 + 2);
		gluProject(0., 1., 10.,
			model_view, projection, viewport,
			coords3, coords3 + 1 , coords3 + 2);

		glFlush();
#endif

		glPopMatrix();

		//glMatrixMode(GL_PROJECTION);
		//glLoadIdentity();
		//glOrtho(0, this->width(), 0, this->height(), -100, 100);
		//glMatrixMode(GL_MODELVIEW);
		//glLoadIdentity();
		//glColor3d(0, 1, 0);
		//glBegin(GL_TRIANGLES);
		//glVertex3d(coords0[0], coords0[1], coords0[2]);
		//glVertex3d(coords1[0], coords1[1], coords1[2]);
		//glVertex3d(coords3[0], coords3[1], coords3[2]);
		//glEnd();
	}
}

void SceneWidget::initPixelBuffer()
{
	if (pbo)
	{
		// unregister this buffer object from CUDA C
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

		// delete old buffer
		glDeleteBuffersARB(1, &pbo);
		glDeleteTextures(1, &tex);
	}

	// create pixel buffer object for display
	glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, this->width()*this->height()*sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

	// create texture for display
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, this->width(), this->height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}

int SceneWidget::iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void SceneWidget::initData()
{
	size_t size = volumeSize.width*volumeSize.height*volumeSize.depth*sizeof(VolumeType);
	void *h_volume = loadRawFile("data/Bucky.raw", size);

	initCuda(h_volume, volumeSize);
	free(h_volume);

	// calculate new grid size
	gridSize = dim3(iDivUp(this->width(), blockSize.x), iDivUp(this->height(), blockSize.y));

	density = 0.05f;
	brightness = 1.0f;
	transferOffset = 0.0f;
	transferScale = 1.0f;
	linearFiltering = true;
}

// Load raw data from disk
void* SceneWidget::loadRawFile(char *filename, size_t size)
{
	FILE *fp = fopen(filename, "rb");

	if (!fp)
	{
		fprintf(stderr, "Error opening file '%s'\n", filename);
		return 0;
	}

	void *data = malloc(size);
	size_t read = fread(data, 1, size, fp);
	fclose(fp);

#if defined(_MSC_VER_)
	printf("Read '%s', %Iu bytes\n", filename, read);
#else
	printf("Read '%s', %zu bytes\n", filename, read);
#endif

	return data;
}


// render image using CUDA
void SceneWidget::render()
{
	copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);

	// map PBO to get CUDA device pointer
	uint *d_output;
	// map PBO to get CUDA device pointer
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
		cuda_pbo_resource));
	//printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

	// clear image
	checkCudaErrors(cudaMemset(d_output, 0, this->width()*this->height() * 4));

	// call CUDA kernel, writing results to PBO
	render_kernel(gridSize, blockSize, d_output, this->width(), this->height(), density, brightness, transferOffset, transferScale);

	getLastCudaError("kernel failed");

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}