#include "scenewidget.h"



#include "functions.h"
#include "macros.h"
#include "flamevu.h"
#include "parameters.h"

#include "planckmapping.h"

SceneWidget::SceneWidget(FlameVU *mw, bool button, const QColor &background)
	: m_mainWindow(mw)
{
	//setMinimumSize(300, 250);
	//setFixedSize(640, 480);

	setFixedSize(640, 480);

	g_volume_size = g_volume_extent.width*g_volume_extent.height*g_volume_extent.depth;

	g_block_size.x = 16;
	g_block_size.y = 16;
	g_grid_size = dim3(iDivUp(this->width(), g_block_size.x), iDivUp(this->height(), g_block_size.y));
	ar_mananger_ = 0;
	debug_info_.open("debug.txt");

	checkCudaErrors(cudaMalloc(&d_background_tex_, sizeof(uint) * this->width() * this->height()));
	checkCudaErrors(cudaMalloc(&d_capture_image_data_, 3 * sizeof(uchar) * this->width() * this->height()));
}
SceneWidget::~SceneWidget()
{
	debug_info_.close();

	// And now release all OpenGL resources.
	makeCurrent();
	//delete program;
	doneCurrent();
	for (int i = 0; i < g_num_frames; ++i)
	{
		SAFE_DELETE_ARRAY(g_volume_data_list[i]);
	}
	g_volume_data_list.clear();

	SAFE_DELETE(ar_mananger_);
	QThread::msleep(1000);
	SAFE_DELETE_DEVICE_ARRAY(d_background_tex_);
	SAFE_DELETE_DEVICE_ARRAY(d_capture_image_data_);
}

void SceneWidget::initAR()
{
	ar_mananger_ = new ARManager();

	ar_mananger_->setup();
	ar_mananger_->initCamera();
	bool status = ar_mananger_->loadFromJsonFile("data/target.json", "flame");
	if (!status)
	{
		std::cout << "failed while loading json file" << std::endl;
	}
	ar_mananger_->start();
	last_time_stamp_ = 0.0;

	QThread::msleep(1000);
}

void SceneWidget::initializeGL()
{
	initAR();

	initializeOpenGLFunctions();
	glewInit();

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	glGenTextures(1, &tex_id_);
	glBindTexture(GL_TEXTURE_2D, tex_id_);        
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

#if 0 // enable shader
#define PROGRAM_VERTEX_ATTRIBUTE 0
#define PROGRAM_TEXCOORD_ATTRIBUTE 1

	QOpenGLShader *vshader = new QOpenGLShader(QOpenGLShader::Vertex, this);
	const char *vsrc =
		"void main()\n"
		"{\n"
		"	gl_TexCoord[0] = gl_MultiTexCoord0;\n"
		"	gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
		"}\n";
	vshader->compileSourceCode(vsrc);

	QOpenGLShader *fshader = new QOpenGLShader(QOpenGLShader::Fragment, this);
	const char *fsrc =
		"uniform sampler2D color_texture;\n"
		"void main(void)\n"
		"{\n"
		"    gl_FragColor = texture2D(color_texture, gl_TexCoord[0].st);\n"
		"}\n";

	fshader->compileSourceCode(fsrc);

	program = new QOpenGLShaderProgram;
	program->addShader(vshader);
	program->addShader(fshader);
	program->link();

	program->bind();
#endif // enable shader

	initData(g_volume_filename);
	initTransferTex();
}

void SceneWidget::paintGL()
{

	bindVolumeTexture(g_volume_data_list[g_current_frame], g_volume_extent);
	g_current_frame = (++g_current_frame) % g_num_frames;



	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


#if 0 // test flame rendering
	glViewport(0, 0, this->width(), this->height());
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60., 1. * this->width() / this->height(), 0.2, 2000);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0, 3, 0, 0, 0, 0, 0, 0, 1);
	computeFrustumPlanes();
	renderColorTemperature();
	drawPbo();
#endif  // test flame rendering

	updateAR();
	drawAR();
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



	update();
}




void SceneWidget::resizeGL(int width, int height)
{
	initPixelBuffer();

	// calculate new grid size
	g_grid_size = dim3(iDivUp(width, g_block_size.x), iDivUp(height, g_block_size.y));

	glViewport(0, 0, width, height);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, this->width(), 0, this->height(), -100, 100);
}

void SceneWidget::updateAR()
{
	EasyAR::Frame frame = ar_mananger_->augmenter_.newFrame();
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
	


	if (last_time_stamp_ != frame.timeStamp())
	{
		checkCudaErrors(cudaMemcpy(d_capture_image_data_, image.data(), 3 * this->width() * this->height(),
			cudaMemcpyHostToDevice));

		align_channels(g_grid_size, g_block_size, d_capture_image_data_, d_background_tex_,
			this->width(), this->height());

		last_time_stamp_ = frame.timeStamp();
	}

#if 0 
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
	glFlush();
#endif
}

void SceneWidget::drawAR()
{
	EasyAR::Frame frame = ar_mananger_->augmenter_.newFrame();
	EasyAR::AugmentedTarget::Status status = frame.targets()[0].status();


	glEnable(GL_DEPTH);
	if (status == EasyAR::AugmentedTarget::kTargetStatusTracked)
	{

		EasyAR::Matrix44F projectionMatrix = EasyAR::getProjectionGL(ar_mananger_->camera_.cameraCalibration(), 0.2f, 500.f);
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

		//glClear(GL_DEPTH_BUFFER_BIT);
		//glEnable(GL_CULL_FACE);
		glTranslatef(0.f, 0.f, 1.f);
#if 1 // draw flames
		computeFrustumPlanes();
		renderColorTemperature(true);
		drawPbo();
#endif // draw flames

#if 0 // draw s pyramid
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

		glEnd();                        // Done Drawing The Pyramid
#endif // draw a pyramid

		glPopMatrix();
	}
	else
	{
		glPushMatrix();
		renderColorTemperature(false);
		drawPbo();
		glPopMatrix();
	}
	
}
void SceneWidget::drawPbo()
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

	// display results
	glClear(GL_COLOR_BUFFER_BIT);

	// draw image from PBO
	glDisable(GL_DEPTH_TEST);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
#if 0
	// draw using glDrawPixels (slower)
	glRasterPos2i(0, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, g_pbo);
	glDrawPixels(gWidth, gHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
#else
	// draw using texture

	// copy from pbo to texture
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, g_pbo);
	glBindTexture(GL_TEXTURE_2D, g_tex);
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
#endif
}

void SceneWidget::renderColorTemperature(bool tracked)
{
	// map PBO to get CUDA device pointer
	uint *d_output;
	// map PBO to get CUDA device pointer
	checkCudaErrors(cudaGraphicsMapResources(1, &g_cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_output, &num_bytes,
		g_cuda_pbo_resource));

	// clear image
	//checkCudaErrors(cudaMemset(d_output, 0, this->width()*this->height() * 4));
	checkCudaErrors(cudaMemcpy(d_output, d_background_tex_,
		sizeof(uint) * this->width() * this->height(), cudaMemcpyDeviceToDevice));

	if (tracked)
	{
		// call CUDA kernel, writing results to PBO
		render_color_temperature_kernel(g_grid_size, g_block_size,
			d_output, this->width(), this->height(),
			g_density, g_brightness, g_sampling_max_step, g_sampling_step,
			g_box_min, g_box_max, g_volume_extent, g_voxel_length);
	}

	checkCudaErrors(cudaGraphicsUnmapResources(1, &g_cuda_pbo_resource, 0));
}


void SceneWidget::computeFrustumPlanes()
{
	GLdouble modelView[16];
	GLdouble projection[16];
	GLint viewport[4];

	glGetDoublev(GL_MODELVIEW_MATRIX, modelView);
	glGetDoublev(GL_PROJECTION_MATRIX, projection);
	glGetIntegerv(GL_VIEWPORT, viewport);

	// compute the near and far planes
	GLdouble objx, objy, objz;
	//¼ÆËãnll(near plane left low)
	gluUnProject(viewport[0], viewport[1], 0.00, modelView, projection, viewport, &objx, &objy, &objz);
	g_near_far[0] = (GLfloat)objx;
	g_near_far[1] = (GLfloat)objy;
	g_near_far[2] = (GLfloat)objz;


	//nlh
	gluUnProject(viewport[0], viewport[1] + viewport[3], 0.00, modelView, projection, viewport, &objx, &objy, &objz);
	g_near_far[3] = (GLfloat)objx;
	g_near_far[4] = (GLfloat)objy;
	g_near_far[5] = (GLfloat)objz;


	//nrh
	gluUnProject(viewport[0] + viewport[2], viewport[1] + viewport[3], 0.00, modelView, projection, viewport, &objx, &objy, &objz);
	g_near_far[6] = (GLfloat)objx;
	g_near_far[7] = (GLfloat)objy;
	g_near_far[8] = (GLfloat)objz;

	//nrl
	gluUnProject(viewport[0] + viewport[2], viewport[1], 0.00, modelView, projection, viewport, &objx, &objy, &objz);
	g_near_far[9] = (GLfloat)objx;
	g_near_far[10] = (GLfloat)objy;
	g_near_far[11] = (GLfloat)objz;

	//¼ÆËãfll
	gluUnProject(viewport[0], viewport[1], 1.00, modelView, projection, viewport, &objx, &objy, &objz);
	g_near_far[12] = (GLfloat)objx;
	g_near_far[13] = (GLfloat)objy;
	g_near_far[14] = (GLfloat)objz;

	//flh
	gluUnProject(viewport[0], viewport[1] + viewport[3], 1.00, modelView, projection, viewport, &objx, &objy, &objz);
	g_near_far[15] = (GLfloat)objx;
	g_near_far[16] = (GLfloat)objy;
	g_near_far[17] = (GLfloat)objz;

	//frh
	gluUnProject(viewport[0] + viewport[2], viewport[1] + viewport[3], 1.00, modelView, projection, viewport, &objx, &objy, &objz);
	g_near_far[18] = (GLfloat)objx;
	g_near_far[19] = (GLfloat)objy;
	g_near_far[20] = (GLfloat)objz;

	//frl
	gluUnProject(viewport[0] + viewport[2], viewport[1], 1.00, modelView, projection, viewport, &objx, &objy, &objz);
	g_near_far[21] = (GLfloat)objx;
	g_near_far[22] = (GLfloat)objy;
	g_near_far[23] = (GLfloat)objz;
	copyNearFar(g_near_far, sizeof(float) * 24);
}

void SceneWidget::initPixelBuffer()
{
	
	if (g_pbo)
	{
		// unregister this buffer object from CUDA C
		checkCudaErrors(cudaGraphicsUnregisterResource(g_cuda_pbo_resource));

		// delete old buffer
		glDeleteBuffersARB(1, &g_pbo);
		glDeleteTextures(1, &g_tex);
	}

	// create pixel buffer object for display
	glGenBuffersARB(1, &g_pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, g_pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, this->width()*this->height()*sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&g_cuda_pbo_resource, g_pbo, cudaGraphicsMapFlagsWriteDiscard));

	// create texture for display
	glGenTextures(1, &g_tex);
	glBindTexture(GL_TEXTURE_2D, g_tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, this->width(), this->height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}

int SceneWidget::iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void SceneWidget::initBoundingBox()
{
	int maxDim = max(g_volume_extent.width, max(g_volume_extent.height, g_volume_extent.depth));
	g_box_min = make_float3(-1.f * g_volume_extent.width / maxDim,
		-1.f * g_volume_extent.height / maxDim,
		-1.f * g_volume_extent.depth / maxDim);
	g_box_max = make_float3(1.f * g_volume_extent.width / maxDim,
		1.f * g_volume_extent.height / maxDim,
		1.f * g_volume_extent.depth / maxDim);

	g_voxel_length = (g_box_max.x - g_box_min.x) / g_volume_extent.width;
	g_sampling_step = g_voxel_length;
}
void SceneWidget::initData(string filename)
{

	ifstream inStream;

	char volume_data_filename[200];
	for (int i = 0; i < g_num_frames; ++i)
	{
		VolumeType* volume_data = new VolumeType[g_volume_size];
		sprintf(volume_data_filename, "bin/data/flame/frame_%d.data", i);
		inStream.open(volume_data_filename, ios_base::binary);
		inStream.read((char*)volume_data, g_volume_size * sizeof(VolumeType));
		inStream.close();
		g_volume_data_list.push_back(volume_data);
	}
	initBoundingBox();
}

