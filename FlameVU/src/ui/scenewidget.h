#ifndef __SCENE_WIDGET_H__
#define __SCENE_WIDGET_H__


#include "headers.h"


#include "armanager.h"

class FlameVU;

QT_FORWARD_DECLARE_CLASS(QOpenGLTexture)
QT_FORWARD_DECLARE_CLASS(QOpenGLShader)
QT_FORWARD_DECLARE_CLASS(QOpenGLShaderProgram)




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
	void initData(std::string filename);
	int iDivUp(int a, int b);
	void computeFrustumPlanes();
	void renderColorTemperature(bool tracked);
	void initBoundingBox();
	void drawPbo();


private:

	FlameVU* m_mainWindow;


	std::ofstream debug_info_;

	ARManager* ar_mananger_;
	double last_time_stamp_;

	GLuint tex_id_;
	uint* d_background_tex_;
	uchar* d_capture_image_data_;

};






#endif	// __SCENE_WIDGET_H__
