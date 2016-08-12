#ifndef FLAMEVU_H
#define FLAMEVU_H

#include <QtWidgets/QMainWindow>
#include "ui_flamevu.h"


QT_FORWARD_DECLARE_CLASS(QOpenGLWidget)
QT_FORWARD_DECLARE_CLASS(SceneWidget)

class FlameVU : public QMainWindow
{
	Q_OBJECT

public:
	FlameVU(QWidget *parent = 0);
	~FlameVU();

private:
	Ui::FlameVUClass ui;
	SceneWidget* scene_widget_;
};

#endif // FLAMEVU_H
