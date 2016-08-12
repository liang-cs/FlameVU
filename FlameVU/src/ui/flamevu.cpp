#include "flamevu.h"

#include "scenewidget.h"

FlameVU::FlameVU(QWidget *parent)
	: QMainWindow(parent)
{
	//ui.setupUi(this);
	setFixedSize(640, 480);
	scene_widget_ = new SceneWidget(this, true, qRgb(20, 20, 50));
	setCentralWidget(scene_widget_);

	QMenu *fileMenu = menuBar()->addMenu("&File");
	fileMenu->addAction("E&xit", this, &QWidget::close);
	QMenu *helpMenu = menuBar()->addMenu("&Help");
	helpMenu->addAction("About Qt", qApp, &QApplication::aboutQt);

}

FlameVU::~FlameVU()
{
	std::cout << "8" << std::endl;
	delete scene_widget_;
	std::cout << "9" << std::endl;
}
