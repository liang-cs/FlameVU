#include "flamevu.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	FlameVU w;
	w.show();
	return a.exec();
}
