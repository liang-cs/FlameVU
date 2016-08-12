#include "armanager.h"

#include <algorithm>
#include <iostream>


class HelloCallBack : public EasyAR::TargetLoadCallBack
{
public:
	virtual ~HelloCallBack() {};
	virtual void operator() (const EasyAR::Target target, const bool status)
	{
		//LOGI("load target: %s (%d) %s\n", target.name(), target.id(), status ? "success" : "fail");
		std::cout << "load target: " << target.name() << " " << target.id() << " " << (status ? "success" : "fail") << std::endl;
		delete this;
	}
};



ARManager::ARManager()
{
	portrait_ = false;
}

ARManager::~ARManager()
{
	clear();
}

void ARManager::setup()
{
	EasyAR::initialize(kKey.c_str());
}

bool ARManager::initCamera()
{
	bool status = true;
	status &= camera_.open();
	camera_.setSize(EasyAR::Vec2I(640, 480));
	status &= tracker_.attachCamera(camera_);
	status &= augmenter_.attachCamera(camera_);
	return status;
}

void ARManager::loadFromImage(const std::string& path)
{
	EasyAR::ImageTarget target;
	std::string jstr = "{\n"
		"  \"images\" :\n"
		"  [\n"
		"    {\n"
		"      \"image\" : \"" + path + "\",\n"
		"      \"name\" : \"" + path.substr(0, path.find_first_of(".")) + "\"\n"
		"    }\n"
		"  ]\n"
		"}";
	target.load(jstr.c_str(), EasyAR::kStorageAssets | EasyAR::kStorageJson);
	tracker_.loadTarget(target, new HelloCallBack());
}

bool ARManager::loadFromJsonFile(const std::string& path, const std::string& targetname)
{
	EasyAR::ImageTarget target;
	bool status = target.load(path.c_str(), EasyAR::kStorageApp, targetname.c_str());
	tracker_.loadTarget(target, new HelloCallBack());
	return status;
}

bool ARManager::start()
{
	bool status = true;
	status &= camera_.start();
	camera_.setFocusMode(EasyAR::CameraDevice::kFocusModeContinousauto);
	status &= tracker_.start();
	return status;
}

bool ARManager::stop()
{
	bool status = true;
	status &= tracker_.stop();
	status &= camera_.stop();
	return status;
}

bool ARManager::clear()
{
	bool status = true;
	status &= augmenter_.detachCamera(camera_);
	status &= stop();
	status &= camera_.close();
	camera_.clear();
	tracker_.clear();
	augmenter_.clear();
	return status;
}
EasyAR::Vec2I ARManager::imageSize() const
{
	return camera_.size();
}
//void ofxEasyAR::resizeGL(int width, int height)
//{
//	EasyAR::Vec2I size = EasyAR::Vec2I(1, 1);
//	if (camera_.isOpened())
//		size = camera_.size();
//	if (size[0] == 0 || size[1] == 0)
//		return;
//	if (portrait_)
//		std::swap(size[0], size[1]);
//	float scaleRatio = std::max((float)width / (float)size[0], (float)height / (float)size[1]);
//	EasyAR::Vec2I viewport_size = EasyAR::Vec2I((int)(size[0] * scaleRatio), (int)(size[1] * scaleRatio));
//	augmenter_.setViewPort(EasyAR::Vec4I(0, height - viewport_size[1], viewport_size[0], viewport_size[1]));
//}
//
//void ofxEasyAR::initGL()
//{
//
//}
//
//void ofxEasyAR::render()
//{
//
//}
//
void ARManager::setPortrait(bool portrait)
{
	portrait_ = portrait;
}
