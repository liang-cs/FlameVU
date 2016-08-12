#ifndef __AR_MANAGER_H__
#define __AR_MANAGER_H__

#include "easyar/camera.hpp"
#include "easyar/augmenter.hpp"
#include "easyar/imagetarget.hpp"
#include "easyar/frame.hpp"
#include "easyar/utility.hpp"
#include "easyar/imagetracker.hpp"
#include <string>


static std::string kKey = "da77381e4f6ea838f5a10525d29554b8MWZRWH77SCx8psczEgSWxVwJ5l4eBeCPUqYEtRtC4JYK646atL4VhIhIvj9oJFQydEwsYvIBMJLw1Nkihegr4FSxkElB0VNde1BWgwmYrx8qxAg5PewJG55zHDT9wak4cwaV8a6iPZpcKovkiPJM9L5jniozU0pl6NihjLcU";

class ARManager
{
public:
	ARManager();
	~ARManager();


	void setup();

	bool initCamera();
	void loadFromImage(const std::string& path);
	bool loadFromJsonFile(const std::string& path, const std::string& targetname);
	bool start();
	bool stop();
	bool clear();

	EasyAR::Vec2I imageSize() const;
	//void initGL();
	//void resizeGL(int width, int height);
	//void render();
	void setPortrait(bool portrait);

	EasyAR::CameraDevice camera_;
	EasyAR::ImageTracker tracker_;
	EasyAR::Augmenter augmenter_;
	bool portrait_;

};


#endif // !__AR_MANAGER_H__
