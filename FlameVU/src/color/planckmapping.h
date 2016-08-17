#ifndef PLANK_MAPPING_H_
#define PLANK_MAPPING_H_

#include "spectrum.h"

#include "macros.h"

#define CLAMP_TEMPERATURE_LOW 1000
#define CLAMP_TEMPERATURE_HIGH 10000

typedef RGBSpectrum Spectrum;

class PlanckMapping 
{
public:
	PlanckMapping();
	~PlanckMapping();

	void Init();


	static Spectrum spectrum_[CLAMP_TEMPERATURE_HIGH - CLAMP_TEMPERATURE_LOW + 1];

};


#endif // PLANK_MAPPING_H_
