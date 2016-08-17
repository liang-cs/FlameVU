#include "planckmapping.h"

Spectrum PlanckMapping::spectrum_[CLAMP_TEMPERATURE_HIGH - CLAMP_TEMPERATURE_LOW + 1];


PlanckMapping::PlanckMapping()
{
	Init();
}


PlanckMapping::~PlanckMapping()
{

}

void PlanckMapping::Init()
{
	float color[nCIESamples];

	for (int i = CLAMP_TEMPERATURE_LOW; i <= CLAMP_TEMPERATURE_HIGH; ++i) {
		Blackbody(CIE_lambda, nCIESamples, i, color);
		spectrum_[i - CLAMP_TEMPERATURE_LOW] = Spectrum::FromSampled(CIE_lambda, color, nCIESamples);

		//normalize
		float length = sqrt(pow(spectrum_[i - CLAMP_TEMPERATURE_LOW].c[0], 2)
			+ pow(spectrum_[i - CLAMP_TEMPERATURE_LOW].c[1], 2)
			+ pow(spectrum_[i - CLAMP_TEMPERATURE_LOW].c[2], 2));
		spectrum_[i - CLAMP_TEMPERATURE_LOW].c[0] /= length;
		spectrum_[i - CLAMP_TEMPERATURE_LOW].c[1] /= length;
		spectrum_[i - CLAMP_TEMPERATURE_LOW].c[2] /= length;

	}
}
