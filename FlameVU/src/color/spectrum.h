
/*
    pbrt source code Copyright(c) 1998-2010 Matt Pharr and Greg Humphreys.

    This file is part of pbrt.

    pbrt is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.  Note that the text contents of
    the book "Physically Based Rendering" are *not* licensed under the
    GNU GPL.

    pbrt is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

 */

#ifndef PBRT_CORE_SPECTRUM_H
#define PBRT_CORE_SPECTRUM_H









#include <vector>
#include <algorithm>
#include <cmath>
#include <float.h>

using namespace std;



#define isnan _isnan




// Global Inline Functions
inline float Lerp(float t, float v1, float v2) {
	return (1.f - t) * v1 + t * v2;
}

// Spectrum Utility Declarations
static const int sampledLambdaStart = 400;
static const int sampledLambdaEnd = 700;
static const int nSpectralSamples = 30;
extern bool SpectrumSamplesSorted(const float *lambda, const float *vals, int n);
extern void SortSpectrumSamples(float *lambda, float *vals, int n);

inline void XYZToRGB(const float xyz[3], float rgb[3]) {
	rgb[0] =  3.240479f*xyz[0] - 1.537150f*xyz[1] - 0.498535f*xyz[2];
	rgb[1] = -0.969256f*xyz[0] + 1.875991f*xyz[1] + 0.041556f*xyz[2];
	rgb[2] =  0.055648f*xyz[0] - 0.204043f*xyz[1] + 1.057311f*xyz[2];


	rgb[0] = (rgb[0]>0) ? rgb[0] : 0.f;
	rgb[1] = (rgb[1]>0) ? rgb[1] : 0.f;
	rgb[2] = (rgb[2]>0) ? rgb[2] : 0.f;
}

// Spectral Data Declarations
static const int nCIESamples = 471;
extern const float CIE_X[nCIESamples];
extern const float CIE_Y[nCIESamples];
extern const float CIE_Z[nCIESamples];
extern const float CIE_lambda[nCIESamples];	


enum SpectrumType { SPECTRUM_REFLECTANCE, SPECTRUM_ILLUMINANT };
extern void Blackbody(const float *wl, int n, float temp, float *vals);
extern float InterpolateSpectrumSamples(const float *lambda, const float *vals,
										int n, float l);


// Spectrum Declarations
template <int nSamples> class CoefficientSpectrum {
public:
	// CoefficientSpectrum Public Methods
	CoefficientSpectrum(float v = 0.f) {
		for (int i = 0; i < nSamples; ++i)
			c[i] = v;
		if(HasNaNs())
		{
			printf("spectrum has nans");
		}
	}


	CoefficientSpectrum &operator*=(float a) {
		for (int i = 0; i < nSamples; ++i)
			c[i] *= a;
		if (HasNaNs())
		{
			printf("spectrum has nans");
		}
		return *this;
	}


	bool HasNaNs() const {
		for (int i = 0; i < nSamples; ++i)
			if (isnan(c[i])) return true;
		return false;
	}

protected:
	// CoefficientSpectrum Protected Data
	float c[nSamples];
};


class RGBSpectrum  : public CoefficientSpectrum<3> {
	    
public:
	using CoefficientSpectrum<3>::c;

	static RGBSpectrum FromXYZ(const float xyz[3],
		SpectrumType type = SPECTRUM_REFLECTANCE) {
			RGBSpectrum r;
			XYZToRGB(xyz, r.c);
			return r;
	}
	static RGBSpectrum FromSampled(const float *lambda, const float *v,
		int n) {
			// Sort samples if unordered, use sorted for returned spectrum
			if (!SpectrumSamplesSorted(lambda, v, n)) {
				vector<float> slambda(&lambda[0], &lambda[n]);
				vector<float> sv(&v[0], &v[n]);
				SortSpectrumSamples(&slambda[0], &sv[0], n);
				return FromSampled(&slambda[0], &sv[0], n);
			}
			float xyz[3] = { 0, 0, 0 };
			float yint = 0.f;
			for (int i = 0; i < nCIESamples; ++i) {
				yint += CIE_Y[i];
				float val = InterpolateSpectrumSamples(lambda, v, n,
					CIE_lambda[i]);
				xyz[0] += val * CIE_X[i];
				xyz[1] += val * CIE_Y[i];
				xyz[2] += val * CIE_Z[i];
			}
			xyz[0] /= yint;
			xyz[1] /= yint;
			xyz[2] /= yint;

			//float sigma = 1.f;
			//xyz[0] = xyz[0] / (sigma + xyz[0]);
			//xyz[1] = xyz[1] / (sigma + xyz[1]);
			//xyz[2] = xyz[2] / (sigma + xyz[2]);
			//printf("x = %f, y = %f, z = %f\n", xyz[0], xyz[1], xyz[2]);

			////  0.4002400  0.7076000 -0.0808100
			////	-0.2263000  1.1653200  0.0457000   M[]
			////	0.0000000  0.0000000  0.9182200
// 			float L, M, S;
// 			L = 0.4002400 * xyz[0] + 0.7076000 * xyz[1] -0.0808100 * xyz[2];
// 			M = -0.2263000 * xyz[0] + 1.1653200 * xyz[1] + 0.0457000 * xyz[2];
// 			S = 0.9182200 * xyz[2];
// 
// 
// 
// 
// 			printf("L = %f, M = %f, S = %f\n", L, M, S);
			//getchar();

			return FromXYZ(xyz);
	}

	static RGBSpectrum FromSampledToneMapping(const float *lambda, const float *v,
		int n, RGBSpectrum whiteLMS) {
			// Sort samples if unordered, use sorted for returned spectrum
			if (!SpectrumSamplesSorted(lambda, v, n)) {
				vector<float> slambda(&lambda[0], &lambda[n]);
				vector<float> sv(&v[0], &v[n]);
				SortSpectrumSamples(&slambda[0], &sv[0], n);
				return FromSampled(&slambda[0], &sv[0], n);
			}
			float xyz[3] = { 0, 0, 0 };
			float yint = 0.f;
			for (int i = 0; i < nCIESamples; ++i) {
				yint += CIE_Y[i];
				float val = InterpolateSpectrumSamples(lambda, v, n,
					CIE_lambda[i]);
				xyz[0] += val * CIE_X[i];
				xyz[1] += val * CIE_Y[i];
				xyz[2] += val * CIE_Z[i];
			}
			xyz[0] /= yint;
			xyz[1] /= yint;
			xyz[2] /= yint;

			//float sigma = 1.f;
			//xyz[0] = xyz[0] / (sigma + xyz[0]);
			//xyz[1] = xyz[1] / (sigma + xyz[1]);
			//xyz[2] = xyz[2] / (sigma + xyz[2]);
			//printf("x = %f, y = %f, z = %f\n", xyz[0], xyz[1], xyz[2]);

			////  0.4002400  0.7076000 -0.0808100
			////	-0.2263000  1.1653200  0.0457000   M[]
			////	0.0000000  0.0000000  0.9182200
			float L, M, S;
			L = 0.4002400 * xyz[0] + 0.7076000 * xyz[1] -0.0808100 * xyz[2];
			M = -0.2263000 * xyz[0] + 1.1653200 * xyz[1] + 0.0457000 * xyz[2];
			S = 0.9182200 * xyz[2];

			float xyza[3];
#if 1
			//1.8599364 -1.1293816  0.2198974
			//	0.3611914  0.6388125 -0.0000064   M-1[]
			//	0.0000000  0.0000000  1.0890636
		
			xyza[0] = 1.8599364 / whiteLMS.c[0] * L - 1.1293816 / whiteLMS.c[1] * M + 0.2198974 / whiteLMS.c[2] * S;
			xyza[1] = 0.3611914 / whiteLMS.c[0] * L + 0.6388125 / whiteLMS.c[1] * M - 0.0000064 / whiteLMS.c[2] * S;
			xyza[2] = 1.0890636 / whiteLMS.c[2] * S;
#else
			float sigma = 0.5f;
			//L = L / (sigma + L);
			//M = M / (sigma + M);
			//S = S / (sigma + S);
			xyza[0] = 1.8599364 * L - 1.1293816 * M + 0.2198974 * S;
			xyza[1] = 0.3611914 * L + 0.6388125 * M - 0.0000064 * S;
			xyza[2] = 1.0890636 * S;
			//printf("L = %f, M = %f, S = %f\n", L, M, S);
			//getchar();
#endif
			return FromXYZ(xyza);
	}

	static RGBSpectrum GetWhitePointLMS(const float *lambda, const float *v,
		int n) {
			// Sort samples if unordered, use sorted for returned spectrum
			if (!SpectrumSamplesSorted(lambda, v, n)) {
				vector<float> slambda(&lambda[0], &lambda[n]);
				vector<float> sv(&v[0], &v[n]);
				SortSpectrumSamples(&slambda[0], &sv[0], n);
				return FromSampled(&slambda[0], &sv[0], n);
			}
			float xyz[3] = { 0, 0, 0 };
			float yint = 0.f;
			for (int i = 0; i < nCIESamples; ++i) {
				yint += CIE_Y[i];
				float val = InterpolateSpectrumSamples(lambda, v, n,
					CIE_lambda[i]);
				xyz[0] += val * CIE_X[i];
				xyz[1] += val * CIE_Y[i];
				xyz[2] += val * CIE_Z[i];
			}
			xyz[0] /= yint;
			xyz[1] /= yint;
			xyz[2] /= yint;

			//float sigma = 0.1f;
			//xyz[0] = xyz[0] / (sigma + xyz[0]);
			//xyz[1] = xyz[1] / (sigma + xyz[1]);
			//xyz[2] = xyz[2] / (sigma + xyz[2]);
			//printf("x = %f, y = %f, z = %f\n", xyz[0], xyz[1], xyz[2]);

			//  0.4002400  0.7076000 -0.0808100
			//	-0.2263000  1.1653200  0.0457000   M[]
			//	0.0000000  0.0000000  0.9182200
			RGBSpectrum wp;
			
			wp.c[0] = 0.4002400 * xyz[0] + 0.7076000 * xyz[1] -0.0808100 * xyz[2];
			wp.c[1] = -0.2263000 * xyz[0] + 1.1653200 * xyz[1] + 0.0457000 * xyz[2];
			wp.c[2] = 0.9182200 * xyz[2];

			//printf("L = %f, M = %f, S = %f\n", L, M, S);
			//getchar();

			wp *= 2;

			return wp;
	}



};




#endif // PBRT_CORE_SPECTRUM_H