
////////////////////////////////////////////////////////////////////////////////
//
// Registration of a pair of 3D images using different metrics and 
// transformations
//
////////////////////////////////////////////////////////////////////////////////

// prastawa@cs.unc.edu 5/2005

#ifndef _PairRegistrationMethod_h
#define _PairRegistrationMethod_h

#include "itkAffineTransform.h"
#include "itkBSplineDeformableTransform.h"
#include "itkImage.h"
#include "itkVector.h"

#include "ChainedAffineTransform3D.h"

#include <fstream>

template <class TPixel>
class PairRegistrationMethod
{

public:

  static const unsigned int SplineOrder = 3;

  // typedefs
  typedef ChainedAffineTransform3D AffineTransformType;
  // 3-D spline, with dimension 3 and order SplineOrder
  //typedef itk::BSplineDeformableTransform<double, 3, SplineOrder>
// Note: VC does not support const template arg???
  typedef itk::BSplineDeformableTransform<double, 3, 3>
    BSplineTransformType;

  typedef itk::Image<TPixel, 3> ImageType;

  typedef itk::Vector<float, 3> VectorType;
  typedef itk::Image<VectorType, 3> DeformationFieldType;

  typedef itk::Image<unsigned char, 3> MaskImageType;

  typedef enum{QuantizeNone, QuantizeFixed, QuantizeMoving, QuantizeBoth}
    QuantizationOption;

  //
  // Registration functions
  //

  static AffineTransformType::Pointer
    RegisterAffine(ImageType* fixedImg, ImageType* movingImg,
      QuantizationOption qopt=QuantizeNone);

  // Affine with unit scaling and zero skew
  static AffineTransformType::Pointer
    RegisterRigid(ImageType* fixedImg, ImageType* movingImg,
      QuantizationOption qopt=QuantizeNone);

  static BSplineTransformType::Pointer
    RegisterBSpline(ImageType* fixedImg, ImageType* movingImg,
      unsigned int nx, unsigned int ny, unsigned int nz,
      QuantizationOption qopt=QuantizeNone,
      MaskImageType* fixedMask=0);

  static DeformationFieldType::Pointer
    RegisterDemons(ImageType* fixedImg, ImageType* movingImg,
      unsigned int numIters=100);

  // Read / write functions
  static AffineTransformType::Pointer
    ReadAffineTransform(const char* fn);
  static void
    WriteAffineTransform(const char* fn, const AffineTransformType* affine);

  static BSplineTransformType::Pointer
    ReadBSplineTransform(const char* fn);
  static void
    WriteBSplineTransform(const char* fn, const BSplineTransformType* bspline);

  static DeformationFieldType::Pointer
    ReadDeformationField(const char* fn);
  static void
    WriteDeformationField(const char* fn, const DeformationFieldType* def);

private:

  // Find next uncommented line (doesn't begin with #)
  static void ReadNextLine(char* s, std::ifstream& infile);


};

#ifndef MU_MANUAL_INSTANTIATION
#include "PairRegistrationMethod.txx"
#endif

#endif
