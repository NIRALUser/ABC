
////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////

// prastawa@cs.unc.edu 3/2004

#ifndef _DiffusionBasedBiasCorrector_h
#define _DiffusionBasedBiasCorrector_h

#include "itkImage.h"
#include "itkObject.h"

#include "vnl/vnl_matrix.h"
#include "vnl/vnl_vector.h"
#include "vnl/algo/vnl_matrix_inverse.h"
#include "vnl/algo/vnl_qr.h"

#include "DynArray.h"

template <class TInputImage, class TProbabilityImage>
class DiffusionBasedBiasCorrector : public itk::Object
{

public:

  /** Standard class typedefs. */
  typedef DiffusionBasedBiasCorrector Self;
  typedef itk::SmartPointer<Self> Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** The dimension of the image. */
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TInputImage::ImageDimension);

  // Image types
  typedef TInputImage InputImageType;
  typedef typename TInputImage::Pointer InputImagePointer;
  typedef typename TInputImage::IndexType InputImageIndexType;
  typedef typename TInputImage::PixelType InputImagePixelType;
  typedef typename TInputImage::RegionType InputImageRegionType;
  typedef typename TInputImage::SizeType InputImageSizeType;
  typedef typename TInputImage::SpacingType InputImageSpacingType;

  typedef TProbabilityImage ProbabilityImageType;
  typedef typename ProbabilityImageType::Pointer ProbabilityImagePointer;
  typedef typename ProbabilityImageType::IndexType ProbabilityImageIndexType;
  typedef typename ProbabilityImageType::PixelType ProbabilityImagePixelType;
  typedef typename ProbabilityImageType::RegionType ProbabilityImageRegionType;
  typedef typename ProbabilityImageType::SizeType ProbabilityImageSizeType;

  typedef itk::Image<float, TInputImage::ImageDimension> InternalImageType;
  typedef typename InternalImageType::Pointer InternalImagePointer;
  typedef typename InternalImageType::IndexType InternalImageIndexType;
  typedef typename InternalImageType::PixelType InternalImagePixelType;
  typedef typename InternalImageType::RegionType InternalImageRegionType;
  typedef typename InternalImageType::SizeType InternalImageSizeType;

  void SetAdditive() { m_DoLog = false; }
  void SetMultiplicative() { m_DoLog = true; }

  void SetProbabilities(DynArray<ProbabilityImagePointer> probs);

  void SetDiffusionTimeStep(double d) { m_DiffusionTimeStep = d; }
  void SetDiffusionIterations(unsigned int n) { m_DiffusionIterations = n; }

  void Correct(InputImagePointer input, InputImagePointer output);

protected:

  DiffusionBasedBiasCorrector();
  ~DiffusionBasedBiasCorrector();

  void CheckInput();
  InternalImagePointer ComputeResidualImage();

private:

  InputImagePointer m_InputData;

  DynArray<ProbabilityImagePointer> m_Probabilities;

  bool m_DoLog;

  double m_DiffusionTimeStep;
  unsigned int m_DiffusionIterations;

};

#ifndef MU_MANUAL_INSTANTIATION
#include "DiffusionBasedBiasCorrector.txx"
#endif

#endif
