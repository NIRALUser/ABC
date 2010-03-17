
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// prastawa@sci.utah.edu 01/2010

#ifndef _IntensityMatcher_h
#define _IntensityMatcher_h

#include "itkObject.h"
#include "itkImage.h"

#include "DynArray.h"

template <class TInputImage, class TProbImage>
class IntensityMatcher: public itk::Object
{

public:

  /** Standard class typedefs. */
  typedef IntensityMatcher Self;
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
  typedef typename TInputImage::OffsetType InputImageOffsetType;
  typedef typename TInputImage::PixelType InputImagePixelType;
  typedef typename TInputImage::RegionType InputImageRegionType;
  typedef typename TInputImage::SizeType InputImageSizeType;
  typedef typename TInputImage::SpacingType InputImageSpacingType;

  typedef TProbImage ProbImageType;
  typedef typename ProbImageType::Pointer ProbImagePointer;
  typedef typename ProbImageType::IndexType ProbImageIndexType;
  typedef typename ProbImageType::OffsetType ProbImageOffsetType;
  typedef typename ProbImageType::PixelType ProbImagePixelType;
  typedef typename ProbImageType::RegionType ProbImageRegionType;
  typedef typename ProbImageType::SizeType ProbImageSizeType;
  typedef typename ProbImageType::SpacingType ProbImageSpacingType;

  void SetSourceImage(InputImagePointer img);
  void SetTargetImage(InputImagePointer img);

  void SetProbabilities(const DynArray<InputImagePointer>& probs);

  InputImagePointer GetOutput() { this->Update(); return m_OutputImage; }

  void Update();

protected:

  IntensityMatcher();
  ~IntensityMatcher();

  double EvaluatePoint(double x);

  InputImagePointer m_SourceImage;
  InputImagePointer m_TargetImage;

  DynArray<ProbImagePointer> m_Probs;

  DynArray<double> m_SourcePoints;
  DynArray<double> m_TargetPoints;

  double m_MinSource;
  double m_MaxSource;
  double m_MinTarget;
  double m_MaxTarget;

  bool m_Modified;

  InputImagePointer m_OutputImage;

};

#ifndef MU_MANUAL_INSTANTIATION
#include "IntensityMatcher.txx"
#endif

#endif
