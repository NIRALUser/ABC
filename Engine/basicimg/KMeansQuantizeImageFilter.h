
////////////////////////////////////////////////////////////////////////////////
//
// Quantization using kmeans, helps with histogram binning
//
////////////////////////////////////////////////////////////////////////////////

// prastawa@cs.unc.edu 3/2007

#ifndef _KMeansQuantizeImageFilter_h
#define _KMeansQuantizeImageFilter_h

#include "itkImageToImageFilter.h"

template <class TInputImage, class TOutputImage>
class KMeansQuantizeImageFilter:
public itk::ImageToImageFilter<TInputImage, TOutputImage>
{

public:

  /** Standard class typedefs. */
  typedef KMeansQuantizeImageFilter Self;
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

  typedef TOutputImage OutputImageType;
  typedef typename TOutputImage::Pointer OutputImagePointer;
  typedef typename TOutputImage::IndexType OutputImageIndexType;
  typedef typename TOutputImage::OffsetType OutputImageOffsetType;
  typedef typename TOutputImage::PixelType OutputImagePixelType;
  typedef typename TOutputImage::RegionType OutputImageRegionType;
  typedef typename TOutputImage::SizeType OutputImageSizeType;
  typedef typename TOutputImage::SpacingType OutputImageSpacingType;

  itkGetConstMacro(NumberOfBins, unsigned int);
  itkSetMacro(NumberOfBins, unsigned int);

  itkGetConstMacro(MaximumSamples, unsigned int);
  itkSetMacro(MaximumSamples, unsigned int);

  void TrimAboveOn() { m_TrimAbove = true; }
  void TrimAboveOff() { m_TrimAbove = false; }

  void TrimBelowOn() { m_TrimBelow = true; }
  void TrimBelowOff() { m_TrimBelow = false; }

  itkGetConstMacro(TrimFraction, double);
  itkSetMacro(TrimFraction, double);

  // Default value for pixels that go beyond the trimmed range
  itkGetConstMacro(TrimAboveValue, OutputImagePixelType);
  itkSetMacro(TrimAboveValue, OutputImagePixelType);
  itkGetConstMacro(TrimBelowValue, OutputImagePixelType);
  itkSetMacro(TrimBelowValue, OutputImagePixelType);

protected:

  KMeansQuantizeImageFilter();
  ~KMeansQuantizeImageFilter() { }

  void GenerateData();

private:

  unsigned int m_NumberOfBins;

  unsigned int m_MaximumSamples;

  bool m_TrimAbove;
  bool m_TrimBelow;

  double m_TrimFraction;

  OutputImagePixelType m_TrimAboveValue;
  OutputImagePixelType m_TrimBelowValue;

};

#ifndef MU_MANUAL_INSTANTIATION
#include "KMeansQuantizeImageFilter.txx"
#endif

#endif
