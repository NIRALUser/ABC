
////////////////////////////////////////////////////////////////////////////////
//
// Generate region partitions from list of input images using watershed
// transform
//
////////////////////////////////////////////////////////////////////////////////

// prastawa@cs.unc.edu 06/2005

#ifndef _BasicPartitionImageFilter_h
#define _BasicPartitionImageFilter_h

#include "itkObject.h"
#include "itkImage.h"

#include "DynArray.h"

template <class TInputImage>
class BasicPartitionImageFilter: public itk::Object
{

public:

  /** Standard class typedefs. */
  typedef BasicPartitionImageFilter Self;
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

  typedef itk::Image<
    unsigned char, itk::GetImageDimension<TInputImage>::ImageDimension>
    MaskImageType;
  typedef typename MaskImageType::Pointer MaskImagePointer;
  typedef typename MaskImageType::IndexType MaskImageIndexType;
  typedef typename MaskImageType::OffsetType MaskImageOffsetType;
  typedef typename MaskImageType::PixelType MaskImagePixelType;
  typedef typename MaskImageType::RegionType MaskImageRegionType;
  typedef typename MaskImageType::SizeType MaskImageSizeType;
  typedef typename MaskImageType::SpacingType MaskImageSpacingType;

  typedef itk::Image<
    unsigned long, itk::GetImageDimension<TInputImage>::ImageDimension>
    OutputImageType;
  typedef typename OutputImageType::Pointer OutputImagePointer;
  typedef typename OutputImageType::IndexType OutputImageIndexType;
  typedef typename OutputImageType::OffsetType OutputImageOffsetType;
  typedef typename OutputImageType::PixelType OutputImagePixelType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;
  typedef typename OutputImageType::SizeType OutputImageSizeType;
  typedef typename OutputImageType::SpacingType OutputImageSpacingType;

  itkGetMacro(BlurVariance, double);
  itkSetMacro(BlurVariance, double);

  itkGetMacro(RescaleMax, double);
  itkSetMacro(RescaleMax, double);

  void SetInputImages(const DynArray<InputImagePointer>& images);

  void SetMask(MaskImagePointer mask);

  OutputImagePointer GetOutput() { this->Update(); return m_Output; }

  void Update();

protected:

  BasicPartitionImageFilter();
  ~BasicPartitionImageFilter() {}

  double m_BlurVariance;

  double m_RescaleMax;

  DynArray<InputImagePointer> m_InputImages;

  MaskImagePointer m_Mask;

  bool m_Modified;

  OutputImagePointer m_Output;

};

#ifndef MU_MANUAL_INSTANTIATION
#include "BasicPartitionImageFilter.txx"
#endif

#endif
