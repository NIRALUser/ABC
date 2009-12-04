
////////////////////////////////////////////////////////////////////////////////
//
// Compute boundaries from multiple images / modalities
//
// H.C. Lee and D.R. Cok. Detecting boundaries in a vector field. IEEE Trans
// in Signal Proc. 1991, 39(5), p 1181-1194.
//
////////////////////////////////////////////////////////////////////////////////

// prastawa@cs.unc.edu 06/2005

#ifndef _MultimodalGradientImageFilter_h
#define _MultimodalGradientImageFilter_h

#include "itkObject.h"
#include "itkImage.h"

#include "DynArray.h"

template <class TInputImage>
class MultimodalGradientImageFilter: public itk::Object
{

public:

  /** Standard class typedefs. */
  typedef MultimodalGradientImageFilter Self;
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
  typedef typename TInputImage::PointType InputImagePointType;
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
    InputImagePixelType, itk::GetImageDimension<TInputImage>::ImageDimension>
    OutputImageType;
  typedef typename OutputImageType::Pointer OutputImagePointer;
  typedef typename OutputImageType::IndexType OutputImageIndexType;
  typedef typename OutputImageType::OffsetType OutputImageOffsetType;
  typedef typename OutputImageType::PixelType OutputImagePixelType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;
  typedef typename OutputImageType::SizeType OutputImageSizeType;
  typedef typename OutputImageType::SpacingType OutputImageSpacingType;

  void SetInputImages(const DynArray<InputImagePointer>& images);

  void SetMask(const MaskImagePointer& m) { m_Mask = m; }

  OutputImagePointer ComputeGradientMagnitudeImage();
  DynArray<OutputImagePointer> ComputeGradientImage();

protected:

  MultimodalGradientImageFilter();
  ~MultimodalGradientImageFilter() {}

  DynArray<InputImagePointer> m_InputImages;

  MaskImagePointer m_Mask;

};

#ifndef MU_MANUAL_INSTANTIATION
#include "MultimodalGradientImageFilter.txx"
#endif

#endif
