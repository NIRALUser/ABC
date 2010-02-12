
#ifndef __VectorBlurImageFilter_h
#define __VectorBlurImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImage.h"

template <class TInputImage, class TOutputImage>
class VectorBlurImageFilter :
    public itk::ImageToImageFilter< TInputImage, TOutputImage >
{
public:
  /** Extract dimension from input and output image. */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension);

  /** Convenient typedefs for simplifying declarations. */
  typedef TInputImage InputImageType;
  typedef TOutputImage OutputImageType;

  /** Standard class typedefs. */
  typedef VectorBlurImageFilter Self;
  typedef itk::ImageToImageFilter< InputImageType, OutputImageType> Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(VectorBlurImageFilter, itk::ImageToImageFilter);
  
  /** Image typedef support. */
  typedef typename InputImageType::PixelType InputPixelType;
  typedef typename OutputImageType::PixelType OutputPixelType;

  typedef typename InputImageType::IndexType InputImageIndexType;
  typedef typename OutputImageType::IndexType OutputImageIndexType;
  
  typedef typename InputImageType::RegionType InputImageRegionType;
  typedef typename OutputImageType::RegionType OutputImageRegionType;

  typedef typename InputImageType::SizeType InputSizeType;

  itkSetMacro(KernelWidth, double);
  itkGetConstMacro(KernelWidth, double);

protected:
  VectorBlurImageFilter();
  virtual ~VectorBlurImageFilter() {}
  void PrintSelf(std::ostream& os, itk::Indent indent) const;

  void GenerateData();

private:
  VectorBlurImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  double m_KernelWidth;
};
  
#ifndef MU_MANUAL_INSTANTIATION
#include "VectorBlurImageFilter.txx"
#endif

#endif
