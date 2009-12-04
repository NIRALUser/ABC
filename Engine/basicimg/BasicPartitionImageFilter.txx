
#ifndef _BasicPartitionImageFilter_txx
#define _BasicPartitionImageFilter_txx

#include "itkDiscreteGaussianImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkWatershedImageFilter.h"

#include "itkImageFileWriter.h"

#include "BasicPartitionImageFilter.h"
#include "MultimodalGradientImageFilter.h"

#include <cmath>

template <class TInputImage>
BasicPartitionImageFilter<TInputImage>
::BasicPartitionImageFilter()
{
  m_BlurVariance = 1.0;
  m_RescaleMax = 65535;
  m_Modified = false;
}

template <class TInputImage>
void
BasicPartitionImageFilter<TInputImage>
::SetInputImages(const DynArray<InputImagePointer>& images)
{
  if (images.GetSize() == 0)
    itkExceptionMacro(<< "Need at least one input image");

  OutputImageSizeType size =
    images[0]->GetLargestPossibleRegion().GetSize();

  OutputImageSpacingType spacing =
    images[0]->GetSpacing();

  for (unsigned int i = 1; i < images.GetSize(); i++)
  {
    OutputImageSizeType size_i =
      images[i]->GetLargestPossibleRegion().GetSize();
    OutputImageSpacingType spacing_i =
      images[i]->GetSpacing();
    if (size != size_i || spacing != spacing_i)
      itkExceptionMacro(<< "Input image parameter mismatch");
  }

  m_InputImages = images;

  m_Modified = true;
}

template <class TInputImage>
void
BasicPartitionImageFilter<TInputImage>
::SetMask(MaskImagePointer mask)
{
  m_Mask = mask;

  m_Modified = true;
}

template <class TInputImage>
void
BasicPartitionImageFilter<TInputImage>
::Update()
{
  if (!m_Modified)
    return;

  // Build filtered images
  DynArray<InputImagePointer> filtImages;
  filtImages.Allocate(m_InputImages.GetSize());
  for (unsigned int i = 0; i < m_InputImages.GetSize(); i++)
  {
    typedef itk::RescaleIntensityImageFilter<InputImageType, InputImageType>
      RescalerType;

    typename RescalerType::Pointer rescaler = RescalerType::New();
    rescaler->SetInput(m_InputImages[i]);
    rescaler->SetOutputMinimum(0);
    rescaler->SetOutputMaximum(m_RescaleMax);
    rescaler->Update();

    if (m_BlurVariance <= 0.0)
    {
      filtImages.Append(rescaler->GetOutput());
      continue;
    }

    typedef itk::DiscreteGaussianImageFilter<InputImageType, InputImageType>
      BlurFilterType;

    typename BlurFilterType::Pointer blurFilter = BlurFilterType::New();
    blurFilter->SetInput(rescaler->GetOutput());
    blurFilter->SetVariance(m_BlurVariance);
    blurFilter->Update();

    filtImages.Append(blurFilter->GetOutput());
  }

  // Compute gradient magnitude
  typedef MultimodalGradientImageFilter<InputImageType> GradientFilterType;
  typename GradientFilterType::Pointer gradf = GradientFilterType::New();
  gradf->SetInputImages(filtImages);
  gradf->SetMask(m_Mask);

  InputImagePointer gradImage = gradf->ComputeGradientMagnitudeImage();

/*
//PP DEBUG
typedef itk::Image<unsigned short, 3> UShortImageType;
typedef itk::RescaleIntensityImageFilter<InputImageType, UShortImageType>
  RescalerType;
typename RescalerType::Pointer res = RescalerType::New();
res->SetInput(gradImage);
res->SetOutputMinimum(0);
res->SetOutputMaximum(32000);
res->Update();
typedef itk::ImageFileWriter<UShortImageType> WriterType;
typename WriterType::Pointer w = WriterType::New();
w->SetFileName("grad.mha");
w->SetInput(res->GetOutput());
w->Update();
*/

  // Zero out the grad mag outside of the mask
  if (!m_Mask.IsNull())
  {
    typedef itk::ImageRegionIteratorWithIndex<InputImageType> IteratorType;
    IteratorType it(gradImage, gradImage->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      OutputImageIndexType ind = it.GetIndex();
      if (m_Mask->GetPixel(ind) == 0)
        it.Set(0);
    }
  }

  // Watershed
  typedef itk::WatershedImageFilter<InputImageType> WatershedFilterType;

  typename WatershedFilterType::Pointer watershedFilter =
    WatershedFilterType::New();
  watershedFilter->SetInput(gradImage);
  watershedFilter->SetLevel(0.0);
  watershedFilter->SetThreshold(0.0);
  watershedFilter->Update();

/*
  // Zero out the partitions outside the mask and set min label to 1
  if (!m_Mask.IsNull())
  {
    OutputImagePointer wImage = watershedFilter->GetOutput();

    typedef itk::ImageRegionIteratorWithIndex<OutputImageType> IteratorType;
    IteratorType it(wImage, wImage->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      OutputImageIndexType ind = it.GetIndex();
      if (m_Mask->GetPixel(ind) == 0)
        it.Set(0);
      else
        it.Set(it.Get()+1);
    }
    m_Output = wImage;
  }
  else
  {
    m_Output = watershedFilter->GetOutput();
  }
*/

  m_Output = watershedFilter->GetOutput();

}

#endif
