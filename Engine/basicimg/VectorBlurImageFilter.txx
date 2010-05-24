
#ifndef _VectorBlurImageFilter_txx
#define _VectorBlurImageFilter_txx
#include "VectorBlurImageFilter.h"

#include "itkImageRegionIteratorWithIndex.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"

#include <vector>

template <class TInputImage, class TOutputImage>
VectorBlurImageFilter<TInputImage, TOutputImage>
::VectorBlurImageFilter()
{
  m_KernelWidth = 1.0;
}

template <class TInputImage, class TOutput>
void
VectorBlurImageFilter<TInputImage, TOutput>
::PrintSelf(
  std::ostream& os, 
  itk::Indent indent) const
{
  Superclass::PrintSelf( os, indent );
  os << "Kernel width = " << m_KernelWidth << std::endl;
}

template< class TInputImage, class TOutputImage>
void
VectorBlurImageFilter< TInputImage, TOutputImage>
::GenerateData()
{
  if (this->GetInput() == 0)
    return;

  InputImageRegionType region = this->GetInput()->GetLargestPossibleRegion();
  this->GetOutput()->CopyInformation(this->GetInput());
  this->GetOutput()->SetRegions(region);
  this->GetOutput()->Allocate();

  typedef itk::Image<typename OutputPixelType::ComponentType, InputImageDimension>
    ScalarImageType;

  for (unsigned int dim = 0; dim < InputImageDimension; dim++)
  {

    typename ScalarImageType::Pointer tmp = ScalarImageType::New();
    tmp->CopyInformation(this->GetInput());
    tmp->SetRegions(region);
    tmp->Allocate();

    typedef itk::ImageRegionIteratorWithIndex<ScalarImageType> IteratorType;

    IteratorType it(tmp, region);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      InputImageIndexType ind = it.GetIndex();
      it.Set(this->GetInput()->GetPixel(ind)[dim]);
    }

#if 1
    typedef itk::SmoothingRecursiveGaussianImageFilter<
      ScalarImageType, ScalarImageType>
      BlurFilterType;
    typename BlurFilterType::Pointer blurf = BlurFilterType::New();
    blurf->SetInput(tmp);
    blurf->SetSigma(m_KernelWidth);
    blurf->SetNormalizeAcrossScale(false);
    blurf->Update();
#else
    typedef itk::DiscreteGaussianImageFilter<
      ScalarImageType, ScalarImageType>
      BlurFilterType;
    typename BlurFilterType::Pointer blurf = BlurFilterType::New();
    blurf->SetInput(tmp);
    blurf->SetVariance(m_KernelWidth*m_KernelWidth);
    blurf->Update();
#endif

    typename ScalarImageType::Pointer smoothtmp = blurf->GetOutput();

    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      InputImageIndexType ind = it.GetIndex();
      OutputPixelType v = this->GetOutput()->GetPixel(ind);
      v[dim] = smoothtmp->GetPixel(ind);
      this->GetOutput()->SetPixel(ind, v);
    }

  } // for dim

}

#endif
