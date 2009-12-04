
#ifndef _MultimodalGradientImageFilter_txx
#define _MultimodalGradientImageFilter_txx

#include "itkGradientImageFilter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "vnl/vnl_matrix.h"
#include "vnl/algo/vnl_svd.h"

#include "MultimodalGradientImageFilter.h"

#include <cmath>

template <class TInputImage>
MultimodalGradientImageFilter<TInputImage>
::MultimodalGradientImageFilter()
{
}

template <class TInputImage>
void
MultimodalGradientImageFilter<TInputImage>
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
}

template <class TInputImage>
typename MultimodalGradientImageFilter<TInputImage>::OutputImagePointer
MultimodalGradientImageFilter<TInputImage>
::ComputeGradientMagnitudeImage()
{
  // Build list of gradient images
  typedef itk::BSplineInterpolateImageFunction<InputImageType>
    InterpolatorType;

  DynArray<typename InterpolatorType::Pointer> interpolators;
  interpolators.Allocate(m_InputImages.GetSize());
  for (unsigned int i = 0; i < m_InputImages.GetSize(); i++)
  {
    typename InterpolatorType::Pointer ip = InterpolatorType::New();
    ip->SetInputImage(m_InputImages[i]);
    ip->SetSplineOrder(3);
    interpolators.Append(ip);
  }

  // Allocate magnitude image
  InputImagePointer magImage = InputImageType::New();
  magImage->SetRegions(m_InputImages[0]->GetLargestPossibleRegion());
  magImage->CopyInformation(m_InputImages[0]);
  magImage->Allocate();
  magImage->FillBuffer(0);

  // Compute grad mag using SVD (largest singular value)
  typedef itk::ImageRegionIteratorWithIndex<InputImageType>
    InputIteratorType;

  typedef vnl_matrix<float> MatrixType;
  typedef vnl_svd<float> MatrixSVDType;

  unsigned int numChannels = m_InputImages.GetSize();
  unsigned int imageDimension = InputImageType::ImageDimension;

  InputIteratorType magIt(magImage, magImage->GetLargestPossibleRegion());
  for (magIt.GoToBegin(); !magIt.IsAtEnd(); ++magIt)
  {
    InputImageIndexType ind = magIt.GetIndex();

    if (!m_Mask.IsNull() && m_Mask->GetPixel(ind) == 0)
      continue;

    InputImagePointType p;
    magImage->TransformIndexToPhysicalPoint(ind, p);

    MatrixType D(numChannels, imageDimension, 0.0);
    for (unsigned int i = 0; i < numChannels; i++)
    {
      typename InterpolatorType::CovariantVectorType dI =
        interpolators[i]->EvaluateDerivative(p);
      for (unsigned int j = 0; j < imageDimension; j++)
        D(i, j) = dI[j];
    }

    MatrixSVDType svd(D);
    magIt.Set(svd.W(0,0));
  }

  return magImage;
}

template <class TInputImage>
DynArray<
  typename MultimodalGradientImageFilter<TInputImage>::OutputImagePointer>
MultimodalGradientImageFilter<TInputImage>
::ComputeGradientImage()
{
  // Build list of gradient images
  typedef itk::GradientImageFilter<InputImageType> GradientFilterType;
  typedef typename GradientFilterType::OutputImageType GradientImageType;
  typedef typename GradientFilterType::OutputPixelType GradientType;

  DynArray<typename GradientImageType::Pointer> gradImages;
  gradImages.Allocate(m_InputImages.GetSize());
  for (unsigned int i = 0; i < m_InputImages.GetSize(); i++)
  {
    typename GradientFilterType::Pointer gradf = GradientFilterType::New();
    gradf->SetInput(m_InputImages[i]);
    gradf->Update();
    gradImages.Append(gradf->GetOutput());
  }

  // Allocate gradient image
  DynArray<InputImagePointer> outGradImage;
  for (unsigned int i = 0; i < m_InputImages.GetSize(); i++)
  {
    InputImagePointer out_i = InputImageType::New();
    out_i->CopyInformation(m_InputImages[0]);
    out_i->SetRegions(m_InputImages[0]->GetLargestPossibleRegion());
    out_i->Allocate();
    out_i->FillBuffer(0);

    outGradImage.Append(out_i);
  }

  // Compute grad direction using SVD (right vector of largest singular value)
  typedef itk::ImageRegionIteratorWithIndex<InputImageType>
    InputIteratorType;

  typedef vnl_matrix<float> MatrixType;
  typedef vnl_vector<float> VectorType;
  typedef vnl_svd<float> MatrixSVDType;

  unsigned int numChannels = m_InputImages.GetSize();
  unsigned int imageDimension = InputImageType::ImageDimension;

  InputIteratorType it(
    outGradImage[0], outGradImage[0]->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    InputImageIndexType ind = it.GetIndex();

    if (!m_Mask.IsNull() && m_Mask->GetPixel(ind) == 0)
      continue;

    MatrixType D(numChannels, imageDimension, 0.0);
    for (unsigned int i = 0; i < numChannels; i++)
    {
      GradientType v = gradImages[i]->GetPixel(ind);
      for (unsigned int j = 0; j < imageDimension; j++)
        D(i, j) = v[j];
    }

    MatrixSVDType svd(D);
    MatrixType rightM = svd.V();

    VectorType dir = rightM.get_row(0);
    dir /= dir.magnitude() + 1e-20;

    for (unsigned int i = 0; i < numChannels; i++)
      outGradImage[i]->SetPixel(ind, dir[i]);
  }

  return outGradImage;
}

#endif
