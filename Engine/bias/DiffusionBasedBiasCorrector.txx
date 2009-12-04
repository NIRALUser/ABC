
#ifndef _DiffusionBasedBiasCorrector_txx
#define _DiffusionBasedBiasCorrector_txx

#include "itkCurvatureFlowImageFilter.h"
#include "itkGradientAnisotropicDiffusionImageFilter.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "DiffusionBasedBiasCorrector.h"

#include "vnl/vnl_math.h"

#include <cmath>

#include <iostream>

#define EXPP(x) (exp((x)/100.0) - 1.0)
#define LOGP(x) (100.0 * log((x)+1.0))
//#define EXPP(x) (exp(x) - 1)
//#define LOGP(x) (log((x)+1))

template <class TInputImage, class TProbabilityImage>
DiffusionBasedBiasCorrector <TInputImage, TProbabilityImage>
::DiffusionBasedBiasCorrector()
{

  m_InputData = 0;

  m_DoLog = false;

  m_DiffusionIterations = 20;
  m_DiffusionTimeStep = 0.05;

}

template <class TInputImage, class TProbabilityImage>
DiffusionBasedBiasCorrector <TInputImage, TProbabilityImage>
::~DiffusionBasedBiasCorrector()
{

}

template <class TInputImage, class TProbabilityImage>
void
DiffusionBasedBiasCorrector <TInputImage, TProbabilityImage>
::CheckInput()
{

  if (m_InputData.IsNull())
    itkExceptionMacro(<< "Input image not initialized");

  if (m_Probabilities.GetSize() < 1)
    itkExceptionMacro(<< "Must have one or more class probabilities");

  InputImageSizeType size =
    m_InputData->GetLargestPossibleRegion().GetSize();

  for (int i = 0; i < m_Probabilities.GetSize(); i++)
  {
    if (m_Probabilities[i]->GetImageDimension() != 3)
      itkExceptionMacro(<< "Probability [" << i << "] has invalid dimension: only supports 3D images");
    ProbabilityImageSizeType psize =
      m_Probabilities[i]->GetLargestPossibleRegion().GetSize();
    if (size[0] != psize[0] || size[1] != psize[1] || size[2] != psize[2])
      itkExceptionMacro(<< "Image data and probabilities 3D size mismatch");
  }

}

template <class TInputImage, class TProbabilityImage>
typename DiffusionBasedBiasCorrector<TInputImage, TProbabilityImage>::InternalImagePointer
DiffusionBasedBiasCorrector <TInputImage, TProbabilityImage>
::ComputeResidualImage()
{

  itkDebugMacro(<< "DiffusionBasedBiasCorrector: Computing means...");

  unsigned int numClasses = m_Probabilities.GetSize();

  DynArray<double> meanValues;
  meanValues.Initialize(numClasses, 0);

  DynArray<double> sumProbs;
  sumProbs.Initialize(numClasses, 0);

  typedef itk::ImageRegionIteratorWithIndex<InputImageType> IteratorType;

  IteratorType it(m_InputData, m_InputData->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    InputImageIndexType ind = it.GetIndex();

    double v = it.Get();
    if (m_DoLog)
      v = LOGP(v);

    for (unsigned int i = 0; i < numClasses; i++)
    {
      double p = m_Probabilities[i]->GetPixel(ind);
      meanValues[i] += p*v;
      sumProbs[i] += p;
    }
  }

  for (unsigned int i = 0; i < numClasses; i++)
    if (sumProbs[i] > 0)
      meanValues[i] /= sumProbs[i];

  itkDebugMacro(<< "DiffusionBasedBiasCorrector: Computing residual...");

  InternalImagePointer resImg = InternalImageType::New();
  resImg->SetRegions(m_InputData->GetLargestPossibleRegion());
  resImg->Allocate();
  resImg->SetOrigin(m_InputData->GetOrigin());
  resImg->SetSpacing(m_InputData->GetSpacing());
  resImg->FillBuffer(0);

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    InputImageIndexType ind = it.GetIndex();

    double v = it.Get();
    if (m_DoLog)
      v = LOGP(v);

    double ptot = 0;
    for (unsigned int i = 0; i < numClasses; i++)
      ptot += m_Probabilities[i]->GetPixel(ind);

    if (ptot < 1e-10)
      continue;

    double v_flat = 0;
    for (unsigned int i = 0; i < numClasses; i++)
    {
      double p = m_Probabilities[i]->GetPixel(ind);
      v_flat += p * meanValues[i];
    }

    resImg->SetPixel(ind, v - v_flat);
  }

  return resImg;

}

template <class TInputImage, class TProbabilityImage>
void
DiffusionBasedBiasCorrector <TInputImage, TProbabilityImage>
::SetProbabilities(DynArray<ProbabilityImagePointer> probs)
{

  itkDebugMacro(<< "SetProbabilities");

  if (probs.GetSize() < 1)
    itkExceptionMacro(<<"Need one or more probabilities");

  for (int i = 0; i < probs.GetSize(); i++)
  {
    if (probs[i].IsNull())
      itkExceptionMacro(<<"One of input probabilities not initialized");
  }

  m_Probabilities = probs;

}

template <class TInputImage, class TProbabilityImage>
void
DiffusionBasedBiasCorrector <TInputImage, TProbabilityImage>
::Correct(InputImagePointer input, InputImagePointer output)
{
  // For convenience
  m_InputData = input;

  // Verify input
  this->CheckInput();

  unsigned int numClasses = m_Probabilities.GetSize();

  // Compute residual image
  InternalImagePointer resImg = this->ComputeResidualImage();

  // Do anisotropic blurring on the residual image
#if 0
  typedef itk::CurvatureFlowImageFilter<InternalImageType, InternalImageType>
    CurvatureFilterType;
  typename CurvatureFilterType::Pointer cfilt = CurvatureFilterType::New();

  cfilt->SetNumberOfIterations(m_DiffusionIterations);
  cfilt->SetTimeStep(m_DiffusionTimeStep);
  cfilt->SetInput(resImg);
  cfilt->Update();

  resImg = cfilt->GetOutput();
#else
  typedef itk::GradientAnisotropicDiffusionImageFilter<InternalImageType, InternalImageType>
    AnisoFilterType;
  typename AnisoFilterType::Pointer afilt = AnisoFilterType::New();

  afilt->SetNumberOfIterations(m_DiffusionIterations);
  afilt->SetTimeStep(m_DiffusionTimeStep);
  afilt->SetInput(resImg);
  afilt->Update();

  resImg = afilt->GetOutput();
#endif

  // Apply correction
  typedef itk::ImageRegionIteratorWithIndex<InputImageType> IteratorType;

  IteratorType it(m_InputData, m_InputData->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    InputImageIndexType ind = it.GetIndex();

    double v = it.Get();
    if (m_DoLog)
      v = LOGP(v);

    double b = resImg->GetPixel(ind);

    double v_flat = v - b;
    if (m_DoLog)
      v_flat = EXPP(v_flat);

    output->SetPixel(ind, (InputImagePixelType)v_flat);
  }

  // Remove reference to input data when done
  m_InputData = 0;

}

#endif
