
#ifndef _EMSegmentationFilter_txx
#define _EMSegmentationFilter_txx

#include "EMSegmentationFilter.h"

#include "itkBinaryBallStructuringElement.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkBinaryErodeImageFilter.h"
#include "itkBoxMeanImageFilter.h"
#include "itkBSplineDownsampleImageFilter.h"
#include "itkDiscreteGaussianImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkNumericTraits.h"
#include "itkResampleImageFilter.h"
#include "itkWarpImageFilter.h"

#include "itkAddImageFilter.h"
#include "itkDivideImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkSubtractImageFilter.h"

#include "itkExpImageFilter.h"

#include "itkTsallisLogImageFilter.h"
#include "itkLogImageFilter.h"
#include "itkStatisticsImageFilter.h"

#include "vnl/algo/vnl_determinant.h"
#include "vnl/vnl_math.h"

#include "ConnectedComponentsFilter.h"
#include "LLSBiasCorrector.h"
#include "Log.h"
#include "MersenneTwisterRNG.h"

#include "FastMCDSampleFilter.h"
#include "KruskalMSTClusteringProcess.h"

#include "IntensityMatcher.h"
#include "SimpleGreedyFluidRegistration.h"
#include "MaxLikelihoodFluidWarpEstimator.h"

#include <iostream>

#include <math.h>
#include <stdlib.h>

#define FLUID_USE_PROBS 1

template <class TInputImage, class TProbabilityImage>
EMSegmentationFilter <TInputImage, TProbabilityImage>
::EMSegmentationFilter()
{

  m_Mask = 0;

  m_Labels = 0;

  m_SampleSpacing = 2.0;

  // Bias
  m_MaxBiasDegree = 4;
  //m_BiasLikelihoodTolerance = 1e-2;
// PP
  m_BiasLikelihoodTolerance = 2e-4;
  // NOTE: warp tol needs to be <= bias tol
  m_WarpLikelihoodTolerance = 2e-4;

  // EM convergence parameters
// PP
  //m_LikelihoodTolerance = 1e-5;
  m_LikelihoodTolerance = 2e-4;
  m_MaximumIterations = 30;

  m_InputModified = false;

  m_PriorWeights = VectorType(0);
  m_NumberOfGaussians = 0;

  m_PriorLookupTable = 0;

  m_FOVMask = 0;

  m_DoMSTClustering = false;

  m_DoWarp = false;

  m_WarpFluidIterations = 10;

  m_WarpFluidMaxStep = 0.5;

  m_WarpFluidKernelWidth = 10.0;

  m_InitialDistributionEstimator = "robust";
}

template <class TInputImage, class TProbabilityImage>
EMSegmentationFilter <TInputImage, TProbabilityImage>
::~EMSegmentationFilter()
{

  delete [] m_NumberOfGaussians;
  delete [] m_PriorLookupTable;

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::CheckInput()
{
  itkDebugMacro(<< "CheckInput");

  if (m_MaximumIterations == 0)
    itkExceptionMacro(<< "Maximum iterations set to zero");

  if (m_InputImages.GetSize() == 0)
    itkExceptionMacro(<< "No input images");

  if (m_Priors.GetSize() < 1)
    itkExceptionMacro(<< "Must have one or more class probabilities");

  InputImageSizeType size =
    m_InputImages[0]->GetLargestPossibleRegion().GetSize();

  for (unsigned i = 1; i < m_InputImages.GetSize(); i++) {
    if (m_InputImages[i]->GetImageDimension() != 3)
      itkExceptionMacro(<< "InputImage [" << i << "] has invalid dimension: only supports 3D images");
    InputImageSizeType isize =
      m_InputImages[i]->GetLargestPossibleRegion().GetSize();
    if (size != isize)
      itkExceptionMacro(<< "Image data 3D size mismatch");
  }

  for (unsigned i = 0; i < m_Priors.GetSize(); i++)
  {
    if (m_Priors[i]->GetImageDimension() != 3)
      itkExceptionMacro(<< "Prior [" << i << "] has invalid dimension: only supports 3D images");
    ProbabilityImageSizeType psize =
      m_Priors[i]->GetLargestPossibleRegion().GetSize();
    if (size != psize)
      itkExceptionMacro(<< "Image data and prior 3D size mismatch");
  }

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::SetInputImages(DynArray<InputImagePointer> data)
{

  itkDebugMacro(<< "SetInputImages");

  if (data.GetSize() == 0)
    itkExceptionMacro(<< "No input images");

  m_InputImages.Clear();
  for (unsigned i = 0; i < data.GetSize(); i++)
    m_InputImages.Append(data[i]);

  // Allocate space for bias corrected input images
  m_CorrectedImages.Clear();
  for (unsigned i = 0; i < data.GetSize(); i++)
  {
    typedef itk::ImageDuplicator<InputImageType> DuperType;
    typename DuperType::Pointer dup = DuperType::New();
    dup->SetInputImage(m_InputImages[i]);
    dup->Update();

    m_CorrectedImages.Append(dup->GetOutput());
  }

  m_InputModified = true;

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::SetPriors(DynArray<ProbabilityImagePointer> priors)
{

  itkDebugMacro(<< "SetPriors");

  unsigned int numPriors = priors.GetSize();

  m_Priors.Clear();
  for (unsigned i = 0; i < numPriors; i++)
    m_Priors.Append(priors[i]);

  ProbabilityImageIndexType ind;
  ProbabilityImageSizeType size =
    m_Priors[0]->GetLargestPossibleRegion().GetSize();

  // Clamp minimum to zero
  for (unsigned iprior = 0; iprior < numPriors; iprior++)
  {
    ProbabilityImagePointer tmp = m_Priors[iprior];
    for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]++)
      for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]++)
        for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]++)
        {
          if (tmp->GetPixel(ind) < 0)
            tmp->SetPixel(ind, 0);
        }
  }

  // Create brain mask for warping
  m_OriginalMask = ByteImageType::New();
  m_OriginalMask->CopyInformation(m_Priors[0]);
  m_OriginalMask->SetRegions(m_Priors[0]->GetLargestPossibleRegion());
  m_OriginalMask->Allocate();
  m_OriginalMask->FillBuffer(0);

  // Normalize priors
  for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]++)
    for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]++)
      for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]++)
      {
        double sumPrior = 0;
        for (unsigned iprior = 0; iprior < numPriors; iprior++)
          sumPrior += m_Priors[iprior]->GetPixel(ind);
        if (sumPrior < 1e-10)
          continue;
        for (unsigned iprior = 0; iprior < numPriors; iprior++)
          m_Priors[iprior]->SetPixel(ind,
            m_Priors[iprior]->GetPixel(ind) / sumPrior + 1e-10);
        m_OriginalMask->SetPixel(ind, 1);
      }

  m_OriginalPriors = m_Priors;

  m_PriorWeights = VectorType(numPriors);
  m_PriorWeights.fill(1.0);

  delete [] m_NumberOfGaussians;
  m_NumberOfGaussians = new unsigned int[numPriors];
  for (unsigned int i = 0; i < numPriors; i++)
    m_NumberOfGaussians[i] = 1;
  m_NumberOfGaussians[numPriors-1] = 3;

  m_InputModified = true;

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::SetPriorWeights(VectorType w)
{

  itkDebugMacro(<< "SetPriorWeights");

  if (w.size() > m_Priors.GetSize())
    itkExceptionMacro(<< "Number of prior weights invalid");

  m_PriorWeights = VectorType(m_Priors.GetSize());
  m_PriorWeights.fill(1.0);

  for (unsigned i = 0; i < w.size() ; i++)
  {
    if (w[i] == 0.0)
      itkExceptionMacro(<< "Prior weight " << i << " is zero");
    m_PriorWeights[i] = w[i];
  }

  m_InputModified = true;

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::SetNumberOfGaussians(unsigned int* ng)
{

  itkDebugMacro(<< "SetNumberOfGaussians");

  if (ng == NULL)
    itkExceptionMacro(<< "Number of cluster info invalid");

  unsigned int numPriors = m_Priors.GetSize();

  for (unsigned i = 0; i < numPriors; i++)
    if (ng[i] == 0)
      itkExceptionMacro(<< "Number of Gaussians for prior " << i << " is zero");

  for (unsigned int i = 0; i < numPriors; i++)
    m_NumberOfGaussians[i] = ng[i];

  m_InputModified = true;

}

template <class TInputImage, class TProbabilityImage>
typename EMSegmentationFilter <TInputImage, TProbabilityImage>::
ByteImagePointer
EMSegmentationFilter <TInputImage, TProbabilityImage>
::GetOutput()
{

  this->Update();

  return m_Labels;

}

template <class TInputImage, class TProbabilityImage>
DynArray<
  typename EMSegmentationFilter <TInputImage, TProbabilityImage>::
  ByteImagePointer>
EMSegmentationFilter <TInputImage, TProbabilityImage>
::GetBytePosteriors()
{

  this->Update();

  unsigned int numClasses = m_Posteriors.GetSize();

  DynArray<ByteImagePointer> bytePosts;
  bytePosts.Allocate(numClasses);

  double max = itk::NumericTraits<unsigned char>::max();

  for (unsigned int iclass = 0; iclass < numClasses; iclass++)
  { 
    ProbabilityImagePointer post = m_Posteriors[iclass];
    
    ByteImagePointer tmp = ByteImageType::New();
    tmp->CopyInformation(post);
    tmp->SetRegions(post->GetLargestPossibleRegion());
    tmp->Allocate();
    
    ProbabilityImageIndexType ind;

    ProbabilityImageSizeType size = post->GetLargestPossibleRegion().GetSize();

    for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]++)
      for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]++)
        for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]++)
        {
          double p = floor(max*post->GetPixel(ind) + 0.5);
          if (p < 0)
            p = 0;
          if (p > max)
            p = max;
          tmp->SetPixel(ind, (unsigned char)p);
        }

    bytePosts.Append(tmp);
  }

  return bytePosts;

}

template <class TInputImage, class TProbabilityImage>
DynArray<
  typename EMSegmentationFilter <TInputImage, TProbabilityImage>::
  ShortImagePointer>
EMSegmentationFilter <TInputImage, TProbabilityImage>
::GetShortPosteriors()
{

  this->Update();

  unsigned int numClasses = m_Posteriors.GetSize();

  DynArray<ShortImagePointer> shortPosts;
  shortPosts.Allocate(numClasses);

  double max = itk::NumericTraits<short>::max();

  for (unsigned int iclass = 0; iclass < numClasses; iclass++)
  { 
    ProbabilityImagePointer post = m_Posteriors[iclass];
    
    ShortImagePointer tmp = ShortImageType::New();
    tmp->CopyInformation(post);
    tmp->SetRegions(post->GetLargestPossibleRegion());
    tmp->Allocate();
    
    ProbabilityImageIndexType ind;

    ProbabilityImageSizeType size = post->GetLargestPossibleRegion().GetSize();

    for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]++)
      for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]++)
        for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]++)
        {
          double p = floor(max*post->GetPixel(ind) + 0.5);
          if (p < 0)
            p = 0;
          if (p > max)
            p = max;
          tmp->SetPixel(ind, (short)p);
        }

    shortPosts.Append(tmp);
  }

  return shortPosts;

}

template <class TInputImage, class TProbabilityImage>
DynArray<
  typename EMSegmentationFilter <TInputImage, TProbabilityImage>::
  ProbabilityImagePointer>
EMSegmentationFilter <TInputImage, TProbabilityImage>
::GetPosteriors()
{

  this->Update();

  return m_Posteriors;

}

template <class TInputImage, class TProbabilityImage>
DynArray<
  typename EMSegmentationFilter <TInputImage, TProbabilityImage>::
  InputImagePointer>
EMSegmentationFilter <TInputImage, TProbabilityImage>
::GetCorrected()
{

  this->Update();

  return m_CorrectedImages;

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::Update()
{
  itkDebugMacro(<< "Update");

  if (!m_InputModified)
    return;

  this->CheckInput();

  InputImageSpacingType spacing = m_InputImages[0]->GetSpacing();
  double minspacing = spacing[0];
  for (int i = 1; i < 3; i++)
    if (spacing[i] < minspacing)
      minspacing = spacing[i];
  m_SampleSpacing = 2.0 * minspacing;

  this->ComputeMask();

  this->ComputePriorLookupTable();

  this->EMLoop();

  this->ComputeLabels();

  this->CleanUp();

  m_InputModified = false;

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::ComputePriorLookupTable()
{

  itkDebugMacro(<< "ComputePriorLookupTable");

  unsigned int numPriors = m_Priors.GetSize();

  unsigned int numClasses = 0;
  for (unsigned int i = 0; i < numPriors; i++)
    numClasses += m_NumberOfGaussians[i];

  delete [] m_PriorLookupTable;
  m_PriorLookupTable = new unsigned int[numClasses];

  unsigned int itab = 0;
  for (unsigned int iprior = 0; iprior < numPriors; iprior++)
  {
    unsigned int n = m_NumberOfGaussians[iprior];
    for (unsigned int j = 0; j < n; j++)
    {
      m_PriorLookupTable[itab+j] = iprior;
    }
    itab += n;
  }

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::ComputeMask()
{

  itkDebugMacro(<< "ComputeMask");

  unsigned int numPriors = m_Priors.GetSize();

  m_Mask = ByteImageType::New();
  m_Mask->CopyInformation(m_Priors[0]);
  m_Mask->SetRegions(m_Priors[0]->GetLargestPossibleRegion());
  m_Mask->Allocate();

  ProbabilityImageIndexType ind;

  ProbabilityImageSizeType size =
    m_Priors[0]->GetLargestPossibleRegion().GetSize();

  for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]++)
    for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]++)
      for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]++)
      {
        double sumP = 0;
        for (unsigned int iprior = 0; iprior < (numPriors-1); iprior++)
          sumP += m_Priors[iprior]->GetPixel(ind);
        if (sumP > 1e-10)
          m_Mask->SetPixel(ind, 1);
        else
          m_Mask->SetPixel(ind, 0);
      }

  if (!m_FOVMask.IsNull())
  {
    if (size != m_FOVMask->GetLargestPossibleRegion().GetSize())
      itkExceptionMacro(<< "FOV mask size mismatch");

    for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]++)
      for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]++)
        for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]++)
        {
          if (m_FOVMask->GetPixel(ind) == 0)
            m_Mask->SetPixel(ind, 0);
        }
  }

#if 0
  // Dilate mask
  typedef itk::BinaryBallStructuringElement<unsigned char, 3> StructElementType;
  typedef
    itk::BinaryDilateImageFilter<ByteImageType, ByteImageType,
      StructElementType> DilateType;

  StructElementType structel;
  structel.SetRadius(2);
  structel.CreateStructuringElement();

  typename DilateType::Pointer dil = DilateType::New();
  dil->SetDilateValue(1);
  dil->SetKernel(structel);
  dil->SetInput(m_Mask);

  dil->Update();

  m_Mask = dil->GetOutput();
#endif

}

#if 1
template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::ComputeDistributions()
{
  itkDebugMacro(<< "ComputeDistributions");

  unsigned numChannels = m_InputImages.GetSize();
  unsigned numPriors = m_Priors.GetSize();

  unsigned int numClasses = 0;
  for (unsigned int i = 0; i < numPriors; i++)
    numClasses += m_NumberOfGaussians[i];

  typedef itk::AddImageFilter<ProbabilityImageType, ProbabilityImageType, ProbabilityImageType>
    AddFilterType;
  typedef itk::DivideImageFilter<ProbabilityImageType, ProbabilityImageType, ProbabilityImageType>
    DivFilterType;
  typedef itk::MultiplyImageFilter<ProbabilityImageType, InputImageType, ProbabilityImageType>
    MulFilterType;
  typedef itk::SubtractImageFilter<ProbabilityImageType, ProbabilityImageType, ProbabilityImageType>
    SubFilterType;

  DynArray<ProbabilityImagePointer> maskedPosteriors;
  for (unsigned int iclass = 0; iclass < numClasses; iclass++)
  {
    typedef itk::MultiplyImageFilter<ProbabilityImageType, ByteImageType, ProbabilityImageType>
      MaskFilterType;
    typename MaskFilterType::Pointer maskf = MaskFilterType::New();
    maskf->SetInput1(m_Posteriors[iclass]);
    maskf->SetInput2(m_Mask);
    maskf->Update();

    maskedPosteriors.Append(maskf->GetOutput());
  }

  typedef itk::StatisticsImageFilter<ProbabilityImageType> StatFilterType;

  VectorType sumClassProb(numClasses);
  for (unsigned iclass = 0; iclass < numClasses; iclass++)
  {
    typename StatFilterType::Pointer statf = StatFilterType::New();
    statf->SetInput(maskedPosteriors[iclass]);
    statf->Update();
    sumClassProb[iclass] = statf->GetSum() + 1e-20;
  }

  m_Means = MatrixType(numChannels, numClasses);

  for (unsigned int iclass = 0; iclass < (numClasses-1); iclass++)
  {
    for (unsigned int ichan = 0; ichan < numChannels; ichan++)
    {
      typename MulFilterType::Pointer mulf = MulFilterType::New();
      mulf->SetInput1(maskedPosteriors[iclass]);
      mulf->SetInput2(m_CorrectedImages[ichan]);
      mulf->Update();

      typename StatFilterType::Pointer statIf = StatFilterType::New();
      statIf->SetInput(mulf->GetOutput());
      statIf->Update();

      m_Means(ichan, iclass) = statIf->GetSum() / sumClassProb[iclass];
    }
  } // end means loop

  // Fix mean of last class to zero vector (background)
  for (unsigned int ichan = 0; ichan < numChannels; ichan++)
    m_Means(ichan, numClasses-1) = 0;

  // Compute covariances
  DynArray<MatrixType> oldCovariances = m_Covariances;
  if (oldCovariances.GetSize() != numClasses)
  {
    oldCovariances.Clear();
    for (unsigned int iclass = 0; iclass < numClasses; iclass++)
    {
      MatrixType C(numChannels, numChannels);
      C.set_identity();
      C *= 1e-10;
      oldCovariances.Append(C);
    }
  }

  m_Covariances.Clear();
  for (unsigned int iclass = 0; iclass < numClasses; iclass++)
  {
    MatrixType covtmp(numChannels, numChannels);

    for (unsigned int r = 0; r < numChannels; r++)
    {
      typename SubFilterType::Pointer subf1 = SubFilterType::New();
      subf1->SetInput1(m_CorrectedImages[r]);
      subf1->SetConstant2(m_Means(r, iclass));
      subf1->Update();

      typename MulFilterType::Pointer pmulf = MulFilterType::New();
      pmulf->SetInput1(maskedPosteriors[iclass]);
      pmulf->SetInput2(subf1->GetOutput());
      pmulf->Update();

      for (unsigned int c = r; c < numChannels; c++)
      {
        typename SubFilterType::Pointer subf2 = SubFilterType::New();
        subf2->SetInput1(m_CorrectedImages[c]);
        subf2->SetConstant2(m_Means(c, iclass));
        subf2->Update();

        typename MulFilterType::Pointer imulf = MulFilterType::New();
        imulf->SetInput1(pmulf->GetOutput());
        imulf->SetInput2(subf2->GetOutput());
        imulf->GetOutput();

        typename StatFilterType::Pointer statIf = StatFilterType::New();
        statIf->SetInput(imulf->GetOutput());
        statIf->Update();

        double v = statIf->GetSum() / sumClassProb[iclass];

        // Adjust diagonal, to make sure covariance is pos-def
        if (r == c)
          v += 1e-20;

        // Assign value to the covariance matrix (symmetric)
        covtmp(r, c) = v;
        covtmp(c, r) = v;

      }
    }

    double detcov = vnl_determinant(covtmp);

    // Hack: use old covariance if the new one is singular
    if (detcov < 1e-20)
      m_Covariances.Append(oldCovariances[iclass]);
    else
      m_Covariances.Append(covtmp);

  } // end covariance loop

}
#else
// Single threaded
template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::ComputeDistributions()
{
  itkDebugMacro(<< "ComputeDistributions");

  unsigned numChannels = m_InputImages.GetSize();
  unsigned numPriors = m_Priors.GetSize();

  unsigned int numClasses = 0;
  for (unsigned int i = 0; i < numPriors; i++)
    numClasses += m_NumberOfGaussians[i];

  InputImageIndexType ind;

  InputImageSizeType size =
    m_InputImages[0]->GetLargestPossibleRegion().GetSize();

  InputImageSpacingType spacing = m_InputImages[0]->GetSpacing();

  InputImageOffsetType skips;
  skips[0] = (unsigned int)fabs(m_SampleSpacing / spacing[0]);
  skips[1] = (unsigned int)fabs(m_SampleSpacing / spacing[1]);
  skips[2] = (unsigned int)fabs(m_SampleSpacing / spacing[2]);

  if (skips[0] == 0)
    skips[0] = 1;
  if (skips[1] == 0)
    skips[1] = 1;
  if (skips[2] == 0)
    skips[2] = 1;

  // Compute sum of posteriors for each class
  VectorType sumClassProb(numClasses);
  for (unsigned iclass = 0; iclass < numClasses; iclass++)
  {
    double tmp = vnl_math::eps;
    for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]+=skips[2])
      for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]+=skips[1])
        for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]+=skips[0])
        {
          if (m_Mask->GetPixel(ind) == 0)
            continue;
          tmp += (double)m_Posteriors[iclass]->GetPixel(ind);
        }
     sumClassProb[iclass] = tmp;
  }

  // Compute means
  m_Means = MatrixType(numChannels, numClasses);
  for (unsigned int iclass = 0; iclass < numClasses; iclass++)
  {
    for (unsigned int ichan = 0; ichan < numChannels; ichan++)
    {

      InputImagePointer img = m_CorrectedImages[ichan];

      double mu = 0;

      // Compute the class mean
      for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]+=skips[2])
        for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]+=skips[1])
          for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]+=skips[0])
          {
            if (m_Mask->GetPixel(ind) == 0)
              continue;
            mu += (double)
              (m_Posteriors[iclass]->GetPixel(ind) * img->GetPixel(ind));
          }

      mu /= sumClassProb[iclass];

      m_Means(ichan, iclass) = mu;

    }
  } // end means loop

  // Fix mean of last class to zero vector (background)
  for (unsigned int ichan = 0; ichan < numChannels; ichan++)
    m_Means(ichan, numClasses-1) = 0;

  // Compute covariances
  DynArray<MatrixType> oldCovariances = m_Covariances;
  if (oldCovariances.GetSize() != numClasses)
  {
    oldCovariances.Clear();
    for (unsigned int iclass = 0; iclass < numClasses; iclass++)
    {
      MatrixType C(numChannels, numChannels);
      C.set_identity();
      C *= 1e-10;
      oldCovariances.Append(C);
    }
  }
  m_Covariances.Clear();
  for (unsigned int iclass = 0; iclass < numClasses; iclass++)
  {

    MatrixType covtmp(numChannels, numChannels);

    for (unsigned int r = 0; r < numChannels; r++)
    {

      double mu1 = m_Means(r, iclass);
      InputImagePointer img1 = m_CorrectedImages[r];

      for (unsigned int c = r; c < numChannels; c++)
      {

        double mu2 = m_Means(c, iclass);
        InputImagePointer img2 = m_CorrectedImages[c];

        double v = 0;
        double diff1 = 0;
        double diff2 = 0;
        for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]+=skips[2])
          for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]+=skips[1])
            for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]+=skips[0])
            {
              if (m_Mask->GetPixel(ind) == 0)
                continue;
              diff1 = (double)(img1->GetPixel(ind)) - mu1;
              diff2 = (double)(img2->GetPixel(ind)) - mu2;
              v += (double)(m_Posteriors[iclass]->GetPixel(ind)) *
                (diff1*diff2);
            }

        v /= sumClassProb[iclass];

        // Adjust diagonal, to make sure covariance is pos-def
        if (r == c)
          v += 1e-20;

        // Assign value to the covariance matrix (symmetric)
        covtmp(r, c) = v;
        covtmp(c, r) = v;

      }

    }

    double detcov = vnl_determinant(covtmp);

    // Hack: use old covariance if the new one is singular
    if (detcov < 1e-20)
      m_Covariances.Append(oldCovariances[iclass]);
    else
      m_Covariances.Append(covtmp);

  } // end covariance loop

}
#endif

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::ComputeDistributionsRobust()
{
  itkDebugMacro(<< "ComputeDistributionsRobust");

  unsigned numChannels = m_InputImages.GetSize();
  unsigned numPriors = m_Priors.GetSize();

  unsigned int numClasses = 0;
  for (unsigned int i = 0; i < numPriors; i++)
    numClasses += m_NumberOfGaussians[i];

  InputImageIndexType ind;

  InputImageSizeType size =
    m_InputImages[0]->GetLargestPossibleRegion().GetSize();

  InputImageSpacingType spacing = m_InputImages[0]->GetSpacing();

  InputImageOffsetType skips;
  skips[0] = (unsigned int)fabs(m_SampleSpacing / spacing[0]);
  skips[1] = (unsigned int)fabs(m_SampleSpacing / spacing[1]);
  skips[2] = (unsigned int)fabs(m_SampleSpacing / spacing[2]);

  if (skips[0] == 0)
    skips[0] = 1;
  if (skips[1] == 0)
    skips[1] = 1;
  if (skips[2] == 0)
    skips[2] = 1;

  // Compute sum of posteriors for each class
  VectorType sumClassProb(numClasses);
  for (unsigned iclass = 0; iclass < numClasses; iclass++)
  {
    double tmp = vnl_math::eps;
    for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]+=skips[2])
      for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]+=skips[1])
        for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]+=skips[0])
        {
          if (m_Mask->GetPixel(ind) == 0)
            continue;
          tmp += (double)m_Posteriors[iclass]->GetPixel(ind);
        }
     sumClassProb[iclass] = tmp;
  }

  // Compute means
  m_Means = MatrixType(numChannels, numClasses, 0.0);
  for (unsigned int iclass = 0; iclass < numClasses; iclass++)
  {
    // Get samples with high probability

    typedef itk::ImageRegionConstIteratorWithIndex<ProbabilityImageType>
      IteratorType;

    IteratorType it(m_Posteriors[iclass], m_Posteriors[iclass]->GetRequestedRegion());

    double maxP = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      double p = it.Get();
      if (p > maxP)
        maxP = p;
    }

    double probThres = 0.9*maxP;

    unsigned int numPossibleSamples = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      ProbabilityImageIndexType ind = it.GetIndex();
      if (m_Mask->GetPixel(ind) == 0)
        continue;
      if (it.Get() >= probThres)
        numPossibleSamples++;
    }

    unsigned char* selectMask = new unsigned char[numPossibleSamples];
    for (unsigned int i = 0; i < numPossibleSamples; i++)
      selectMask[i] = 0;

    unsigned int numSamples = numPossibleSamples / 2 + 1;
    if (numSamples > 100000)
      numSamples = 100000;

    MersenneTwisterRNG* rng = MersenneTwisterRNG::GetGlobalInstance();

    if (numSamples < numPossibleSamples)
    {
      unsigned int* selectInd =
        rng->GenerateIntegerSequence(numSamples, numPossibleSamples-1);
      for (unsigned int i = 0; i < numSamples; i++)
        selectMask[selectInd[i]] = 1;
      delete [] selectInd;
    }
    else
    {
      for (unsigned int i = 0; i < numSamples; i++)
        selectMask[i] = 1;
    }

    FastMCDSampleFilter::SampleMatrixType sampleM(numSamples, numChannels, 0.0);

    unsigned int r = 0;
    unsigned int s = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      ProbabilityImageIndexType ind = it.GetIndex();
      if (m_Mask->GetPixel(ind) == 0)
        continue;

      if (it.Get() < probThres)
        continue;

      if (selectMask[r] != 0)
      {
        for (unsigned int ichan = 0; ichan < numChannels; ichan++)
          sampleM(s, ichan) = m_CorrectedImages[ichan]->GetPixel(ind);
        s++;

        if (s >= numSamples)
          break;
      }

      r++;

      if (r >= numPossibleSamples)
        break;
    }

    delete [] selectMask;


    // Compute robust mean
    FastMCDSampleFilter mcdf;

    mcdf.SetChangeTolerance(1e-4);
    mcdf.SetCoverFraction(0.5);
    mcdf.SetNumberOfStarts(100);
    mcdf.SetMaxCStepIterations(20);

    FastMCDSampleFilter::VectorType mu;
    FastMCDSampleFilter::MatrixType sigma; // Discarded

    mcdf.GetRobustEstimate(mu, sigma, sampleM);

    // m_Means.set_column(iclass, mu);
    for (unsigned int ichan = 0; ichan < numChannels; ichan++)
      m_Means(ichan, iclass) = mu[ichan];

  } // end means loop

  // Compute covariances
  DynArray<MatrixType> oldCovariances = m_Covariances;
  if (oldCovariances.GetSize() != numClasses)
  {
    oldCovariances.Clear();
    for (unsigned int iclass = 0; iclass < numClasses; iclass++)
    {
      MatrixType C(numChannels, numChannels);
      C.set_identity();
      C *= 1e-10;
      oldCovariances.Append(C);
    }
  }
  m_Covariances.Clear();
  for (unsigned int iclass = 0; iclass < numClasses; iclass++)
  {

    MatrixType covtmp(numChannels, numChannels, 0.0);

    for (unsigned int r = 0; r < numChannels; r++)
    {

      double mu1 = m_Means(r, iclass);
      InputImagePointer img1 = m_CorrectedImages[r];

      for (unsigned int c = r; c < numChannels; c++)
      {

        double mu2 = m_Means(c, iclass);
        InputImagePointer img2 = m_CorrectedImages[c];

        double v = 0;
        double diff1 = 0;
        double diff2 = 0;
        for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]+=skips[2])
          for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]+=skips[1])
            for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]+=skips[0])
            {
              if (m_Mask->GetPixel(ind) == 0)
                continue;
              diff1 = (double)(img1->GetPixel(ind)) - mu1;
              diff2 = (double)(img2->GetPixel(ind)) - mu2;
              v += (double)(m_Posteriors[iclass]->GetPixel(ind)) *
                (diff1*diff2);
            }

        v /= sumClassProb[iclass];

        // Adjust diagonal, to make sure covariance is pos-def
        if (r == c)
          v += 1e-20;

        // Assign value to the covariance matrix (symmetric)
        covtmp(r, c) = v;
        covtmp(c, r) = v;

      }

    }

    double detcov = vnl_determinant(covtmp);

    // Hack: use old covariance if the new one is singular
    if (detcov < 1e-20)
      m_Covariances.Append(oldCovariances[iclass]);
    else
      m_Covariances.Append(covtmp);

  } // end covariance loop

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::ClusterFromPriorMST(unsigned int iprior)
{
  itkDebugMacro(<< "ClusterFromPriorMST(" << iprior << ")");

  unsigned int numChannels = m_InputImages.GetSize();
  unsigned int numPriors = m_Priors.GetSize();

  if (iprior >= numPriors)
    itkExceptionMacro(<< "Invalid prior index");

  if (m_NumberOfGaussians[iprior] == 1)
    return; // No split necessary

  unsigned int numClasses = 0;
  for (unsigned int i = 0; i < numPriors; i++)
    numClasses += m_NumberOfGaussians[i];

  muLogMacro(<< "Splitting distributions for prior " << iprior+1 << "\n");

  InputImageIndexType ind;

  InputImageSizeType size =
    m_InputImages[0]->GetLargestPossibleRegion().GetSize();

  InputImageSpacingType spacing = m_InputImages[0]->GetSpacing();

  InputImageOffsetType skips;
  skips[0] = (unsigned int)fabs(m_SampleSpacing / spacing[0]);
  skips[1] = (unsigned int)fabs(m_SampleSpacing / spacing[1]);
  skips[2] = (unsigned int)fabs(m_SampleSpacing / spacing[2]);

  if (skips[0] == 0)
    skips[0] = 1;
  if (skips[1] == 0)
    skips[1] = 1;
  if (skips[2] == 0)
    skips[2] = 1;

  // Find beginning class index for this prior
  unsigned int istart = 0;
  for (unsigned int j = 0; j < iprior; j++)
    istart += m_NumberOfGaussians[j];

  ProbabilityImagePointer probImg = m_Priors[iprior];

  // Find max prob
  double maxP = 0.0;
  for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]++) 
    for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]++) 
      for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]++) 
      {
        if (m_Mask->GetPixel(ind) == 0)
          continue;
        if (probImg->GetPixel(ind) > maxP)
          maxP = probImg->GetPixel(ind);
      }

  // Select samples by thresholding prior with value above tau
  double tau = 0.85 * maxP;

  muLogMacro(<< "Sampling with threshold tau = " << tau << "\n");

  unsigned int numPossibleSamples = 0;
  for (ind[2] = 0; ind[2] < (long)size[2]; ind[2] += skips[2])
    for (ind[1] = 0; ind[1] < (long)size[1]; ind[1] += skips[1])
      for (ind[0] = 0; ind[0] < (long)size[0]; ind[0] += skips[0])
      {
        if (m_Mask->GetPixel(ind) == 0)
          continue;
        if (probImg->GetPixel(ind) >= tau)
          numPossibleSamples++;
     }

  // Sample selection mask
  unsigned char* selectMask = new unsigned char[numPossibleSamples];
  for (unsigned int i = 0; i < numPossibleSamples; i++)
    selectMask[i] = 0;

  unsigned int numSamples = numPossibleSamples;
  if (numSamples > 20000)
    numSamples = 20000;

  muLogMacro(<< "  Selecting " << numSamples << " / " << numPossibleSamples << "\n");

  MersenneTwisterRNG rng;

  if (numSamples < numPossibleSamples)
  {
    unsigned int c = 0;
    while (c < numSamples)
    {
      //double r = (double)rand() / (double)RAND_MAX;
      //unsigned int which = (unsigned int)((numPossibleSamples-1) * r);
      unsigned int which = (unsigned int)
        rng.GenerateUniformIntegerUpToK(numPossibleSamples - 1);
      if (selectMask[which] != 0)
        continue;
      selectMask[which] = 1;
      c++;
    }
  }
  else
  {
    for (unsigned int i = 0; i < numSamples; i++)
      selectMask[i] = 1;
  }

  DynArray<KruskalMSTClusteringProcess::VertexType> samples;
  samples.Allocate(numSamples);

  muLogMacro(<< "  Finding samples...\n");

  unsigned int r = 0;
  for (ind[2] = 0; ind[2] < (long)size[2]; ind[2] += skips[2])
    for (ind[1] = 0; ind[1] < (long)size[1]; ind[1] += skips[1])
      for (ind[0] = 0; ind[0] < (long)size[0]; ind[0] += skips[0])
      {
        if (m_Mask->GetPixel(ind) == 0)
          continue;
        if (probImg->GetPixel(ind) < tau)
          continue;

        if (selectMask[r] != 0)
        {
          KruskalMSTClusteringProcess::VertexType x(numChannels);
          for (unsigned int ichan = 0; ichan < numChannels; ichan++)
            x[ichan] = m_CorrectedImages[ichan]->GetPixel(ind);
          samples.Append(x);
        }

        ++r;
      }

  delete [] selectMask;

  muLogMacro(<< "  Create MST...\n");

  KruskalMSTClusteringProcess mstProc;
  mstProc.SetInputVertices(samples);
  mstProc.SortOn();

  muLogMacro(<< "  Allocate maps...\n");
  unsigned int* clusterMap = new unsigned int[numSamples];

  double T;
  for (T = 2.0; T >= 1.0; T -= 0.01)
  {
    muLogMacro(<< "  MST clustering, T = " << T << "\n");
    unsigned int numClusters = mstProc.GetClusters(clusterMap, T);

    if (numClusters < m_NumberOfGaussians[iprior])
      continue;

    // Check cluster sizes
    bool sizeOK = true;
    for (unsigned int m = 0; m < m_NumberOfGaussians[iprior]; m++)
    {
      unsigned int count = 0;
      for (unsigned int i = 0; i < numSamples; i++)
        if (clusterMap[i] == m)
          count++;
      if (count < 10)
        sizeOK = false;
    }
    
    if (sizeOK)
      break;
  }

  if (T < 1.0)
    itkExceptionMacro(<< "Failed clustering prior " << iprior+1);

  // Use the largest clusters to estimate the Gaussian parameters
  for (unsigned int m = 0; m < m_NumberOfGaussians[iprior]; m++)
  {
    unsigned int iclass = istart+m;

    unsigned int count = 0;
    for (unsigned int i = 0; i < numSamples; i++)
      if (clusterMap[i] == m)
        count++;

    muLogMacro(<< "  Estimating Gaussian parameters for class " << iclass);
    muLogMacro(<< " with " << count << " samples\n");

    for (unsigned int ichan = 0; ichan < numChannels; ichan++)
    { 
      double mu = 0.0; 
      for (unsigned int i = 0; i < numSamples; i++)
        if (clusterMap[i] == m)
        {
          KruskalMSTClusteringProcess::VertexType x = samples[i];
          mu += x[ichan];
        }
      mu /= count;
      m_Means(ichan, iclass) = mu;
    }

    MatrixType covtmp(numChannels, numChannels);
    for (unsigned int r = 0; r < numChannels; r++)
    { 
      double mu1 = m_Means(r, iclass);
      for (unsigned int c = r; c < numChannels; c++)
      { 
        double mu2 = m_Means(c, iclass);

        double v = 0.0;
        double diff1 = 0;
        double diff2 = 0;
        for (unsigned int i = 0; i < numSamples; i++)
          if (clusterMap[i] == m)
          {
            KruskalMSTClusteringProcess::VertexType x = samples[i];
            diff1 = x[r] - mu1;
            diff2 = x[c] - mu2;
            v += (diff1*diff2);
          }
        v /= (count - 1 + vnl_math::eps);

        if (r == c)
          v += 1e-20;

        covtmp(r, c) = v;
        covtmp(c, r) = v;
      }
    }

    // Scale the covariance up a bit for softer initialization
    covtmp *= 1.2;

    m_Covariances[iclass] = covtmp;

  }

  samples.Clear();

  delete [] clusterMap;

  // Special case for background, always set the "darkest" mean
  // to be the last and set it to zero
  if (iprior == (numPriors-1))
  {
    unsigned int imin = istart;
    double minm = m_Means.get_column(istart).squared_magnitude();
    for (unsigned int m = 1; m < m_NumberOfGaussians[iprior]; m++)
    {
      unsigned int iclass = istart+m;
      double mag = m_Means.get_column(iclass).squared_magnitude();
      if (mag < minm)
      {
        imin = iclass;
        minm = mag;
      }
    }

    if (imin != (numClasses-1))
    {
      muLogMacro(
        << "  Replacing " << m_Means.get_column(imin) << " with zero\n");
      VectorType v = m_Means.get_column(numClasses-1);
      m_Means.set_column(imin, v);
    }
    for (unsigned int ichan = 0; ichan < numChannels; ichan++)
      m_Means(ichan, numClasses-1) = 0;
  }

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::ClusterFromPrior(unsigned int iprior)
{
  itkDebugMacro(<< "ClusterFromPrior(" << iprior << ")");

  unsigned int numChannels = m_InputImages.GetSize();
  unsigned int numPriors = m_Priors.GetSize();

  if (iprior >= numPriors)
    itkExceptionMacro(<< "Invalid prior index");

  if (m_NumberOfGaussians[iprior] == 1)
    return; // No split necessary

  unsigned int numClasses = 0;
  for (unsigned int i = 0; i < numPriors; i++)
    numClasses += m_NumberOfGaussians[i];

  muLogMacro(<< "Splitting distributions for prior " << iprior+1 << "\n");

  // Get the start class index
  unsigned int istart = 0;
  for (unsigned int i = 0; i < iprior; i++)
    istart += m_NumberOfGaussians[i];

  // Scale the means
  for (unsigned int k = 0; k < m_NumberOfGaussians[iprior]; k++)
  {
    double s = 1.0 / pow(1.25, (double)k);
    unsigned int iclass = istart+k;
    for (unsigned int ichan = 0; ichan < numChannels; ichan++)
      m_Means(ichan, iclass) = s * m_Means(ichan, iclass);
  }

  // Set mean of last class to zero
  for (unsigned int ichan = 0; ichan < numChannels; ichan++)
    m_Means(ichan, numClasses-1) = 0;

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::ComputePosteriors(bool fullRes)
{
  itkDebugMacro(<< "ComputePosteriors");

  unsigned numChannels = m_InputImages.GetSize();
  unsigned numPriors = m_Priors.GetSize();

  unsigned int numClasses = 0;
  for (unsigned i = 0; i < numPriors; i++)
    numClasses += m_NumberOfGaussians[i];

  typedef itk::AddImageFilter<ProbabilityImageType, ProbabilityImageType, ProbabilityImageType>
    AddFilterType;
  typedef itk::DivideImageFilter<ProbabilityImageType, ProbabilityImageType, ProbabilityImageType>
    DivFilterType;
  typedef itk::MultiplyImageFilter<ProbabilityImageType, ProbabilityImageType, ProbabilityImageType>
    MulFilterType;
  typedef itk::SubtractImageFilter<ProbabilityImageType, ProbabilityImageType, ProbabilityImageType>
    SubFilterType;

  // Compute likelihoods
  for (unsigned int iclass = 0; iclass < numClasses; iclass++)
  {
    double detcov = vnl_determinant(m_Covariances[iclass]);

    if (detcov <= 0.0)
      itkExceptionMacro(<< "Determinant of covariance for class " << iclass
        << " is <= 0.0 (" << detcov << "), covariance matrix:\n"
        << m_Covariances[iclass]);

    // Normalizing constant for the Gaussian
    double denom =
      pow(2*vnl_math::pi, m_InputImages.GetSize()/2.0) * sqrt(detcov)
      + vnl_math::eps;

    MatrixType invcov = MatrixInverseType(m_Covariances[iclass]);

    DynArray<InputImagePointer> diffImages;
    for (unsigned int ichan = 0; ichan < numChannels; ichan++)
    {
      typename SubFilterType::Pointer subf = SubFilterType::New();
      subf->SetInput1(m_CorrectedImages[ichan]);
      subf->SetConstant2(m_Means(ichan, iclass));
      subf->Update();

      diffImages.Append(subf->GetOutput());
    }

    DynArray<InputImagePointer> scaledDiffImages;
    for (unsigned int r = 0; r < numChannels; r++)
    {
      InputImagePointer sImg = InputImageType::New();
      sImg->CopyInformation(m_CorrectedImages[r]);
      sImg->SetRegions(m_CorrectedImages[r]->GetLargestPossibleRegion());
      sImg->Allocate();
      sImg->FillBuffer(0);

      for (unsigned int c = 0; c < numChannels; c++)
      {
        typename MulFilterType::Pointer mulf = MulFilterType::New();
        mulf->SetInput1(diffImages[c]);
        mulf->SetConstant2(invcov(r, c));
        mulf->Update();

        typename AddFilterType::Pointer addf = AddFilterType::New();
        addf->SetInput1(sImg);
        addf->SetInput2(mulf->GetOutput());
        addf->Update();

        sImg = addf->GetOutput();
      }

      scaledDiffImages.Append(sImg);
    }

    m_Likelihoods[iclass]->FillBuffer(0);
    for (unsigned int ichan = 0; ichan < numChannels; ichan++)
    {
      typename MulFilterType::Pointer mulf = MulFilterType::New();
      mulf->SetInput1(diffImages[ichan]);
      mulf->SetInput2(scaledDiffImages[ichan]);
      mulf->Update();

      typename AddFilterType::Pointer addf = AddFilterType::New();
      addf->SetInput1(m_Likelihoods[iclass]);
      addf->SetInput2(mulf->GetOutput());
      addf->Update();

      m_Likelihoods[iclass] = addf->GetOutput();
    }

    typename MulFilterType::Pointer scalf = MulFilterType::New();
    scalf->SetInput1(m_Likelihoods[iclass]);
    scalf->SetConstant2(-0.5);
    scalf->Update();

    typedef itk::ExpImageFilter<ProbabilityImageType, ProbabilityImageType> ExpFilterType;
    typename ExpFilterType::Pointer expf = ExpFilterType::New();
    expf->SetInput(scalf->GetOutput());
    expf->Update();

    typename DivFilterType::Pointer divf = DivFilterType::New();
    divf->SetInput1(expf->GetOutput());
    divf->SetInput2(denom);
    divf->Update();

    // Mask the likelihoods
    typedef itk::MultiplyImageFilter<ProbabilityImageType, ByteImageType, ProbabilityImageType>
      MaskFilterType;
    typename MaskFilterType::Pointer maskf = MaskFilterType::New();
    maskf->SetInput1(divf->GetOutput());
    maskf->SetInput2(m_Mask);
    maskf->Update();

    m_Likelihoods[iclass] = maskf->GetOutput();

  } // for iclass

  // Compute posterior
  for (unsigned int iclass = 0; iclass < numClasses; iclass++)
  {
    unsigned int iprior = m_PriorLookupTable[iclass];

    double priorScale =
      m_PriorWeights[iprior] / m_NumberOfGaussians[iprior];

    typename MulFilterType::Pointer mulf = MulFilterType::New();
    mulf->SetInput1(m_Likelihoods[iclass]);
    mulf->SetInput2(m_Priors[iprior]);
    mulf->Update();

    typename MulFilterType::Pointer scalf = MulFilterType::New();
    scalf->SetInput1(mulf->GetOutput());
    scalf->SetConstant2(priorScale);
    scalf->Update();

    m_Posteriors[iclass] = scalf->GetOutput();
  }

#if 0
  if (fullRes)
    muLogMacro(<< "Computing posteriors at full resolution\n");

  ProbabilityImageIndexType ind;

  ProbabilityImageSizeType size =
    m_Priors[0]->GetLargestPossibleRegion().GetSize();

  ProbabilityImageSpacingType spacing = m_Priors[0]->GetSpacing();

  ProbabilityImageOffsetType skips;
  skips[0] = (unsigned int)fabs(m_SampleSpacing / spacing[0]);
  skips[1] = (unsigned int)fabs(m_SampleSpacing / spacing[1]);
  skips[2] = (unsigned int)fabs(m_SampleSpacing / spacing[2]);

  if (skips[0] == 0 || fullRes)
    skips[0] = 1;
  if (skips[1] == 0 || fullRes)
    skips[1] = 1;
  if (skips[2] == 0 || fullRes)
    skips[2] = 1;

  for (unsigned int iclass = 0; iclass < numClasses; iclass++)
  {
    unsigned int iprior = m_PriorLookupTable[iclass];

    ProbabilityImagePointer post = m_Posteriors[iclass];
    ProbabilityImagePointer prior = m_Priors[iprior];
    ProbabilityImagePointer lik = m_Likelihoods[iclass];

    post->FillBuffer(0);
    lik->FillBuffer(0);

    double priorScale =
      m_PriorWeights[iprior] / m_NumberOfGaussians[iprior];

    double detcov = vnl_determinant(m_Covariances[iclass]);

    if (detcov <= 0.0)
      itkExceptionMacro(<< "Determinant of covariance for class " << iclass
        << " is <= 0.0 (" << detcov << "), covariance matrix:\n"
        << m_Covariances[iclass]);

    // Normalizing constant for the Gaussian
    double denom =
      pow(2*vnl_math::pi, m_InputImages.GetSize()/2.0) * sqrt(detcov)
      + vnl_math::eps;
      
    MatrixType invcov = MatrixInverseType(m_Covariances[iclass]);

    for (ind[2] = 0; ind[2] < (long)size[2]; ind[2] += skips[2])
      for (ind[1] = 0; ind[1] < (long)size[1]; ind[1] += skips[1])
        for (ind[0] = 0; ind[0] < (long)size[0]; ind[0] += skips[0])
        {
          if (m_Mask->GetPixel(ind) == 0)
            continue;

          MatrixType X(numChannels, 1);
          for (unsigned int ichan = 0; ichan < numChannels; ichan++)
            X(ichan, 0) =
              m_CorrectedImages[ichan]->GetPixel(ind) - m_Means(ichan, iclass);

          MatrixType Y = invcov * X;

          double mahalo = 0.0;
          for (unsigned int ichan = 0; ichan < numChannels; ichan++)
            mahalo += X(ichan, 0) * Y(ichan, 0);

          double g = exp(-0.5 * mahalo) / denom;
          
          post->SetPixel(ind, (ProbabilityImagePixelType)
            (priorScale * prior->GetPixel(ind) * g));

          lik->SetPixel(ind, (ProbabilityImagePixelType) g);
        } // for 0


  } // end class loop
#endif

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::CorrectBias(unsigned int degree, bool fullRes)
{

  muLogMacro(<< "Bias correction, polynomial degree = " << degree << "\n");

  if (degree == 0)
    return;

  unsigned int numPriors = m_Priors.GetSize();

  unsigned int numFGClasses = 0;
  for (unsigned int i = 0; i < (numPriors-1); i++)
    numFGClasses += m_NumberOfGaussians[i];

  unsigned int numClasses = numFGClasses + m_NumberOfGaussians[numPriors-1];

  // Perform bias correction
  DynArray<ProbabilityImagePointer> biasPosteriors;
  for (unsigned j = 0; j < numClasses-1; j++)
  // Focus only on FG classes, more accurate if bg classification is bad
  // but sacrifices accuracy in border regions (tend to overcorrect)
  //for (unsigned j = 0; j < numFGClasses; j++)
    biasPosteriors.Append(m_Posteriors[j]);

  typedef LLSBiasCorrector<InputImageType, ProbabilityImageType>
    BiasCorrectorType;
  typedef typename BiasCorrectorType::Pointer BiasCorrectorPointer;

  BiasCorrectorPointer biascorr = BiasCorrectorType::New();

// TODO: need parameters for log(I)
  //biascorr->SetMeans(m_Means.get_n_columns(0, numClasses-1));
  //biascorr->SetCovariances(m_Covariances.Slice(0, numClasses-2));

  biascorr->SetClampBias(true);
  biascorr->SetMaximumBiasMagnitude(4.0);
  biascorr->SetMaxDegree(degree);
  biascorr->SetSampleSpacing(2.0*m_SampleSpacing);
  biascorr->SetWorkingSpacing(m_SampleSpacing);

  biascorr->SetMask(m_Mask);
  biascorr->SetProbabilities(biasPosteriors);

  if (this->GetDebug())
    biascorr->DebugOn();

  biascorr->CorrectImages(m_InputImages, m_CorrectedImages, fullRes);

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::EMLoop()
{

  itkDebugMacro(<< "EMLoop");

  unsigned int numChannels = m_InputImages.GetSize();
  unsigned int numPriors = m_Priors.GetSize();

  unsigned int numClasses = 0;
  for (unsigned int i = 0; i < numPriors; i++)
    numClasses += m_NumberOfGaussians[i];

  InputImageIndexType ind;

  InputImageSizeType size =
    m_InputImages[0]->GetLargestPossibleRegion().GetSize();

  InputImageSpacingType spacing = m_InputImages[0]->GetSpacing();

  InputImageOffsetType skips;
  skips[0] = (unsigned)fabs(m_SampleSpacing / spacing[0]);
  skips[1] = (unsigned)fabs(m_SampleSpacing / spacing[1]);
  skips[2] = (unsigned)fabs(m_SampleSpacing / spacing[2]);

  if (skips[0] == 0)
    skips[0] = 1;
  if (skips[1] == 0)
    skips[1] = 1;
  if (skips[2] == 0)
    skips[2] = 1;

  // Allocate space for posterior and likelihood images
  m_Posteriors.Clear();
  m_Likelihoods.Clear();
  for (unsigned iclass = 0; iclass < numClasses; iclass++)
  {
    ProbabilityImagePointer post = ProbabilityImageType::New();
    post->CopyInformation(m_InputImages[0]);
    post->SetRegions(m_InputImages[0]->GetLargestPossibleRegion());
    post->Allocate();
    post->FillBuffer(0);
    m_Posteriors.Append(post);

    ProbabilityImagePointer lik = ProbabilityImageType::New();
    lik->CopyInformation(m_InputImages[0]);
    lik->SetRegions(m_InputImages[0]->GetLargestPossibleRegion());
    lik->Allocate();
    lik->FillBuffer(0);
    m_Likelihoods.Append(lik);
  }

  // Initialize first iteration posteriors with masked, scaled priors
  for (unsigned iclass = 0; iclass < numClasses; iclass++)
  {
    unsigned iprior = m_PriorLookupTable[iclass];

    typedef itk::MultiplyImageFilter<ProbabilityImageType, ProbabilityImageType, ProbabilityImageType>
      MulFilterType;
    typename MulFilterType::Pointer mulf = MulFilterType::New();
    mulf->SetInput1(m_Priors[iprior]);
    mulf->SetConstant2(m_PriorWeights[iprior] / m_NumberOfGaussians[iprior]);
    mulf->Update();

    typedef itk::MultiplyImageFilter<ProbabilityImageType, ByteImageType, ProbabilityImageType>
      MaskFilterType;
    typename MaskFilterType::Pointer maskf = MaskFilterType::New();
    maskf->SetInput1(mulf->GetOutput());
    maskf->SetInput2(m_Mask);
    maskf->Update();

    m_Posteriors[iclass] = maskf->GetOutput();
  }
  this->NormalizePosteriors();

  // Compute the reference mean (first class)
  VectorType refMean(numChannels);
  for (unsigned int ichan = 0; ichan < numChannels; ichan++)
  {
    InputImagePointer img = m_InputImages[ichan];
    double mu = 0;
    double sumfirst = vnl_math::eps;
    for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]+=skips[2])
      for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]+=skips[1])
        for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]+=skips[0])
        {
          mu += (double)
            (m_Posteriors[0]->GetPixel(ind) * img->GetPixel(ind));
          sumfirst += (double)m_Posteriors[0]->GetPixel(ind);
        }
    mu /= sumfirst;
    refMean(ichan) = mu;
  }

  // Compute initial distribution parameters
  if (m_InitialDistributionEstimator.compare("robust") == 0)
    this->ComputeDistributionsRobust();
  else
    this->ComputeDistributions();
  refMean = m_Means.get_column(0);

  // Split distributions with the same prior
  muLogMacro(<< "Splitting distributions with same prior\n");
  for (unsigned int iprior = 0; iprior < numPriors; iprior++)
  {
    if (m_NumberOfGaussians[iprior] > 1)
    {
      if (m_DoMSTClustering)
        this->ClusterFromPriorMST(iprior);
      else
        this->ClusterFromPrior(iprior);
    }
  }

  // Update posteriors using the initial distribution parameters
  this->ComputePosteriors(true);

  // Initialize warping vars
  m_WarpedTemplateImage = m_TemplateImage;
  m_TemplateFluidMomenta = 0;
  m_TemplateFluidVelocity = 0;

  //double logLikelihood = vnl_huge_val(1.0);
  double logLikelihood = 1.0 / vnl_math::eps;
  double deltaLogLikelihood = 1.0;

  unsigned int biasdegree = 0;

  // EM loop
  bool converged = false;
  unsigned int iter = 0;
  while (!converged)
  {

    iter++;

    muLogMacro(<< "\n\nEM iteration " << iter << "\n");
    muLogMacro(<< "---------------------\n");

#if 0
    // Fix the mean for the first class (speeds up convergence)
    // Makes sure that image intensity in the same range after bias correction
    MatrixType offsetMeans(numChannels, numClasses, 0);
    for (unsigned int ichan = 0; ichan < numChannels; ichan++)
    {
      for (unsigned int iclass = 0; iclass < (numClasses-1); iclass++)
        offsetMeans(ichan, iclass) =
          -1.0 * (m_Means(ichan, 0) - refMean(ichan));
    }
    m_Means += offsetMeans;
#endif

    for (unsigned int iclass = 0; iclass < numClasses; iclass++)
    {
      muLogMacro(<< "Class " << (iclass+1) << " mean: ");
      for (unsigned int ichan = 0; ichan < numChannels; ichan++)
        muLogMacro(<< m_Means(ichan, iclass) << "\t");
      muLogMacro(<< "\n");
    }
/*
    muLogMacro(<< "\n");
    for (unsigned int iclass = 0; iclass < numClasses; iclass++)
    {
      muLogMacro(<< "Class " << (iclass+1) << " covariance:\n");
      muLogMacro(<< m_Covariances[iclass]);
    }
*/

    // Update distribution parameters after bias correction
    muLogMacro(<< "  Updating distributions\n");
    this->ComputeDistributions();

    // Recompute posteriors
    muLogMacro(<< "  Updating posteriors\n");
    this->ComputePosteriors(true);

    double prevLogLikelihood = logLikelihood;
    if (prevLogLikelihood == 0)
      prevLogLikelihood = vnl_math::eps;

    // Compute log-likelihood and normalize posteriors
    logLikelihood = this->NormalizePosteriors();

    muLogMacro(<< "log(likelihood) = " << logLikelihood << "\n");

    deltaLogLikelihood =
      fabs((logLikelihood - prevLogLikelihood) / prevLogLikelihood);
      //(logLikelihood - prevLogLikelihood) / fabs(prevLogLikelihood);

    muLogMacro(<< "delta log(likelihood) = " << deltaLogLikelihood << "\n");

    bool dowarp = m_DoWarp && (deltaLogLikelihood < m_WarpLikelihoodTolerance);

    // Bias correction
    if (m_MaxBiasDegree > 0)
    {
      if ((deltaLogLikelihood < m_BiasLikelihoodTolerance)
          &&
          (biasdegree < m_MaxBiasDegree))
      {
        biasdegree++;
      }
/*
      if (dowarp)
        this->CorrectBias(biasdegree, true);
      else
        this->CorrectBias(biasdegree, false);
*/
      // Need to correct all voxels with multithreaded EM
      this->CorrectBias(biasdegree, true);

      // Disable warping if bias correction is enabled but not yet performed
      if (biasdegree < m_MaxBiasDegree)
        dowarp = false;
    }

    if (dowarp)
    {
      this->ComputePosteriors(true); // Update likelihood images before warping
      this->ComputeAtlasWarpingFromProbabilities();
      this->ComputePosteriors(true); // Update posteriors after warping // TODO: separate computelik computepost?
    }

    // Convergence check
    converged =
      (iter >= m_MaximumIterations)
// Ignore jumps in the log likelihood
//    ||
//    (deltaLogLikelihood < 0)
      ||
      ((deltaLogLikelihood < m_LikelihoodTolerance)
        &&
        (biasdegree == m_MaxBiasDegree));

  } // end EM loop

  muLogMacro(<< "Done computing posteriors with " << iter << " iterations\n");

  // Bias correction at full resolution, still using downsampled images
  // for computing the bias field coeficients
  if (m_MaxBiasDegree > 0)
  {
    this->CorrectBias(biasdegree, true);
    this->ComputeDistributions();
  }

  // Recompute posteriors at full resolution
  this->ComputePosteriors(true);

  this->NormalizePosteriors();

}

// Labeling using maximum a posteriori, also do brain stripping using
// mathematical morphology and connected component
template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::ComputeLabels()
{

  itkDebugMacro(<< "ComputeLabels");

  unsigned int numPriors = m_Priors.GetSize();

  unsigned int numClasses = 0;
  for (unsigned int i = 0; i < numPriors; i++)
    numClasses += m_NumberOfGaussians[i];

  unsigned int numFGClasses = numClasses - m_NumberOfGaussians[numPriors-1];

  InputImageRegionType region = m_InputImages[0]->GetLargestPossibleRegion();

  m_Labels = ByteImageType::New();
  m_Labels->CopyInformation(m_InputImages[0]);
  m_Labels->SetRegions(region);
  m_Labels->Allocate();
  m_Labels->FillBuffer(0);

  ByteImagePointer mask = ByteImageType::New();
  mask->CopyInformation(m_InputImages[0]);
  mask->SetRegions(region);
  mask->Allocate();
  mask->FillBuffer(0);

  MersenneTwisterRNG rng;

  ByteImageIndexType ind;
  ByteImageSizeType size = m_Labels->GetLargestPossibleRegion().GetSize();

  for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]++)
    for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]++)
      for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]++)
      {
        if (m_Mask->GetPixel(ind) == 0)
          continue;

        double maxv = m_Posteriors[0]->GetPixel(ind);
        unsigned int imax = 0;

        for (unsigned int iclass = 1; iclass < numClasses; iclass++)
        {
          double v = m_Posteriors[iclass]->GetPixel(ind);
          if (v > maxv)
          {
            maxv = v;
            imax = iclass;
          }
        }

//PP
/*
        // Choose label at random if there's a tie
        // Only for brain classes
        DynArray<unsigned int> tieIndices;
        tieIndices.Allocate(numFGClasses);
        tieIndices.Append(imax);

        for (unsigned int iclass = 0; iclass < numFGClasses; iclass++)
        {
          if (imax == iclass)
            continue;
          double v = m_Posteriors[iclass]->GetPixel(ind);
          if (vnl_math_abs(v - maxv) < 1e-20)
            tieIndices.Append(iclass);
        }

        if (tieIndices.GetSize() > 1)
        {
          unsigned int whichIndex =
            (unsigned int)
              rng.GenerateUniformIntegerUpToK(tieIndices.GetSize() - 1);
          imax = tieIndices[whichIndex];
          maxv = m_Posteriors[imax]->GetPixel(ind);
        }
*/

        unsigned char label = 0;
        unsigned char fgflag = 0;

        // Only use non-zero probabilities and foreground classes
        if (maxv > 0 && imax < numFGClasses)
        {
          label = (unsigned char)(imax+1);
          fgflag = 1;
        }

        m_Labels->SetPixel(ind, label);
        mask->SetPixel(ind, fgflag);

      }

  // Binary opening
  typedef itk::BinaryBallStructuringElement<unsigned char, 3> StructElementType;
  typedef
    itk::BinaryDilateImageFilter<ByteImageType, ByteImageType,
      StructElementType> DilateType;
  typedef
    itk::BinaryErodeImageFilter<ByteImageType, ByteImageType,
      StructElementType> ErodeType;

  StructElementType structel;
  structel.SetRadius(3);
  structel.CreateStructuringElement();

  typename ErodeType::Pointer erode = ErodeType::New();
  erode->SetErodeValue(1);
  erode->SetInput(mask);
  erode->SetKernel(structel);

  erode->Update();

  typename DilateType::Pointer dil = DilateType::New();
  dil->SetDilateValue(1);
  dil->SetInput(erode->GetOutput());
  dil->SetKernel(structel);

  dil->Update();

  // Find largest cluster
  typedef itk::Image<unsigned short, 3> ShortImageType;
  typedef ShortImageType::Pointer ShortImagePointer;

  typedef ConnectedComponentsFilter<ByteImageType, ShortImageType> CCType;
  typename CCType::Pointer ccfilter = CCType::New();

  ccfilter->SetInput(dil->GetOutput());
  ccfilter->Update();

  ShortImagePointer components = ccfilter->GetOutput();

  // Remove floating clusters
  for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]++)
    for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]++)
      for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]++)
      {
        if (components->GetPixel(ind) != 1)
          m_Labels->SetPixel(ind, 0);
      }

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::CleanUp()
{

  itkDebugMacro(<< "CleanUp");

  unsigned int numChannels = m_InputImages.GetSize();

  unsigned int numPriors = m_Priors.GetSize();

  unsigned int numClasses = 0;
  for (unsigned int i = 0; i < numPriors; i++)
    numClasses += m_NumberOfGaussians[i];

  unsigned int numFGClasses = numClasses - m_NumberOfGaussians[numPriors-1];

  InputImageIndexType ind;

  InputImageSizeType size =
    m_InputImages[0]->GetLargestPossibleRegion().GetSize();

  // Make sure all output corrected image intensities are within a "nice" range
  for (unsigned int ichan = 0; ichan < numChannels; ichan++)
  {
    InputImagePointer img = m_CorrectedImages[ichan];

    double minv = vnl_huge_val(1.0f);
    double maxv = -vnl_huge_val(1.0f);
    for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]++)
      for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]++)
        for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]++)
        {
          if (m_Labels->GetPixel(ind) == 0)
            continue;

          double v = img->GetPixel(ind);
          if (v < minv)
            minv = v;
          if (v > maxv)
            maxv = v;
        }

    double range = maxv - minv;

    double outmax = itk::NumericTraits<short>::max();

    // Renormalize to [0, outmax]
    for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]++)
      for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]++)
        for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]++)
        {
          double v = img->GetPixel(ind);
          if (v < minv)
            v = minv;
          if (v > maxv)
            v = maxv;
          v = (v - minv) / range * outmax;
          v = floor(v + 0.5);
          img->SetPixel(ind, v);
        }
  }

  // Zero the foreground posteriors with label 0
  for (unsigned int iclass = 0; iclass < numFGClasses; iclass++)
  {
    ProbabilityImagePointer post = m_Posteriors[iclass];

    for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]++)
      for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]++)
        for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]++)
        {
          if (m_Labels->GetPixel(ind) == 0)
            post->SetPixel(ind, 0);
        }
  }

  this->NormalizePosteriors();

}

template <class TInputImage, class TProbabilityImage>
double
EMSegmentationFilter <TInputImage, TProbabilityImage>
::NormalizePosteriors()
{
  itkDebugMacro(<< "NormalizePosteriors");

  typedef itk::AddImageFilter<ProbabilityImageType, ProbabilityImageType, ProbabilityImageType>
    AddFilterType;

  unsigned int numClasses = m_Posteriors.GetSize();

  ProbabilityImagePointer sumPImg = ProbabilityImageType::New();
  sumPImg->CopyInformation(m_Posteriors[0]);
  sumPImg->SetRegions(m_Posteriors[0]->GetLargestPossibleRegion());
  sumPImg->Allocate();
  sumPImg->FillBuffer(0);

  for (unsigned int iclass = 0; iclass < numClasses; iclass++)
  {
    typename AddFilterType::Pointer addf = AddFilterType::New();
    addf->SetInput1(sumPImg);
    addf->SetInput2(m_Posteriors[iclass]);
    addf->Update();

    sumPImg = addf->GetOutput();
  }

  //typedef itk::LogImageFilter<ProbabilityImageType, ProbabilityImageType> LogFilterType;
  typedef itk::TsallisLogImageFilter<ProbabilityImageType, ProbabilityImageType> LogFilterType;
  typename LogFilterType::Pointer logf = LogFilterType::New();
  logf->SetInput(sumPImg);
  logf->Update();

  typedef itk::StatisticsImageFilter<ProbabilityImageType> StatFilterType;
  typename StatFilterType::Pointer statf = StatFilterType::New();
  statf->SetInput(logf->GetOutput());
  statf->Update();

  double logL = statf->GetSum();

  typename AddFilterType::Pointer epsf = AddFilterType::New();
  epsf->SetInput1(sumPImg);
  epsf->SetConstant2(1e-20);
  epsf->Update();
  sumPImg = epsf->GetOutput();

  for (unsigned int iclass = 0; iclass < numClasses; iclass++)
  {
    typedef itk::DivideImageFilter<ProbabilityImageType, ProbabilityImageType, ProbabilityImageType>
      DivFilterType;
    typename DivFilterType::Pointer divf = DivFilterType::New();
    divf->SetInput1(m_Posteriors[iclass]);
    divf->SetInput2(sumPImg);
    divf->Update();

    m_Posteriors[iclass] = divf->GetOutput();
  }

  return logL;
}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::ComputeAtlasWarpingFromProbabilities()
{
  itkDebugMacro(<< "ComputeAtlasWarpingFromProbabilities");

  muLogMacro(<< "  Computing fluid warping with " << m_WarpFluidIterations
    << " iterations\n");

  if (m_WarpFluidIterations == 0)
    return;

  unsigned int numPriors = m_OriginalPriors.GetSize();
  unsigned int numClasses = m_Posteriors.GetSize();

  // Merge likelihoods
  DynArray<ProbabilityImagePointer> mergedLikelihoods;
  for (unsigned int iprior = 0; iprior < numPriors; iprior++)
  {
    ProbabilityImagePointer tmp = ProbabilityImageType::New();
    tmp->CopyInformation(m_Priors[0]);
    tmp->SetRegions(m_Priors[0]->GetLargestPossibleRegion());
    tmp->Allocate();
    tmp->FillBuffer(0);

    mergedLikelihoods.Append(tmp);
  }
  for (unsigned int iclass = 0; iclass < numClasses; iclass++)
  {
    unsigned int iprior = m_PriorLookupTable[iclass];

    typedef itk::AddImageFilter<ProbabilityImageType, ProbabilityImageType, ProbabilityImageType>
      AddFilterType;
    typename AddFilterType::Pointer addf = AddFilterType::New();
    addf->SetInput1(mergedLikelihoods[iprior]);
    addf->SetInput2(m_Likelihoods[iclass]);
    addf->Update();

    mergedLikelihoods[iprior] = addf->GetOutput();
  }

  typedef MaxLikelihoodFluidWarpEstimator<ProbabilityImagePixelType, 3>
    FluidWarperType;
  typename FluidWarperType::Pointer fluid = FluidWarperType::New();

  fluid->SetLikelihoodImages(mergedLikelihoods);
  fluid->SetPriorImages(m_Priors);

  fluid->SetInitialMomenta(m_TemplateFluidMomenta);
  //fluid->SetMask(m_OriginalMask);
  fluid->SetIterations(m_WarpFluidIterations);
  fluid->SetMaxStep(m_WarpFluidMaxStep);
  fluid->SetKernelWidth(m_WarpFluidKernelWidth);
  fluid->SetRegularityWeight(1e-12);
  fluid->SetNumberOfScales(1);
  fluid->Update();

  m_TemplateFluidMomenta = fluid->GetMomenta();
  m_TemplateFluidVelocity = fluid->GetVelocity();

  m_Priors = fluid->GetWarpedPriorImages();

  // Update mask? Not needed with large dilation of initial mask
  this->ComputeMask();

  // Warp the template image
  typedef itk::WarpImageFilter<
    InputImageType, InputImageType, VectorFieldType>
    WarperType;
  typename WarperType::Pointer warpf = WarperType::New();
  warpf->SetInput(m_TemplateImage);
  warpf->SetDisplacementField(m_TemplateFluidVelocity);
  warpf->SetOutputDirection(m_TemplateImage->GetDirection());
  warpf->SetOutputOrigin(m_TemplateImage->GetOrigin());
  warpf->SetOutputSpacing(m_TemplateImage->GetSpacing());
  warpf->Update();

  m_WarpedTemplateImage = warpf->GetOutput();

}

template <class TInputImage, class TProbabilityImage>
void
EMSegmentationFilter <TInputImage, TProbabilityImage>
::ComputeAtlasWarpingFromIntensities()
{
  itkDebugMacro(<< "ComputeAtlasWarpingFromIntensities");

  muLogMacro(<< "Computing fluid warping with " << m_WarpFluidIterations
    << " iterations\n");

  if (m_WarpFluidIterations == 0)
    return;

  unsigned int numPriors = m_OriginalPriors.GetSize();
  unsigned int numClasses = m_Posteriors.GetSize();

  typedef SimpleGreedyFluidRegistration<InputImagePixelType, 3>
    FluidWarperType;
  typename FluidWarperType::Pointer fluid = FluidWarperType::New();

  // Make subject 1st channel intensities similar to template
  typedef IntensityMatcher<InputImageType, ProbabilityImageType> MatcherType;
  typename MatcherType::Pointer matcher = MatcherType::New();
  matcher->SetSourceImage(m_WarpedTemplateImage);
  matcher->SetTargetImage(m_CorrectedImages[0]);
  matcher->SetProbabilities(m_Posteriors);
  matcher->Update();

  InputImagePointer matchImg = matcher->GetOutput();

  DynArray<InputImagePointer> fixedSet;
  fixedSet.Append(m_CorrectedImages[0]);
  fluid->SetFixedImages(fixedSet);
  DynArray<InputImagePointer> movingSet;
  movingSet.Append(matchImg);
  fluid->SetMovingImages(movingSet);

  fluid->SetInitialDisplacementField(m_TemplateFluidVelocity);
  //fluid->SetMask(m_OriginalMask);
  fluid->SetIterations(m_WarpFluidIterations);
  fluid->SetMaxStep(m_WarpFluidMaxStep);
  fluid->SetKernelWidth(m_WarpFluidKernelWidth);
  fluid->SetNumberOfScales(2);
  fluid->Update();

  m_TemplateFluidVelocity = fluid->GetDisplacementField();

  m_Priors.Clear();
  for (unsigned int i = 0; i < numPriors; i++)
  {
    typedef itk::WarpImageFilter<
      ProbabilityImageType, ProbabilityImageType, VectorFieldType>
      WarperType;
    typename WarperType::Pointer warpf = WarperType::New();
    warpf->SetInput(m_OriginalPriors[i]);
    if (i < (numPriors-1))
      warpf->SetEdgePaddingValue(0.0);
    else
      warpf->SetEdgePaddingValue(1.0);
    warpf->SetDisplacementField(m_TemplateFluidVelocity);
    warpf->SetOutputDirection(m_OriginalPriors[i]->GetDirection());
    warpf->SetOutputOrigin(m_OriginalPriors[i]->GetOrigin());
    warpf->SetOutputSpacing(m_OriginalPriors[i]->GetSpacing());
    warpf->Update();
    m_Priors.Append(warpf->GetOutput());
  }

  // Update mask? Not needed with large dilation of initial mask
  this->ComputeMask();

  // Warp the template image
  typedef itk::WarpImageFilter<
    InputImageType, InputImageType, VectorFieldType>
    WarperType;
  typename WarperType::Pointer warpf = WarperType::New();
  warpf->SetInput(m_TemplateImage);
  warpf->SetDisplacementField(m_TemplateFluidVelocity);
  warpf->SetOutputDirection(m_TemplateImage->GetDirection());
  warpf->SetOutputOrigin(m_TemplateImage->GetOrigin());
  warpf->SetOutputSpacing(m_TemplateImage->GetSpacing());
  warpf->Update();

  m_WarpedTemplateImage = warpf->GetOutput();

}

#endif
