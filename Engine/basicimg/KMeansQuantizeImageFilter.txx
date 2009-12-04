
#ifndef _KMeansQuantizeImageFilter_txx
#define _KMeansQuantizeImageFilter_txx

#include "itkImageRegionIterator.h"

#include "KMeansQuantizeImageFilter.h"

#include "MersenneTwisterRNG.h"
#include "KMeansEstimator.h"

#include "vnl/vnl_math.h"

#include <algorithm>
#include <cmath>

template <class TInputImage, class TOutputImage>
KMeansQuantizeImageFilter<TInputImage, TOutputImage>
::KMeansQuantizeImageFilter()
{
  m_NumberOfBins = 100;

  m_MaximumSamples = 500000;

  m_TrimAbove = false;
  m_TrimBelow = false;

  m_TrimFraction = 0.001;

  m_TrimAboveValue = m_NumberOfBins+1;
  m_TrimBelowValue = 0;
}

template <class TInputImage, class TOutputImage>
void
KMeansQuantizeImageFilter<TInputImage, TOutputImage>
::GenerateData()
{
  InputImageRegionType region = this->GetInput()->GetLargestPossibleRegion();

  InputImageSizeType size = region.GetSize();

  this->GetOutput()->CopyInformation(this->GetInput());
  this->GetOutput()->SetRegions(region);
  this->GetOutput()->Allocate();
  this->GetOutput()->FillBuffer(0);

  // Get intensity range
  typedef itk::ImageRegionConstIterator<InputImageType> InputIteratorType;
  InputIteratorType inputIt(this->GetInput(), region);

  double minv = vnl_huge_val(1.0);
  double maxv = -vnl_huge_val(1.0);

  for (inputIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt)
  {
    double v = inputIt.Get();
    if (v < minv)
      minv = v;
    if (v > maxv)
      maxv = v;
  }

  double rangev = maxv - minv;

  double trimv = m_TrimFraction * rangev;
  minv += trimv;
  maxv -= trimv;
  rangev -= 2*trimv;

  // Get samples
  MersenneTwisterRNG* rng = MersenneTwisterRNG::GetGlobalInstance();

  unsigned long numPossibleSamples = 0;

  for (inputIt.GoToBegin(); !inputIt.IsAtEnd(); ++inputIt)
  {
    double v = inputIt.Get();
    if (m_TrimAbove && (v > maxv))
      continue;
    if (m_TrimBelow && (v < minv))
      continue;
    numPossibleSamples++;
  }

  // Only use at most a fraction of the image pixels to find the intensities
  unsigned long numSamples = numPossibleSamples / 5;
  if (numSamples > m_MaximumSamples)
    numSamples = m_MaximumSamples;

  if (numSamples < m_NumberOfBins)
    muExceptionMacro(<< "Not enough samples to quantize to "
      << m_NumberOfBins << " values");

  typedef itk::Image<unsigned char, ImageDimension> ByteImageType;
  typename ByteImageType::Pointer sampleMask = ByteImageType::New();
  sampleMask->CopyInformation(this->GetInput());
  sampleMask->SetRegions(region);
  sampleMask->Allocate();
  sampleMask->FillBuffer(0);

  KMeansEstimator::MatrixType samples(numSamples, 1);

  unsigned int count = 0;
  while (count < numSamples)
  {
    InputImageIndexType ind;
    for (unsigned int d = 0; d < ImageDimension; d++)
      ind[d] = rng->GenerateUniformIntegerUpToK(size[d]-1);

    if (sampleMask->GetPixel(ind) != 0)
      continue;

    // Mark as visited
    sampleMask->SetPixel(ind, 1);

    double v = this->GetInput()->GetPixel(ind);
    if (m_TrimAbove && (v > maxv))
      continue;
    if (m_TrimBelow && (v < minv))
      continue;

    samples(count++, 0) = v;
  }

  // Do K-means
  KMeansEstimator kmeans;
  kmeans.UseKdTreeOn();
  kmeans.SetMaximumIterations(100);
  kmeans.SetNumberOfClusters(m_NumberOfBins);
  kmeans.SetNumberOfStarts(5);
  kmeans.SetInput(samples);

  KMeansEstimator::MatrixType means = kmeans.GetMeans();

  // Sort the mean values
  std::sort(means.data_block(), means.data_block()+m_NumberOfBins);

  // Do the actual quantization
  typedef itk::ImageRegionIterator<OutputImageType> OutputIteratorType;
  OutputIteratorType outputIt(this->GetOutput(), region);

  inputIt.GoToBegin();
  for (outputIt.GoToBegin(); !outputIt.IsAtEnd(); ++inputIt, ++outputIt)
  {
    double v = inputIt.Get();

    if (m_TrimAbove && (v > maxv))
    {
      outputIt.Set(m_TrimAboveValue);
      continue;
    }
    if (m_TrimBelow && (v < minv))
    {
      outputIt.Set(m_TrimBelowValue);
      continue;
    }

    // Find nearest cluster mean
    double mindist = vnl_huge_val(1.0);
    OutputImagePixelType q = m_NumberOfBins + 1;

// TODO: faster, binary type search???
/*
    long left = 0;
    long right = m_NumberOfBins - 1;

    long best = 0;

    while (true)
    {
      long extent = right - left + 1;
      if (extent < 2)
        break;

      long mid0 = left + extent / 2;
      long mid1 = mid0 + 1;
      double d0 = fabs(v - means(mid0, 0));
      double d1 = fabs(v - means(mid1, 0));

      if (d0 < mindist)
      {
        best = mid0;
        mindist = d0;
      }
      if (d1 < mindist)
      {
        best = mid1;
        mindist = d1;
      }

      if (d0 < d1)
      {
        right = mid0;
      }
      else
      {
        left = mid1;
      }
    }

    q = static_cast<OutputImagePixelType>(best);
    outputIt.Set(q);
*/

    for (unsigned int i = 0; i < m_NumberOfBins; i++)
    {
      double dist = fabs(v - means(i, 0));
      if (dist < mindist)
      {
        mindist = dist;
        q = static_cast<OutputImagePixelType>(i);
      }
    }

    outputIt.Set(q);

  }

}

#endif
