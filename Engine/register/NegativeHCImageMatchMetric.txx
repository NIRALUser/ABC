
#ifndef _NegativeHCImageMatchMetric_txx
#define _NegativeHCImageMatchMetric_txx

#include "itkContinuousIndex.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkNumericTraits.h"

#include "vnl/vnl_math.h"

#include "NegativeHCImageMatchMetric.h"

#include "MersenneTwisterRNG.h"

#include <cfloat>
#include <cmath>


// Image to histogram index mapping using linear mapping
template <class TImage, class TIndexImage>
typename itk::SmartPointer<TIndexImage>
_hc_linearMapIntensityToHistogramIndex(
  const TImage* img, unsigned int numBins, double sampleSpacing)
{
  typename TImage::SizeType size = img->GetLargestPossibleRegion().GetSize();
  typename TImage::SpacingType spacing = img->GetSpacing();

  typename TImage::OffsetType skips; 
  skips[0] = (unsigned)floor(sampleSpacing / spacing[0]);
  skips[1] = (unsigned)floor(sampleSpacing / spacing[1]);
  skips[2] = (unsigned)floor(sampleSpacing / spacing[2]);
  
  if (skips[0] == 0)
    skips[0] = 1;
  if (skips[1] == 0) 
    skips[1] = 1; 
  if (skips[2] == 0)
    skips[2] = 1;

  double minv = vnl_huge_val(1.0);
  double maxv = -vnl_huge_val(1.0);

  typename TImage::IndexType ind;
  for (ind[2] = 0; ind[2] < (long)size[2]; ind[2] += skips[2])
    for (ind[1] = 0; ind[1] < (long)size[1]; ind[1] += skips[1])
      for (ind[0] = 0; ind[0] < (long)size[0]; ind[0] += skips[0])
      {
        double v = img->GetPixel(ind);
        if (v < minv)
          minv = v;
        if (v > maxv)
          maxv = v;
      }

  double rangev = maxv - minv;

  double t = 0.005 * rangev;
  minv += t;
  maxv -= t;
  rangev -= 2*t;

  // Allocate index image
  typename itk::SmartPointer<TIndexImage> mapImg = TIndexImage::New();
  mapImg->CopyInformation(img);
  mapImg->SetRegions(img->GetLargestPossibleRegion());
  mapImg->Allocate();

  // Map whole image to histogram index
  typedef itk::ImageRegionConstIterator<TImage> ImageIteratorType;
  ImageIteratorType it(img, img->GetLargestPossibleRegion());

  typedef itk::ImageRegionIterator<TIndexImage> IndexIteratorType;
  IndexIteratorType mapIt(mapImg, img->GetLargestPossibleRegion());

  it.GoToBegin();
  mapIt.GoToBegin();

  for (; !it.IsAtEnd(); ++it, ++mapIt)
  {
    double v = it.Get();

    double u = (v - minv) / rangev;

    unsigned int map = 0;

#if 0
    // Exclude extreme intensities
    if (u < 0.0 || u > 1.0)
      map = numBins+1;
    else
      map = static_cast<unsigned int>(u * (numBins-1));
#else
    // Clamp extremes
    if (u < 0.0)
      u = 0.0;
    if (u > 1.0)
      u = 1.0;
    map = static_cast<unsigned int>(u * (numBins-1));
#endif

    mapIt.Set(map);
  }

  return mapImg;
}

////////////////////////////////////////////////////////////////////////////////

template <class TFixedImage, class TMovingImage>
NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
::NegativeHCImageMatchMetric()
{
  m_NumberOfBins = 255;

  m_HistogramPointer = new HistogramType(m_NumberOfBins, m_NumberOfBins);
  m_HistogramPointer->fill(0);

  this->m_FixedImage = 0;
  this->m_MovingImage = 0;

  m_FixedIndexImage = 0;
  m_MovingIndexImage = 0;

  m_SampleSpacing = 4.0;

  m_Skips[0] = 1;
  m_Skips[1] = 1;
  m_Skips[2] = 1;

  m_Alpha = 1.1;

  m_DerivativeStepLengths = ParametersType(1);
  m_DerivativeStepLengths.Fill(1e-2);
}

template <class TFixedImage, class TMovingImage>
void
NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "NumberOfBins: ";
  os << m_NumberOfBins << std::endl;
}

template <class TFixedImage, class TMovingImage>
void
NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
::SetFixedImage(
  const typename NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
    ::FixedImageType* img)
{

  itkDebugMacro(<< "SetFixedImage");

  if (img->GetImageDimension() != 3)
    itkExceptionMacro(<< "Fixed image dimension invalid: only supports 3D");

  if (this->m_FixedImage != img)
  {
    this->m_FixedImage = img;
    this->Modified();
  }

  FixedImageSizeType size =
    this->m_FixedImage->GetLargestPossibleRegion().GetSize();
  FixedImageSpacingType spacing = this->m_FixedImage->GetSpacing();

  // Compute skips for downsampling
  m_Skips[0] = (unsigned int)(m_SampleSpacing / spacing[0]);
  m_Skips[1] = (unsigned int)(m_SampleSpacing / spacing[1]);
  m_Skips[2] = (unsigned int)(m_SampleSpacing / spacing[2]);

  if (m_Skips[0] == 0)
    m_Skips[0] = 1;
  if (m_Skips[1] == 0)
    m_Skips[1] = 1;
  if (m_Skips[2] == 0)
    m_Skips[2] = 1;

  this->MapFixedImage();

}

template <class TFixedImage, class TMovingImage>
void
NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
::SetSampleSpacing(double s)
{
  m_SampleSpacing = s;

  if (!this->m_FixedImage.IsNull())
  {
    FixedImageSpacingType spacing = this->m_FixedImage->GetSpacing();

    // Compute skips for downsampling
    m_Skips[0] = (unsigned int)(m_SampleSpacing / spacing[0]);
    m_Skips[1] = (unsigned int)(m_SampleSpacing / spacing[1]);
    m_Skips[2] = (unsigned int)(m_SampleSpacing / spacing[2]);

    if (m_Skips[0] == 0)
      m_Skips[0] = 1;
    if (m_Skips[1] == 0)
      m_Skips[1] = 1;
    if (m_Skips[2] == 0)
      m_Skips[2] = 1;
  }
}

template <class TFixedImage, class TMovingImage>
void
NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
::SetMovingImage(
  const typename NegativeHCImageMatchMetric<TFixedImage,TMovingImage>
  ::MovingImageType* img)
{

  itkDebugMacro(<< "SetMovingImage");

  if (img->GetImageDimension() != 3)
    itkExceptionMacro(<< "Moving image dimension invalid: only supports 3D");

  if (this->m_MovingImage != img)
  {
    this->m_MovingImage = img;
    this->Modified();
  }

  MovingImageSizeType size =
    this->m_MovingImage->GetLargestPossibleRegion().GetSize();
  MovingImageSpacingType spacing = this->m_MovingImage->GetSpacing();

  this->MapMovingImage();

}

template <class TFixedImage, class TMovingImage>
void
NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
::MapFixedImage()
{
  itkDebugMacro(<< "MapFixedImage");

  if (this->m_FixedImage.IsNull())
    return;

  m_FixedIndexImage =
    _hc_linearMapIntensityToHistogramIndex<FixedImageType, IndexImageType>(
      this->m_FixedImage, m_NumberOfBins, m_SampleSpacing);
}

template <class TFixedImage, class TMovingImage>
void
NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
::MapMovingImage()
{
  itkDebugMacro(<< "MapMovingImage");

  if (this->m_MovingImage.IsNull())
    return;

  m_MovingIndexImage =
    _hc_linearMapIntensityToHistogramIndex<MovingImageType, IndexImageType>(
      this->m_MovingImage, m_NumberOfBins, m_SampleSpacing);
}

template <class TFixedImage, class TMovingImage>
void
NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
::SetNumberOfBins(unsigned int n)
{

  // Clamp to minimum of 2
  if (n < 2)
  {
    itkWarningMacro(<< "Clamping number of bins to " << 2);
    n = 2;
  }

  unsigned int maxbins = itk::NumericTraits<unsigned int>::max() - 1;

  if (n > maxbins)
  {
    itkWarningMacro(<< "Clamping number of bins to " << maxbins);
    n = maxbins;
  }

  m_NumberOfBins = n;

  delete m_HistogramPointer;
  m_HistogramPointer = new HistogramType(m_NumberOfBins, m_NumberOfBins);
  m_HistogramPointer->fill(0);

  this->MapFixedImage();
  this->MapMovingImage();

  this->Modified();
}

template <class TFixedImage, class TMovingImage>
void
NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
::ComputeHistogram() const
{

  itkDebugMacro(<< "ComputeHistogram");

  HistogramType& H = *m_HistogramPointer;
  H.fill(0);

  FixedImagePointType fixedOrigin = m_FixedIndexImage->GetOrigin();

  FixedImageSpacingType fixedSpacing = m_FixedIndexImage->GetSpacing();

  FixedImageSizeType fixedSize =
    m_FixedIndexImage->GetLargestPossibleRegion().GetSize();

  MovingImagePointType movingOrigin = m_MovingIndexImage->GetOrigin();

  MovingImageSpacingType movingSpacing = m_MovingIndexImage->GetSpacing();

  MovingImageSizeType movingSize =
    m_MovingIndexImage->GetLargestPossibleRegion().GetSize();

  FixedImageIndexType ind;

  for (ind[2] = 0; ind[2] < (long)fixedSize[2]; ind[2] += m_Skips[2])
    for (ind[1] = 0; ind[1] < (long)fixedSize[1]; ind[1] += m_Skips[1])
      for (ind[0] = 0; ind[0] < (long)fixedSize[0]; ind[0] += m_Skips[0])
      {
        // Get sampled fixed image histogram index
        unsigned int r = m_FixedIndexImage->GetPixel(ind);

        // Skip if fixed image histogram index is invalid
        if (r >= m_NumberOfBins)
        {
          continue;
        }

        FixedImagePointType fixedPoint;
        this->m_FixedImage->TransformIndexToPhysicalPoint(ind, fixedPoint);

        MovingImagePointType mappedPoint =
          this->m_Transform->TransformPoint(fixedPoint);

        // Use Partial Volume interpolation
    
        // Get continuous moving image coordinates (in voxels)
        typedef itk::ContinuousIndex<double, 3> ContinuousIndexType;
        ContinuousIndexType movingInd;
        this->m_MovingImage->TransformPhysicalPointToContinuousIndex(
          mappedPoint, movingInd);

        // Get image neighborhood
        int x0 = (int)movingInd[0];
        int y0 = (int)movingInd[1];
        int z0 = (int)movingInd[2];

        int x1 = x0 + 1;
        int y1 = y0 + 1;
        int z1 = z0 + 1;

        // Get distances to the image grid
        double fx = movingInd[0] - (double)x0;
        double fy = movingInd[1] - (double)y0;
        double fz = movingInd[2] - (double)z0;

        double gx = 1.0 - fx;
        double gy = 1.0 - fy;
        double gz = 1.0 - fz;

        // Moving image histogram index (column)
        unsigned int c = 0;

/*
// Linear interpolation
// Note: Do not use linear interp with non-uniform spaced bins
// Need to account for hist spacing h += area * 1
        double c_interp = 0;

#define interpWeightMacro(x, y, z, w) \
  if ((0 <= (x)) && ((x) < (long)movingSize[0]) && \
    (0 <= (y)) && ((y) < (long)movingSize[1]) && \
    (0 <= (z)) && ((z) < (long)movingSize[2])) \
  { \
    MovingImageIndexType local_ind = {{(x), (y), (z)}}; \
    c_interp += (w) * m_MovingIndexImage->GetPixel(local_ind); \
  }
        interpWeightMacro(x0, y0, z0, gx*gy*gz);
        interpWeightMacro(x0, y0, z1, gx*gy*fz);
        interpWeightMacro(x0, y1, z0, gx*fy*gz);
        interpWeightMacro(x0, y1, z1, gx*fy*fz);
        interpWeightMacro(x1, y0, z0, fx*gy*gz);
        interpWeightMacro(x1, y0, z1, fx*gy*fz);
        interpWeightMacro(x1, y1, z0, fx*fy*gz);
        interpWeightMacro(x1, y1, z1, fx*fy*fz);

#undef interpWeightMacro

        c = (unsigned int)(c_interp + 0.5);
        if (c >= m_NumberOfBins)
          continue;

        H(r, c) += 1.0;
*/

// PV interpolation
// Macro for adding trilinear weights
// Only add if inside moving image and moving index is valid
#define partialVolumeWeightMacro(x, y, z, w) \
  if ((0 <= (x)) && ((x) < (long)movingSize[0]) && \
    (0 <= (y)) && ((y) < (long)movingSize[1]) && \
    (0 <= (z)) && ((z) < (long)movingSize[2])) \
  { \
    MovingImageIndexType pvind = {{(x), (y), (z)}}; \
    c = m_MovingIndexImage->GetPixel(pvind); \
    if (c < m_NumberOfBins) \
      H(r, c) += (w); \
  }

        // Fill histogram with trilinear weights
        partialVolumeWeightMacro(x0, y0, z0, gx*gy*gz);
        partialVolumeWeightMacro(x0, y0, z1, gx*gy*fz);
        partialVolumeWeightMacro(x0, y1, z0, gx*fy*gz);
        partialVolumeWeightMacro(x0, y1, z1, gx*fy*fz);
        partialVolumeWeightMacro(x1, y0, z0, fx*gy*gz);
        partialVolumeWeightMacro(x1, y0, z1, fx*gy*fz);
        partialVolumeWeightMacro(x1, y1, z0, fx*fy*gz);
        partialVolumeWeightMacro(x1, y1, z1, fx*fy*fz);

#undef partialVolumeWeightMacro

      }

  // Normalize histogram values
  double sumHist = 0;
  for (unsigned int r = 0; r < m_NumberOfBins; r++)
    for (unsigned int c = 0; c < m_NumberOfBins; c++)
      sumHist += H(r, c);
  if (sumHist != 0)
    H /= sumHist;

}

template <class TFixedImage, class TMovingImage>
double
NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
::ComputeHC() const
{
  // Compute histogram
  this->ComputeHistogram();

  itkDebugMacro(<< "Start MI");

  HistogramType& H = *m_HistogramPointer;

  HistogramType marginalA(m_NumberOfBins, 1, 0.0);
  HistogramType marginalB(m_NumberOfBins, 1, 0.0);

  for (unsigned int i = 0; i < m_NumberOfBins; i++)
  {
    for (unsigned int j = 0; j < m_NumberOfBins; j++)
    {
      marginalA(i, 0) += H(i, j);
      marginalB(i, 0) += H(j, i);
    }
  }

#if 0

  double ent1 = 0.0;
  double ent2 = 0.0;
  for (unsigned int i = 0; i < m_NumberOfBins; i++)
  {
    ent1 += pow(marginalA(i, 0), m_Alpha);
    ent2 += pow(marginalB(i, 0), m_Alpha);
  }
  ent1 = ent1 - 1.0;
  ent2 = ent2 - 1.0;

  double jointEnt = 0.0;
  for (unsigned int i = 0; i < m_NumberOfBins; i++)
    for (unsigned int j = 0; j < m_NumberOfBins; j++)
    {
      double p = H(i, j);
      if (p <= 0.0)
        continue;
      jointEnt += pow(p, m_Alpha);
    }
  jointEnt = jointEnt - 1.0;

  double info = 0.0;
  if (m_Alpha != 1.0)
  {
    //info = ent1 + ent2 + (1.0-m_Alpha)*ent1*ent2 - jointEnt;
    info = ent1 + ent2 - jointEnt;
    info /= (1.0-m_Alpha);
  }
  else
  {
//TODO: use log if alpha = 1
    info = ent1 + ent2 - jointEnt;
  }

#else

  // Use ratios (Pluim)
  double info = 0.0;
  for (unsigned int i = 0; i < m_NumberOfBins; i++)
  {
    for (unsigned int j = 0; j < m_NumberOfBins; j++)
    {
      double p = H(i, j);
      if (p <= 0.0)
        continue;
      double prodMarginals = marginalA(i, 0) * marginalB(j, 0);
      if (prodMarginals <= 0.0)
        continue;
      info += pow(p, m_Alpha) / pow(prodMarginals, m_Alpha-1.0);
    }
  }

  info /= m_Alpha*(m_Alpha-1.0);

#endif

  return info;

}

template <class TFixedImage, class TMovingImage>
typename NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
::MeasureType
NegativeHCImageMatchMetric<TFixedImage,TMovingImage>
::GetValue(const ParametersType& parameters) const
{
  // Make sure the transform has the current parameters
  this->m_Transform->SetParameters(parameters);

  return -1.0*this->ComputeHC();
}

template <class TFixedImage, class TMovingImage>
void
NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
::GetValueAndDerivative(const ParametersType& parameters, MeasureType& value,
  DerivativeType& derivative) const
{
  value = this->GetValue(parameters);
  //this->GetDerivative(parameters, derivative);
  this->GetStochasticDerivative(parameters, derivative);
}

template <class TFixedImage, class TMovingImage>
void
NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
::GetDerivative(const ParametersType& parameters, DerivativeType & derivative) const
{
  unsigned int numParams = this->m_Transform->GetNumberOfParameters();

  if (m_DerivativeStepLengths.GetSize() != numParams)
  {
    itkExceptionMacro(<< "Derivative step lengths not set");
  }

  derivative = DerivativeType(numParams);
  derivative.Fill(0);

  for (unsigned int i = 0; i < numParams; i++)
  {
    ParametersType p1 = parameters;
    p1[i] -= m_DerivativeStepLengths[i];
    double v1 = this->GetValue(p1);

    ParametersType p2 = parameters;
    p2[i] += m_DerivativeStepLengths[i];
    double v2 = this->GetValue(p2);

    derivative[i] = (v2 - v1) / (2.0*m_DerivativeStepLengths[i]);
  }
}

// Compute derivative following SPSA (Spall)
template <class TFixedImage, class TMovingImage>
void
NegativeHCImageMatchMetric<TFixedImage, TMovingImage>
::GetStochasticDerivative(const ParametersType& parameters, DerivativeType & derivative) const
{

  unsigned int numParams = this->m_Transform->GetNumberOfParameters();

  if (m_DerivativeStepLengths.GetSize() != numParams)
  {
    itkExceptionMacro(<< "Derivative step lengths not set");
  }

  derivative = DerivativeType(numParams);
  derivative.Fill(0);

  ParametersType dp = ParametersType(numParams);

  MersenneTwisterRNG* rng = MersenneTwisterRNG::GetGlobalInstance();

  for (unsigned int i = 0; i < numParams; i++)
  {
    // Flip forward/backward only
    double r = rng->GenerateUniformRealClosedInterval();
    if (r >= 0.5)
      dp[i] = m_DerivativeStepLengths[i];
    else
      dp[i] = -m_DerivativeStepLengths[i];

/*
    // r in [-1, 1]
    double r = 2.0*rng->GenerateUniformRealClosedInterval() - 1.0;
    dp[i] = r*m_DerivativeStepLengths[i];
    if (fabs(dp[i]) < 1e-10)
      dp[i] = 1e-10;
*/

  }

  ParametersType p1(numParams);
  ParametersType p2(numParams);
  for (unsigned int i = 0; i < numParams; i++)
  {
    p1[i] = parameters[i] + dp[i];
    p2[i] = parameters[i] - dp[i];
  }

  double v1 = this->GetValue(p1);
  double v2 = this->GetValue(p2);

  double v_diff = v1 - v2;

  for (unsigned int i = 0; i < numParams; i++)
  {
    derivative[i] = v_diff / (2.0*dp[i]);
  }
}

#endif
