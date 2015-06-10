
#ifndef _NegativeMIImageMatchMetric_txx
#define _NegativeMIImageMatchMetric_txx

#include "itkContinuousIndex.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkNumericTraits.h"

#include "vnl/vnl_math.h"

#include "NegativeMIImageMatchMetric.h"

#include "KMeansQuantizeImageFilter.h"
#include "MersenneTwisterRNG.h"

#include <cfloat>
#include <cmath>


// Image to histogram index mapping using linear mapping
template <class TImage, class TIndexImage>
typename itk::SmartPointer<TIndexImage>
_linearMapIntensityToHistogramIndex(
  const TImage* img, unsigned int numBins, float sampleSpacing)
{
  if (sampleSpacing < 0)
    std::cerr << "Negative sample spacing" << std::endl;

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

  float minv = vnl_huge_val(1.0f);
  float maxv = -vnl_huge_val(1.0f);

  typename TImage::IndexType ind;
  for (ind[2] = 0; ind[2] < (long)size[2]; ind[2] += skips[2])
    for (ind[1] = 0; ind[1] < (long)size[1]; ind[1] += skips[1])
      for (ind[0] = 0; ind[0] < (long)size[0]; ind[0] += skips[0])
      {
        float v = img->GetPixel(ind);
        if (v < minv)
          minv = v;
        if (v > maxv)
          maxv = v;
      }

  float rangev = maxv - minv;

  float t = 0.005 * rangev;
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
    float v = it.Get();

    float u = (v - minv) / rangev;

    unsigned int map = 0;

    if (u < 0.0 || u > 1.0)
      map = numBins+1;
    else
      map = static_cast<unsigned int>(u * (numBins-1));

    mapIt.Set(map);
  }

  return mapImg;
}

// Image to histogram index mapping using K-means clustering
template <class TImage, class TIndexImage>
typename itk::SmartPointer<TIndexImage>
_kMeansMapIntensityToHistogramIndex(
  const TImage* img, unsigned int numBins, float sampleSpacing)
{
  typedef KMeansQuantizeImageFilter<TImage, TIndexImage>
    QuantizerType;

  if (sampleSpacing < 0)
    std::cerr << "Negative sample spacing" << std::endl;

  typename QuantizerType::Pointer qfilter = QuantizerType::New();
  qfilter->SetInput(img);
  qfilter->SetMaximumSamples(500000);
  qfilter->SetNumberOfBins(numBins);
  qfilter->SetTrimFraction(0.001);
  qfilter->TrimAboveOff();
  qfilter->TrimBelowOff();
  qfilter->SetTrimAboveValue(numBins+1);
  qfilter->SetTrimBelowValue(numBins+1);
  qfilter->Update();

  return qfilter->GetOutput();
}

////////////////////////////////////////////////////////////////////////////////

template <class TFixedImage, class TMovingImage>
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
::NegativeMIImageMatchMetric()
{

  m_HistogramPointer = 0;
  m_ThreadHistograms = 0;

  this->m_FixedImage = 0;
  this->m_MovingImage = 0;

  m_FixedIndexImage = 0;
  m_MovingIndexImage = 0;

  m_KMeansSampleSpacing = 4.0;
  m_SampleSpacing = 4.0;

  m_QuantizeFixed = false;
  m_QuantizeMoving = false;

  m_Skips[0] = 1;
  m_Skips[1] = 1;
  m_Skips[2] = 1;

  m_Normalized = false;

  m_RandomSampling = false;

  m_DerivativeStepLengths = ParametersType(1);
  m_DerivativeStepLengths.Fill(1e-2);

  m_ThreadHistograms = 0;

  //m_NumberOfBins = 255;
  this->SetNumberOfBins(255);

  m_ThreadIndexCount = new int;
}

template <class TFixedImage, class TMovingImage>
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
::~NegativeMIImageMatchMetric()
{
  delete m_HistogramPointer;
  delete [] m_ThreadHistograms;
  delete m_ThreadIndexCount;
}

template <class TFixedImage, class TMovingImage>
void
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "NumberOfBins: ";
  os << m_NumberOfBins << std::endl;
}

template <class TFixedImage, class TMovingImage>
void
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
::SetFixedImage(
  const typename NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
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

  m_ThreadIndices.clear();

  FixedImageIndexType ind;
  for (ind[2] = m_Skips[2]; ind[2] < (long)size[2]; ind[2] += m_Skips[2])
    for (ind[1] = m_Skips[1]; ind[1] < (long)size[1]; ind[1] += m_Skips[1])
      for (ind[0] = m_Skips[0]; ind[0] < (long)size[0]; ind[0] += m_Skips[0])
      {
        m_ThreadIndices.push_back(ind);
      }

  // Random selection of indices
  if (m_RandomSampling)
  {
    unsigned int numIndices = m_ThreadIndices.size();

    MersenneTwisterRNG* rng = MersenneTwisterRNG::GetGlobalInstance();

    unsigned int* selection = rng->GenerateIntegerSequence(numIndices/2, numIndices-1);

    std::vector<FixedImageIndexType> selectedIndices;
    for (unsigned int i = 0; i < numIndices/2; i++)
      selectedIndices.push_back(m_ThreadIndices[selection[i]]);

    delete [] selection;

    m_ThreadIndices = selectedIndices;
  }
}

template <class TFixedImage, class TMovingImage>
void
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
::SetSampleSpacing(float s)
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
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
::SetMovingImage(
  const typename NegativeMIImageMatchMetric<TFixedImage,TMovingImage>
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
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
::MapFixedImage()
{
  itkDebugMacro(<< "MapFixedImage");

  if (this->m_FixedImage.IsNull())
    return;

  if (m_QuantizeFixed)
  {
    m_FixedIndexImage =
      _kMeansMapIntensityToHistogramIndex<FixedImageType, IndexImageType>(
        this->m_FixedImage, m_NumberOfBins, m_KMeansSampleSpacing);
  }
  else
  {
    m_FixedIndexImage =
      _linearMapIntensityToHistogramIndex<FixedImageType, IndexImageType>(
        this->m_FixedImage, m_NumberOfBins, m_KMeansSampleSpacing);
  }
}

template <class TFixedImage, class TMovingImage>
void
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
::MapMovingImage()
{
  itkDebugMacro(<< "MapMovingImage");

  if (this->m_MovingImage.IsNull())
    return;

  if (m_QuantizeMoving)
  {
    m_MovingIndexImage =
      _kMeansMapIntensityToHistogramIndex<MovingImageType, IndexImageType>(
        this->m_MovingImage, m_NumberOfBins, m_KMeansSampleSpacing);
  }
  else
  {
    m_MovingIndexImage =
      _linearMapIntensityToHistogramIndex<MovingImageType, IndexImageType>(
        this->m_MovingImage, m_NumberOfBins, m_KMeansSampleSpacing);
  }
}

template <class TFixedImage, class TMovingImage>
void
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
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

  // Initialize thread histograms (use default number of threads)
  int numThreads = itk::MultiThreader::GetGlobalDefaultNumberOfThreads();

  delete [] m_ThreadHistograms;
  m_ThreadHistograms = new HistogramType[numThreads];

  for (unsigned int i = 0; i < numThreads; i++)
  {
    m_ThreadHistograms[i].set_size(m_NumberOfBins, m_NumberOfBins);
    m_ThreadHistograms[i].fill(0);
  }

  this->Modified();
}

template <class TFixedImage, class TMovingImage>
void
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
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

  for (ind[2] = m_Skips[2]; ind[2] < (long)fixedSize[2]; ind[2] += m_Skips[2])
    for (ind[1] = m_Skips[1]; ind[1] < (long)fixedSize[1]; ind[1] += m_Skips[1])
      for (ind[0] = m_Skips[0]; ind[0] < (long)fixedSize[0]; ind[0] += m_Skips[0])
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
        float fx = movingInd[0] - (float)x0;
        float fy = movingInd[1] - (float)y0;
        float fz = movingInd[2] - (float)z0;

        float gx = 1.0 - fx;
        float gy = 1.0 - fy;
        float gz = 1.0 - fz;

        // Moving image histogram index (column)
        unsigned int c = 0;

/*
// Nearest-neighbor interp of quantized values
    MovingImageIndexType nn_ind;
    nn_ind[0] = (long)(movingInd[0] + 0.5);
    nn_ind[1] = (long)(movingInd[1] + 0.5);
    nn_ind[2] = (long)(movingInd[2] + 0.5);
    if (nn_ind[0] < 0 || nn_ind[0] >= (long)movingSize[0]
        ||
        nn_ind[1] < 0 || nn_ind[1] >= (long)movingSize[1]
        ||
        nn_ind[2] < 0 || nn_ind[2] >= (long)movingSize[2])
// PP: Add BG component???
      continue;
    c = m_MovingIndexImage->GetPixel(nn_ind);
    if (c >= m_NumberOfBins)
      continue;
    H(r, c) += 1.0;
*/

/*
// Linear interpolation
// Note: Do not use linear interp with non-uniform spaced bins
// Need to account for hist spacing h += area * 1
        float c_interp = 0;

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
  float sumHist = 0;
  for (unsigned int r = 0; r < m_NumberOfBins; r++)
    for (unsigned int c = 0; c < m_NumberOfBins; c++)
      sumHist += H(r, c);
  if (sumHist != 0)
    H /= sumHist;

}

template <class TFixedImage, class TMovingImage>
void
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
::ThreadedComputeHistogram() const
{

  itkDebugMacro(<< "ThreadedComputeHistogram");

  *m_ThreadIndexCount = 0;

  int numThreads = itk::MultiThreader::GetGlobalDefaultNumberOfThreads();

  for (int i = 0; i < numThreads; i++)
    m_ThreadHistograms[i].fill(0);

  itk::MultiThreader::Pointer threader = itk::MultiThreader::New();

  threader->SetNumberOfThreads(numThreads);
  threader->SetSingleMethod(
    &NegativeMIImageMatchMetric::_threadFillHistogram, (void*)this);
  threader->SingleMethodExecute();

  HistogramType& H = *m_HistogramPointer;
  H.fill(0);

  for (int i = 0; i < numThreads; i++)
    H += m_ThreadHistograms[i];

  // Normalize histogram values
  float sumHist = 0;
  for (unsigned int r = 0; r < m_NumberOfBins; r++)
    for (unsigned int c = 0; c < m_NumberOfBins; c++)
      sumHist += H(r, c);
  if (sumHist != 0)
    H /= sumHist;

}

template <class TFixedImage, class TMovingImage>
ITK_THREAD_RETURN_TYPE
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
::_threadFillHistogram(void* arg)
{
  typedef itk::MultiThreader::ThreadInfoStruct ThreadInfoType;
  ThreadInfoType * infoStruct = static_cast< ThreadInfoType * >( arg );

  const unsigned int threadId = infoStruct->ThreadID;

  NegativeMIImageMatchMetric* obj = static_cast< NegativeMIImageMatchMetric* >( infoStruct->UserData );

  HistogramType& H = obj->m_ThreadHistograms[threadId];

  FixedImagePointType fixedOrigin = obj->m_FixedIndexImage->GetOrigin();

  FixedImageSpacingType fixedSpacing = obj->m_FixedIndexImage->GetSpacing();

  FixedImageSizeType fixedSize =
    obj->m_FixedIndexImage->GetLargestPossibleRegion().GetSize();

  MovingImagePointType movingOrigin = obj->m_MovingIndexImage->GetOrigin();

  MovingImageSpacingType movingSpacing = obj->m_MovingIndexImage->GetSpacing();

  MovingImageSizeType movingSize =
    obj->m_MovingIndexImage->GetLargestPossibleRegion().GetSize();

  while (true)
  {

    obj->m_Mutex.Lock();
    int pos = (*obj->m_ThreadIndexCount)++;
    obj->m_Mutex.Unlock();

    if (pos >= obj->m_ThreadIndices.size())
      break;

    FixedImageIndexType ind = obj->m_ThreadIndices[pos];

    // Get sampled fixed image histogram index
    unsigned int r = obj->m_FixedIndexImage->GetPixel(ind);

    // Skip if fixed image histogram index is invalid
    if (r >= obj->m_NumberOfBins)
      continue;

    FixedImagePointType fixedPoint;
    obj->m_FixedImage->TransformIndexToPhysicalPoint(ind, fixedPoint);

    MovingImagePointType mappedPoint =
      obj->m_Transform->TransformPoint(fixedPoint);

    // Use Partial Volume interpolation
    
    // Get continuous moving image coordinates (in voxels)
    typedef itk::ContinuousIndex<double, 3> ContinuousIndexType;
    ContinuousIndexType movingInd;
    obj->m_MovingImage->TransformPhysicalPointToContinuousIndex(
      mappedPoint, movingInd);

    // Get image neighborhood
    int x0 = (int)movingInd[0];
    int y0 = (int)movingInd[1];
    int z0 = (int)movingInd[2];

    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    // Get distances to the image grid
    float fx = movingInd[0] - (float)x0;
    float fy = movingInd[1] - (float)y0;
    float fz = movingInd[2] - (float)z0;

    float gx = 1.0 - fx;
    float gy = 1.0 - fy;
    float gz = 1.0 - fz;

    // Moving image histogram index (column)
    unsigned int c = 0;

/*
// Nearest-neighbor interp of quantized values
    MovingImageIndexType nn_ind;
    nn_ind[0] = (long)(movingInd[0] + 0.5);
    nn_ind[1] = (long)(movingInd[1] + 0.5);
    nn_ind[2] = (long)(movingInd[2] + 0.5);
    if (nn_ind[0] < 0 || nn_ind[0] >= (long)movingSize[0]
        ||
        nn_ind[1] < 0 || nn_ind[1] >= (long)movingSize[1]
        ||
        nn_ind[2] < 0 || nn_ind[2] >= (long)movingSize[2])
// PP: Add BG component???
      continue;
    c = obj->m_MovingIndexImage->GetPixel(nn_ind);
    if (c >= obj->m_NumberOfBins)
      continue;
    H(r, c) += 1.0;
*/

/*
// Linear interpolation
// Note: Do not use linear interp with non-uniform spaced bins
// Need to account for hist spacing h += area * 1
        float c_interp = 0;

#define interpWeightMacro(x, y, z, w) \
  if ((0 <= (x)) && ((x) < (long)movingSize[0]) && \
    (0 <= (y)) && ((y) < (long)movingSize[1]) && \
    (0 <= (z)) && ((z) < (long)movingSize[2])) \
  { \
    MovingImageIndexType local_ind = {{(x), (y), (z)}}; \
    c_interp += (w) * obj->m_MovingIndexImage->GetPixel(local_ind); \
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
        if (c >= obj->m_NumberOfBins)
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
    c = obj->m_MovingIndexImage->GetPixel(pvind); \
    if (c < obj->m_NumberOfBins) \
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

  return ITK_THREAD_RETURN_VALUE;
}

template <class TFixedImage, class TMovingImage>
float
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
::ComputeMI() const
{
  // Compute histogram
  //this->ComputeHistogram();
  this->ThreadedComputeHistogram();

  itkDebugMacro(<< "Start MI");

  HistogramType& H = *m_HistogramPointer;

#if 1
  // ITK version
  float totalf = 0.0;
  for (unsigned c = 0; c < m_NumberOfBins; c++)
  {
    for (unsigned r = 0; r < m_NumberOfBins; r++)
    {
      totalf += H(r, c);
    }
  }

  // All probabilities are zero, -E[log(p)] assumed to be zero
  if (totalf <= 0.0)
    return 0;

  float logtotalf = logf(totalf);

  float entropyA = 0.0;
  for (unsigned r = 0; r < m_NumberOfBins; r++)
  {
    float f = 0.0;
    for (unsigned c = 0; c < m_NumberOfBins; c++)
    {
      f += H(r, c);
    }
    if (f > 0.0)
      entropyA += f*logf(f);
  }
  // Negate sum and normalize histogram values
  entropyA = -entropyA / totalf + logtotalf;

  float entropyB = 0.0;
  for (unsigned c = 0; c < m_NumberOfBins; c++)
  {
    float f = 0.0;
    for (unsigned r = 0; r < m_NumberOfBins; r++)
    {
      f += H(r, c);
    }
    if (f > 0.0)
      entropyB += f*logf(f);
  }
  entropyB = -entropyB / totalf + logtotalf;

  float jointEntropy = 0.0;
  for (unsigned c = 0; c < m_NumberOfBins; c++)
  {
    for (unsigned r = 0; r < m_NumberOfBins; r++)
    {
      float f = H(r, c);
      if (f > 0.0)
        jointEntropy += f*logf(f);
    }
  }
  jointEntropy = -jointEntropy / totalf + logtotalf;

  if (m_Normalized)
  {
    if (jointEntropy == 0.0)
      return 0.0;
    else
      return (entropyA + entropyB) / jointEntropy;
  }

  return (entropyA + entropyB) - jointEntropy;
#else

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

  float mi = 0;
  for (unsigned int i = 0; i < m_NumberOfBins; i++)
    for (unsigned int j = 0; j < m_NumberOfBins; j++)
    {
      float p = H(i, j);
      if (p <= 0.0)
        continue;
      float prodMarginals = marginalA(i, 0) * marginalB(j, 0);
      if (prodMarginals <= 0.0)
        continue;
      mi += p * logf(p / prodMarginals);
    }

  if (m_Normalized)
  {
    float entropyA = 0;
    for (unsigned int i = 0; i < m_NumberOfBins; i++)
    {
      float p = marginalA(i, 0);
      if (p <= 0.0)
        continue;
      entropyA -= p * logf(p);
    }

    float entropyB = 0;
    for (unsigned int i = 0; i < m_NumberOfBins; i++)
    {
      float p = marginalB(i, 0);
      if (p <= 0.0)
        continue;
      entropyB -= p * logf(p);
    }

    float denom = (entropyA + entropyB);
    if (denom != 0.0)
      mi /= denom;
  }

  return mi;

#endif

}

template <class TFixedImage, class TMovingImage>
typename NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
::MeasureType
NegativeMIImageMatchMetric<TFixedImage,TMovingImage>
::GetValue(const ParametersType& parameters) const
{

  // Make sure the transform has the current parameters
  this->m_Transform->SetParameters(parameters);

  return -1.0*this->ComputeMI();

}

template <class TFixedImage, class TMovingImage>
void
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
::GetValueAndDerivative(const ParametersType& parameters, MeasureType& value,
  DerivativeType& derivative) const
{
  value = this->GetValue(parameters);
  //this->GetDerivative(parameters, derivative);
  this->GetStochasticDerivative(parameters, derivative);
}

template <class TFixedImage, class TMovingImage>
void
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
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
    float v1 = this->GetValue(p1);

    ParametersType p2 = parameters;
    p2[i] += m_DerivativeStepLengths[i];
    float v2 = this->GetValue(p2);

    derivative[i] = (v2 - v1) / (2.0*m_DerivativeStepLengths[i]);
  }
}

// Compute derivative following SPSA (Spall)
template <class TFixedImage, class TMovingImage>
void
NegativeMIImageMatchMetric<TFixedImage, TMovingImage>
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
    float r = rng->GenerateUniformRealClosedInterval();
    if (r >= 0.5)
      dp[i] = m_DerivativeStepLengths[i];
    else
      dp[i] = -m_DerivativeStepLengths[i];

/*
    // r in [-1, 1]
    float r = 2.0*rng->GenerateUniformRealClosedInterval() - 1.0;
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

  float v1 = this->GetValue(p1);
  float v2 = this->GetValue(p2);

  float v_diff = v1 - v2;

  for (unsigned int i = 0; i < numParams; i++)
  {
    derivative[i] = v_diff / (2.0*dp[i]);
  }
}

#endif
