
#ifndef _IntensityMatcher_txx
#define _IntensityMatcher_txx

#include "IntensityMatcher.h"

#include "itkImageRegionIteratorWithIndex.h"

#include "vnl/vnl_math.h"

#include "Heap.h"

#include <cmath>

// Object for doing argsort
struct IDouble{
  double v;
  unsigned int i;
  bool operator < ( const IDouble& arg ) const
  {
    return v < arg.v;
  }
};

template <class TInputImage, class TProbImage>
IntensityMatcher<TInputImage, TProbImage>
::IntensityMatcher()
{
  m_Modified = false;
}

template <class TInputImage, class TProbImage>
IntensityMatcher<TInputImage, TProbImage>
::~IntensityMatcher()
{

}

template <class TInputImage, class TProbImage>
void
IntensityMatcher<TInputImage, TProbImage>
::SetSourceImage(InputImagePointer img)
{
  m_Modified = true;
  m_SourceImage = img;
}

template <class TInputImage, class TProbImage>
double
IntensityMatcher<TInputImage, TProbImage>
::EvaluatePoint(double x)
{
  unsigned int numClasses = m_SourcePoints.GetSize();

  double x1, x2;
  double y1, y2;

  if (x < m_SourcePoints[0])
  {
    x1 = m_SourcePoints[0];
    x2 = m_SourcePoints[1];
    y1 = m_TargetPoints[0];
    y2 = m_TargetPoints[1];
    double v = (y2 - y1) / (x2 - x1 + 1e-20) * (x - x1) + y1;
    if (v < m_MinTarget)
      v = m_MinTarget;
    if (v > m_MaxTarget)
      v = m_MaxTarget;
    return m_TargetPoints[0];
  }

  if (x > m_SourcePoints[numClasses-1])
  {
    x1 = m_SourcePoints[numClasses-2];
    x2 = m_SourcePoints[numClasses-1];
    y1 = m_TargetPoints[numClasses-2];
    y2 = m_TargetPoints[numClasses-1];
    double v = (y2 - y1) / (x2 - x1 + 1e-20) * (x - x2) + y2;
    if (v < m_MinTarget)
      v = m_MinTarget;
    if (v > m_MaxTarget)
      v = m_MaxTarget;
    return v;
  }

  unsigned int c = 0;
  for (; c < (numClasses-1); c++)
  {
    if (x >= m_SourcePoints[c] && x <= m_SourcePoints[c+1])
      break;
  }

  x1 = m_SourcePoints[c];
  x2 = m_SourcePoints[c+1];
  y1 = m_TargetPoints[c];
  y2 = m_TargetPoints[c+1];

  return (y2 - y1) / (x2 - x1 + 1e-20) * (x - x1) + y1;
}

template <class TInputImage, class TProbImage>
void
IntensityMatcher<TInputImage, TProbImage>
::SetTargetImage(InputImagePointer img)
{
  m_Modified = true;
  m_TargetImage = img;
}

template <class TInputImage, class TProbImage>
void
IntensityMatcher<TInputImage, TProbImage>
::SetProbabilities(const DynArray<InputImagePointer>& probs)
{
  m_Modified = true;
  m_Probs = probs;
}

template <class TInputImage, class TProbImage>
void
IntensityMatcher<TInputImage, TProbImage>
::Update()
{
  if (!m_Modified)
    return;

  unsigned int numClasses = m_Probs.GetSize();

  // Estimate class intensities
  m_SourcePoints.Initialize(numClasses, 0.0);
  m_TargetPoints.Initialize(numClasses, 0.0);

  DynArray<double> sumProbs;
  sumProbs.Initialize(numClasses, 1e-20);

  typedef itk::ImageRegionIteratorWithIndex<ProbImageType> IteratorType;
  IteratorType it(m_Probs[0], m_Probs[0]->GetLargestPossibleRegion());

  m_MinSource = vnl_huge_val(1.0);
  m_MinTarget = vnl_huge_val(1.0);

  m_MaxSource = -vnl_huge_val(1.0);
  m_MaxTarget = -vnl_huge_val(1.0);

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    InputImageIndexType ind = it.GetIndex();

    double v_s = m_SourceImage->GetPixel(ind);
    double v_t = m_TargetImage->GetPixel(ind);

    if (v_s < m_MinSource)
      m_MinSource = v_s;
    if (v_s > m_MaxSource)
      m_MaxSource = v_s;

    if (v_t < m_MinTarget)
      m_MinTarget = v_t;
    if (v_t > m_MaxTarget)
      m_MaxTarget = v_t;

    for (unsigned int c = 0; c < numClasses; c++)
    {
      double p = pow((double)m_Probs[c]->GetPixel(ind), (double)2.0);
      m_SourcePoints[c] += p * v_s;
      m_TargetPoints[c] += p * v_t;
      sumProbs[c] += p;
    }
  }

  for (unsigned int c = 0; c < numClasses; c++)
  {
    m_SourcePoints[c] /= sumProbs[c];
    m_TargetPoints[c] /= sumProbs[c];
  }

  // Add BG, special case for MRI
  m_SourcePoints.Append(0);
  m_TargetPoints.Append(0);
  numClasses++;

  // Sort based on the source points
  Heap<IDouble> heap;
  for (unsigned int c = 0; c < numClasses; c++)
  {
    IDouble d;
    d.v = m_SourcePoints[c];
    d.i = c;
    heap.Insert(d);
  }

  DynArray<double> oldSourcePoints = m_SourcePoints;
  DynArray<double> oldTargetPoints = m_TargetPoints;

  for (unsigned int c = 0; c < numClasses; c++)
  {
    IDouble min = heap.ExtractMinimum();
    unsigned int i = min.i;
    m_SourcePoints[c] = oldSourcePoints[i];
    m_TargetPoints[c] = oldTargetPoints[i];
  }

  // Remap image intensities
  m_OutputImage = InputImageType::New();
  m_OutputImage->CopyInformation(m_SourceImage);
  m_OutputImage->SetRegions(m_SourceImage->GetLargestPossibleRegion());
  m_OutputImage->Allocate();

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    InputImageIndexType ind = it.GetIndex();
    m_OutputImage->SetPixel(ind,
      this->EvaluatePoint(m_SourceImage->GetPixel(ind)) );
  }

  m_Modified = false;
}


#endif
