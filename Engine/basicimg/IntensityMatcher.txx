
#ifndef _IntensityMatcher_txx
#define _IntensityMatcher_txx

#include "IntensityMatcher.h"

#include "itkImageRegionIteratorWithIndex.h"


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

  if (x < m_SourcePoints[0])
    return m_TargetPoints[0];
  if (x > m_SourcePoints[numClasses-1])
    return m_TargetPoints[numClasses-1];

  unsigned int c = 0;
  for (; c < (numClasses-1); c++)
  {
    if (x >= m_SourcePoints[c] && x <= m_SourcePoints[c+1])
      break;
  }

  if (c >= (numClasses-1))
    return m_TargetPoints[numClasses-1];

  double x1 = m_SourcePoints[c];
  double x2 = m_SourcePoints[c+1];
  double y1 = m_TargetPoints[c];
  double y2 = m_TargetPoints[c+1];

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

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    InputImageIndexType ind = it.GetIndex();

    for (unsigned int c = 0; c < numClasses; c++)
    {
      double p = m_Probs[c]->GetPixel(ind);
      m_SourcePoints[c] += p * m_SourceImage->GetPixel(ind);
      m_TargetPoints[c] += p * m_TargetImage->GetPixel(ind);
      sumProbs[c] += p;
    }
  }

  for (unsigned int c = 0; c < numClasses; c++)
  {
    m_SourcePoints[c] /= sumProbs[c];
    m_TargetPoints[c] /= sumProbs[c];
  }

  // Add BG
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
