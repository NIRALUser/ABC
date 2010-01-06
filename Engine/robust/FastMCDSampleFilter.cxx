
#include "FastMCDSampleFilter.h"

#include "vnl/algo/vnl_determinant.h"
#include "vnl/vnl_math.h"

#include "DynArray.h"
#include "Heap.h"
//#include "Log.h"
#include "MersenneTwisterRNG.h"
#include "muException.h"

#include <iostream>

#include <float.h>
#include <math.h>

// Factorial, to find number of possible subset selections
static double factorial(double n)
{
  if (n <= 1)
    return 1;

  double f = 1;
  for (unsigned int k = 1; k <= n; k++)
    f *= k;

  return f;
}

////////////////////////////////////////////////////////////////////////////////
// Objects for doing argsort
////////////////////////////////////////////////////////////////////////////////

class TaggedFloat
{
public:
  TaggedFloat() { tag = 0; value = 0; }
  bool operator<(const TaggedFloat& f) const { return this->value < f.value; }
  const TaggedFloat& operator=(const TaggedFloat& f)
  { this->tag = f.tag; this->value = f.value; return *this; }
public:
  unsigned int tag;
  double value;
};


////////////////////////////////////////////////////////////////////////////////
// Begin class definitions
////////////////////////////////////////////////////////////////////////////////

FastMCDSampleFilter
::FastMCDSampleFilter()
{
  m_ChangeTolerance = 1e-10;

  m_CoverFraction = 0.5;

  m_NumberOfStarts = 1000;

  m_MaxCStepIterations = 100;
}

FastMCDSampleFilter
::~FastMCDSampleFilter()
{

}

FastMCDSampleFilter::SampleMatrixType
FastMCDSampleFilter
::GetInliers(const SampleMatrixType& samples, double mahaThres)
{

  unsigned int numSamples = samples.rows();
  unsigned int dimension = samples.columns();

  // Number of samples to cover
  unsigned int h = static_cast<unsigned int>(
    floor((numSamples + dimension + 1) / 2.0) + 1);

  if (numSamples <= dimension || numSamples <= h || mahaThres <= 0.0)
  {
    SampleMatrixType samplesCopy = samples;
    return samplesCopy;
  }

  MatrixType mean;
  MatrixType covariance;
  this->GetRobustEstimate(mean, covariance, samples);

  // Inverse covariance
  MatrixType invCov = MatrixInverseType(covariance);

  // Threshold sample values, find inlier row indices
  DynArray<unsigned int> inlierInds;
  MatrixType x(1, dimension);

  for (unsigned int i = 0; i < numSamples; i++)
  {
    for (unsigned int j = 0; j < dimension; j++)
      x(0, j) = samples(i, j) - mean(0, j);

    MatrixType d = x * invCov * x.transpose();
    double mahalanobis = sqrt(d(0, 0));
    if (mahalanobis <= mahaThres)
      inlierInds.Append(i);
  }

  // Build list of inliers
  SampleMatrixType inliers(inlierInds.GetSize(), dimension, 0.0);
  for (unsigned int i = 0; i < inlierInds.GetSize(); i++)
  {
    for (unsigned int j = 0; j < dimension; j++)
      inliers(i, j) = samples(inlierInds[i], j);
  }

  return inliers;

}

void
FastMCDSampleFilter
::GetRobustEstimate(MatrixType& mean, MatrixType& covariance,
  const SampleMatrixType& samples)
{

  unsigned int numSamples = samples.rows();
  unsigned int dimension = samples.columns();

//std::cout << "PP robest w/ " << numSamples << " samples" << std::endl;

  // Number of samples to cover
  unsigned int h = static_cast<unsigned int>(
    floor((numSamples + dimension + 1) *m_CoverFraction) + 1);

  // Give whole sample mean and covariance if sample size is too small
  if (numSamples <= dimension || numSamples <= h)
  {
    IndexList tmpInd;
    tmpInd.Allocate(numSamples);
    for (unsigned int i = 0; i < numSamples; i++)
      tmpInd.Append(i);
    this->_ComputeEllipsoid(mean, covariance, samples, tmpInd);
    return;
  }

  // Get the Mersenne twister RNG
  MersenneTwisterRNG* rng = MersenneTwisterRNG::GetGlobalInstance();

  // Find total number of subset selections
  unsigned long numPossibleStarts = static_cast<unsigned long>(
    factorial(numSamples) / (factorial(numSamples - h) * factorial(h)));

  // Limit number of starts
  unsigned int numStarts = m_NumberOfStarts;
  if (numStarts > numPossibleStarts)
    numStarts = numPossibleStarts;

  // Minimum number of starts (just to be safe in case of overflow etc)
  if (numStarts < 2)
    numStarts = 2;

  // Create a list of a set of random sample indices
  DynArray<IndexList> subsetList;
  subsetList.Allocate(numStarts);
  for (unsigned int i = 0; i < numStarts; i++)
  {
    unsigned int* seq = rng->GenerateIntegerSequence(h, numSamples-1);

    // Wrap the "raw" array
    IndexList tmpInds;
    tmpInds.WrapArray(seq, h);
    subsetList.Append(tmpInds);
  }

  // Heap for doing argsort based on determinant of covariance
  Heap<TaggedFloat> dheap;
  dheap.Allocate(numStarts);

  // Perfom C-steps
  unsigned int initMaxCStepIters = m_MaxCStepIterations / 10 + 1;
  for (unsigned int k = 0; k < numStarts; k++)
  {
    TaggedFloat f;
    f.tag = k;
    f.value = this->_CSteps(samples, subsetList[k], initMaxCStepIters);

    if (f.value < 1e-10)
      continue;

    dheap.Insert(f);
  }

  unsigned int numCandidates = numStarts / 10 + 3;

  // Do C-Steps on best candidates
  unsigned int ibest = numStarts; // Which one is best candidate?
  double mindetcov = vnl_huge_val(1.0);

  for (unsigned int k = 0; k < numCandidates; k++)
  {
    if (dheap.IsEmpty())
      break;

    TaggedFloat f = dheap.ExtractMinimum();

    unsigned int r = f.tag;

    double detcov = 
      this->_CSteps(samples, subsetList[r], m_MaxCStepIterations);

    if ((detcov > 0.0) && (detcov < mindetcov))
    {
      ibest = r;
      mindetcov = detcov;
    }
  }

  if (ibest == numStarts)
  {
    // No best candidates, give robust estimate as the actual whole sample
    // mean and covariance
    //muLogMacro(<< "FastMCD: no best candidates\n");
    IndexList tmpInd;
    tmpInd.Allocate(numSamples);
    for (unsigned int i = 0; i < numSamples; i++)
      tmpInd.Append(i);
    this->_ComputeEllipsoid(mean, covariance, samples, tmpInd);
    return;
  }

  //muLogMacro(
  //  << "FastMCD: final determinant of covariance = " << mindetcov << "\n");

  // Compute the robust mean and covariance using the optimal subset
  this->_ComputeEllipsoid(mean, covariance, samples, subsetList[ibest]);

}

void
FastMCDSampleFilter
::GetRobustEstimate(VectorType& mean, MatrixType& covariance,
  const SampleMatrixType& samples)
{
  MatrixType muM;

  this->GetRobustEstimate(muM, covariance, samples);

  mean = muM.get_row(0);
}

void
FastMCDSampleFilter
::SetMaxCStepIterations(unsigned int n)
{
  if (n < 1)
  {
    muExceptionMacro(
      << "MCD max C-step iterations need to be greater than zero");
  }

  m_MaxCStepIterations = n;
}

double
FastMCDSampleFilter
::_CSteps(const SampleMatrixType& samples, const IndexList& indices,
  unsigned int maxIters)
{

  unsigned int numSamples = samples.rows();
  unsigned int dimension = samples.columns();

  // Subset size
  unsigned int numInd = indices.GetSize();

  MatrixType mean;
  MatrixType covariance;
  this->_ComputeEllipsoid(mean, covariance, samples, indices);

  // Compute det covar
  double detCov = vnl_determinant(covariance);
  double prevDetCov = detCov;

  if (detCov <= 0)
    return detCov;

  unsigned int numIters = 0;

  IndexList backupInd = indices;

  while (1)
  {
    numIters++;

//std::cout << "PP CStep " << numIters << " detcov = " << detCov << std::endl;

    // Heap for argsort based on Mahalanobis distances
    Heap<TaggedFloat> dheap;
    dheap.Allocate(numSamples);
  
    // Compute Mahalanobis distances
    MatrixType invCov = MatrixInverseType(covariance);
    MatrixType x(1, dimension);

    for (unsigned int i = 0; i < numSamples; i++)
    {
      for (unsigned int j = 0; j < dimension; j++)
        x(0, j) = samples(i, j) - mean(0, j);

      MatrixType dmat = x * invCov * x.transpose();
      double mahalanobis = sqrt(dmat(0, 0));

      TaggedFloat f;
      f.tag = i;
      f.value = mahalanobis;

      dheap.Insert(f);
    }

    // Store the current subset selection
    backupInd = indices;

    // Select the points closest to current mean    
    for (unsigned int k = 0; k < numInd; k++)
    {
      TaggedFloat f = dheap.ExtractMinimum();
      indices[k] = f.tag;
    }

    // Update ellipsoid
    this->_ComputeEllipsoid(mean, covariance, samples, indices);

    prevDetCov = detCov;
    detCov = vnl_determinant(covariance);

    if (detCov <= 1e-10 || detCov > prevDetCov)
    {
      // Restore previous subset selection and stop
      for (unsigned int k = 0; k < numInd; k++)
        indices[k] = backupInd[k];
      detCov = prevDetCov;
      break;
    }

    //double changeDetCov = fabs(prevDetCov - detCov);
    double changeDetCov =
      fabs((prevDetCov - detCov) / (prevDetCov + DBL_EPSILON));

    if (numIters >= maxIters || changeDetCov < m_ChangeTolerance)
      break;

  }

  return detCov;

}

void
FastMCDSampleFilter
::_ComputeEllipsoid(MatrixType& mean, MatrixType& covariance,
  const SampleMatrixType& samples, const IndexList& indices)
{

  if (indices.GetSize() == 0)
  {
    muExceptionMacro(
      << "[FastMCDSampleFilter::ComputeEllipsoid] Zero sample size");
  }

  unsigned int dimension = samples.columns();

  unsigned int numInd = indices.GetSize();

  mean.set_size(1, dimension);
  for (unsigned int j = 0; j < dimension; j++)
    mean(0, j) = 0.0;

  for (unsigned int i = 0; i < numInd; i++)
  {
    for (unsigned int j = 0; j < dimension; j++)
      mean(0, j) += samples(indices[i], j);
  }
  mean /= (double)numInd;

  covariance.set_size(dimension, dimension);
  for (unsigned int i = 0; i < dimension; i++)
    for (unsigned int j = 0; j < dimension; j++)
      covariance(i, j) = 0.0;

  if (numInd <= 1)
  {
std::cerr << "Subset size too small: " << numInd << std::endl;
    return;
  }

  for (unsigned int i = 0; i < dimension; i++)
  {      
    for (unsigned int j = i; j < dimension; j++)
    {
      double v = 0.0;
      for (unsigned int k = 0; k < numInd; k++)
      {
        double diff1 = samples(indices[k], i) - mean(0, i);
        double diff2 = samples(indices[k], j) - mean(0, j);
        v += diff1 * diff2;
      }

      v /= (numInd - 1);

      // Adjust diagonal element, make sure covariance is pos-def
      if (i == j)
        v += 1e-10;

      covariance(i, j) = v;
      covariance(j, i) = v;
    }
  }

}
