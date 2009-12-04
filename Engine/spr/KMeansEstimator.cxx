
#include "KMeansEstimator.h"

#include "itkVariableLengthVector.h"
#include "itkListSample.h"
#include "itkKdTree.h"
#include "itkWeightedCentroidKdTreeGenerator.h"
#include "itkKdTreeBasedKmeansEstimator.h"

#include "DynArray.h"
#include "Heap.h"
#include "Log.h"
#include "MersenneTwisterRNG.h"

#include "muException.h"

#include "vnl/vnl_math.h"

KMeansEstimator
::KMeansEstimator()
{

  m_Samples = MatrixType(0, 1);

  m_MaximumIterations = 100;

  m_NumberOfClusters = 2;

  m_NumberOfStarts = 10;

  m_Modified = false;

  m_UseKdTree = false;

}

KMeansEstimator
::~KMeansEstimator()
{

}

void
KMeansEstimator
::SetInput(MatrixType& samples)
{

  m_Samples = samples;

  m_Modified = true;
}

void
KMeansEstimator
::SetNumberOfClusters(unsigned int k)
{
  if (k == 0)
    muExceptionMacro(<< "Cannot find 0 clusters");

  m_NumberOfClusters = k;

  m_Modified = true;
}

void
KMeansEstimator
::SetNumberOfStarts(unsigned int s)
{
  if (s == 0)
    muExceptionMacro(<< "Cannot have zero starts");

  m_NumberOfStarts = s;

  m_Modified = true;
}

void
KMeansEstimator
::SetMaximumIterations(unsigned int n)
{
  if (n == 0)
    muExceptionMacro(<< "Cannot have zero max iterations");

  m_MaximumIterations = n;

  m_Modified = true;
}

void
KMeansEstimator
::Update()
{

  unsigned int k = m_NumberOfClusters;
  unsigned int n = m_Samples.rows();

  unsigned int dim = m_Samples.cols();

  if (k > n)
    muExceptionMacro(<<" Cannot find more clusters than data points");

  if (k == n)
  {
    m_Means = m_Samples;

    m_Labels.Clear();
    m_Labels.Allocate(n);
    for (unsigned int i = 0; i < n; i++)
      m_Labels.Append(i);

    m_Modified = false;

    return;
  }

//std::cout << "Kmeans: " << n << " samples, k = " << k << std::endl;

  DynArray<unsigned int> tempLabels;
  tempLabels.Initialize(n, 0);

  MatrixType tempMeans(k, dim, 0);

  double minsum = vnl_huge_val(1.0);

  for (unsigned int istart = 0; istart < m_NumberOfStarts; istart++)
  {
//std::cout << "Start " << istart << std::endl;
    double sumdist = this->_Guess(tempLabels, tempMeans);

//std::cout << "Start " << istart << ", sum distance = " << sumdist << std::endl;

    if (sumdist < minsum) 
    {
      minsum = sumdist;

      m_Labels = tempLabels;
      m_Means = tempMeans;
    }

  }

//std::cout << "Final within-cluster sum distance = " << minsum << std::endl;

  m_Modified = false;

}

DynArray<unsigned int>
KMeansEstimator
::GetLabels()
{
  if (m_Modified)
    this->Update();

  return m_Labels;
}

KMeansEstimator::MatrixType
KMeansEstimator
::GetMeans()
{
  if (m_Modified)
    this->Update();

  return m_Means;
}

double
KMeansEstimator
::_Guess(DynArray<unsigned int>& labels, MatrixType& means)
{
  unsigned int k = m_NumberOfClusters;
  unsigned int n = m_Samples.rows();
  unsigned int dim = m_Samples.cols();

  if (labels.GetSize() != n)
    labels.Initialize(n, 0);

  if (means.rows() != k || means.cols() != dim)
    means.set_size(k, dim);

  // Select sample points at random as initial guess for the means
  MersenneTwisterRNG* rng = MersenneTwisterRNG::GetGlobalInstance();
  unsigned int* guessIndices = rng->GenerateIntegerSequence(k, n-1);

  means.set_size(k, dim);
  for (unsigned int i = 0; i < k; i++)
    means.set_row(i, m_Samples.get_row(guessIndices[i]));

  delete [] guessIndices;

  // Jiggle the means
  VectorType center(dim);
  for (unsigned i = 0; i < k; i++)
    center += means.get_row(i);
  center /= k;

  VectorType std(dim);
  for (unsigned i = 0; i < k; i++)
    std += means.get_row(i) - center;
  std /= k;

  for (unsigned int i = 0; i < k; i++)
  {
    VectorType mu_i = means.get_row(i);
    double r = rng->GenerateUniformRealClosedInterval();
    mu_i += std * (0.1 * r);
    means.set_row(i, mu_i);
  }

  // Start the K-means process
  if (m_UseKdTree)
  {
    //
    // K-d tree based k-means estimation
    // Use k-d tree partitions to determine the cluster means
    // Faster, takes more memory
    //

    typedef itk::VariableLengthVector<double> MeasurementVectorType;
    typedef itk::Statistics::ListSample<MeasurementVectorType> SampleType;
    SampleType::Pointer kdsample = SampleType::New();
    kdsample->SetMeasurementVectorSize(dim);

    for (unsigned int i = 0; i < n; i++)
    {
      MeasurementVectorType mv(dim);
      for (unsigned int d = 0; d < dim; d++)
        mv[d] = m_Samples(i, d);
      kdsample->PushBack(mv);
    }

    // Bucket size needs to be > 2, otherwise crash (why?)
    unsigned int bucketSize = n / 10 + 4;
    if (bucketSize >  100)
      bucketSize = 100;

    typedef itk::Statistics::WeightedCentroidKdTreeGenerator<SampleType>
      TreeGeneratorType;
    TreeGeneratorType::Pointer treeGen = TreeGeneratorType::New();
    treeGen->SetSample(kdsample);
    treeGen->SetBucketSize(bucketSize);
    treeGen->Update();

    typedef itk::Statistics::KdTreeBasedKmeansEstimator<
      TreeGeneratorType::KdTreeType> EstimatorType;
    EstimatorType::Pointer estimator = EstimatorType::New();

    EstimatorType::ParametersType initialMeans(k*dim);
    unsigned int idx = 0;
    for (unsigned int j = 0; j < k; j++)
    {
      for (unsigned int d = 0; d < dim; d++)
      {
        initialMeans[idx] = means(j, d);
        idx++;
      }
    }

    estimator->SetParameters(initialMeans);
    estimator->SetKdTree(treeGen->GetOutput());
    estimator->SetMaximumIteration(m_MaximumIterations);
    estimator->SetCentroidPositionChangesThreshold(0.0);
    estimator->StartOptimization();

    EstimatorType::ParametersType outMeans = estimator->GetParameters();

    // Update means
    idx = 0;
    for (unsigned int j = 0; j < k; j++)
    {
      for (unsigned int d = 0; d < dim; d++)
      {
        means(j, d) = outMeans[idx];
        idx++;
      }
    }

    // Clean up k-d tree vars
    treeGen = 0;
    estimator = 0;
    kdsample = 0;

    // Classify sample points based on the means
    for (unsigned int isample = 0; isample < n; isample++)
    {
      VectorType x = m_Samples.get_row(isample);

      unsigned int which = 0;
      double mindist = vnl_huge_val(1.0);

      for (unsigned int iclass = 0; iclass < k; iclass++)
      {
        VectorType dvec = x - means.get_row(iclass);
        double dist = dvec.squared_magnitude();
        if (dist < mindist)
        {
          which = iclass;
          mindist = dist;
        }
      }

      labels[isample] = which;
    }

  }
  else
  {
    //
    // Lloyd's algorithm
    // Slow but does not take much memory
    //

    // Loop for max likelihood fit
    unsigned int iter = 0;
    while (true)
    {
      iter++;

      bool noLabelChange = true;

      // E-step: Classify sample points based on the means
      for (unsigned int isample = 0; isample < n; isample++)
      {
        VectorType x = m_Samples.get_row(isample);

        unsigned int which = 0;
        double mindist = vnl_huge_val(1.0);

        for (unsigned int iclass = 0; iclass < k; iclass++)
        {
          VectorType d = x - means.get_row(iclass);
          double dist = d.squared_magnitude();
          if (dist < mindist)
          {
            which = iclass;
            mindist = dist;
          }
        }

        if (labels[isample] != which)
          noLabelChange = false;

        labels[isample] = which;
      } // for isample

      // Check convergence
      if (iter >= m_MaximumIterations)
        break;
      if (noLabelChange)
        break;

      // M-step: Recompute means
      for (unsigned int iclass = 0; iclass < k; iclass++)
      {
        VectorType mu(dim);
        mu.fill(0);

        unsigned int nclass = 0;

        for (unsigned int isample = 0; isample < n; isample++)
        {
          if (labels[isample] == iclass)
          {
            mu += m_Samples.get_row(isample);
            nclass++;
          }
        }

        if (nclass > 0)
          mu /= nclass;

        means.set_row(iclass, mu);
      } // for iclass

    } // EM loop
  } // if-else UseKdTree

  // Compute sum of within-cluster distances
  double sumdist = 0;
  for (unsigned int iclass = 0; iclass < k; iclass++)
  {
    VectorType y = means.get_row(iclass);
    for (unsigned int isample = 0; isample < n; isample++)
    {
      if (labels[isample] == iclass)
      {
        VectorType d = y - m_Samples.get_row(isample);
        sumdist += d.squared_magnitude();
      }
    }
  }

  return sumdist;

}
