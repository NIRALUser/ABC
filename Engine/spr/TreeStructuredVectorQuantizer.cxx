
#include "TreeStructuredVectorQuantizer.h"
#include "KMeansEstimator.h"

#include "itkNumericTraits.h"

#include "DynArray.h"
#include "Heap.h"
#include "Log.h"

#include "muException.h"

#include "vnl/vnl_math.h"

// Macros for traversing the tree in array form
// Root is 0
#define TREE_PARENT(x) ((x) / 2)
#define TREE_LEFT(x) ((x)*2 + 1)
#define TREE_RIGHT(x) ((x)*2 + 2)

// For doing argmin on code word distances
class TaggedFloat
{
public:
  TaggedFloat() { tag = 0; value = 0; }
  bool operator<(const TaggedFloat& f) const { return this->value < f.value; }
  const TaggedFloat& operator=(const TaggedFloat& f)
  { this->tag = f.tag; this->value = f.value; return *this; }
public:
  unsigned int tag;
  float value;
};

TreeStructuredVectorQuantizer
::TreeStructuredVectorQuantizer()
{
  m_MaxTreeDepth = 8;

  m_MinNodeSize = 2;

  m_ExcludedDimension = itk::NumericTraits<unsigned int>::max();

  m_Lambda = 0.5;
}

TreeStructuredVectorQuantizer
::~TreeStructuredVectorQuantizer()
{

}

void
TreeStructuredVectorQuantizer
::ConstructTree(MatrixType& samples)
{

  unsigned int n = samples.rows();
  unsigned int dim = samples.cols();

  unsigned int maxTreeSize = (unsigned int)pow(2.0, (int)m_MaxTreeDepth) - 1;
  // Tree cannot have more elements than samples
  if (n < maxTreeSize)
    maxTreeSize = n;

  // Initialize tree code words
  m_TreeCodes.Clear();
  m_TreeCodes.Allocate(maxTreeSize);

  // Initialize list of radius for each tree element
  m_TreeRadii.Clear();
  m_TreeRadii.Allocate(maxTreeSize);

  // Initialize root with overall mean
  VectorType mean(dim, 0.0);
  for (unsigned int i = 0; i < n; i++)
    mean += samples.get_row(i);
  mean /= n;
  m_TreeCodes.Append(mean);

  // Initialize root radius with largest distance to mean
  float maxDist = 0;
  for (unsigned int i = 0; i < n; i++)
  {
    VectorType dvec = mean - samples.get_row(i);
    if (m_ExcludedDimension < dim)
      dvec[m_ExcludedDimension] = 0.0;
    float dist = dvec.magnitude();
    if (dist > maxDist)
      maxDist = dist;
  }
  m_TreeRadii.Append(maxDist);

  // Label all elements as member of root node
  DynArray<unsigned int> nodeLabels;
  nodeLabels.Initialize(n, 0);

  for (unsigned int k = 0; k < maxTreeSize; k++)
  {
    if (k >= n)
      break;

    unsigned int k_left = TREE_LEFT(k);
    unsigned int k_right = TREE_RIGHT(k);

    // Gather the samples for this node
    unsigned int count = 0;
    for (unsigned int j = 0; j < n; j++)
      if (nodeLabels[j] == k)
        count++;

    if (count < m_MinNodeSize)
    {
      // Stop when we hit a level with too few elements for each node
      break;
    }

    MatrixType nodeSamples(count, dim, 0.0);

    DynArray<unsigned int> nodeSampleIndices;
    nodeSampleIndices.Allocate(count);
    
    count = 0;
    for (unsigned int j = 0; j < n; j++)
    {
      if (nodeLabels[j] == k)
      {
        nodeSampleIndices.Append(j);
        nodeSamples.set_row(count, samples.get_row(j));
        // Make sure distance ignores ExcludedDimension, generated Kmean
        // should not be used due to this modification
        if (m_ExcludedDimension < dim)
          nodeSamples(count, m_ExcludedDimension) = 0.0;
        count++;
      }
    }

    // Split into two
    KMeansEstimator kmeans;
    kmeans.SetNumberOfClusters(2);
    kmeans.SetNumberOfStarts(5);
    kmeans.UseKdTreeOn();
    kmeans.SetMaximumIterations(100);
    kmeans.SetInput(nodeSamples);

    // Mark samples belonging to left and right children
    DynArray<unsigned int> splitLabels = kmeans.GetLabels();

    // Compute the (complete) means for the left and right clusters
    VectorType mean_left(dim, 0.0);
    VectorType mean_right(dim, 0.0);

    unsigned int num_left = 0;
    unsigned int num_right = 0;

    for (unsigned int j = 0; j < nodeSampleIndices.GetSize(); j++)
    {
      unsigned int index = nodeSampleIndices[j];
      if (splitLabels[j] == 0)
      {
        mean_left += samples.get_row(index);
        num_left++;
      }
      else
      {
        mean_right += samples.get_row(index);
        num_right++;
      }
    }

    mean_left /= num_left;
    mean_right /= num_right;

    // Compute radius for backtracking
    float rad_left = 0;
    float rad_right = 0;

    for (unsigned int j = 0; j < nodeSampleIndices.GetSize(); j++)
    {
      unsigned int index = nodeSampleIndices[j];
      if (splitLabels[j] == 0)
      {
        nodeLabels[index] = k_left;
        VectorType dvec = mean_left - samples.get_row(index);
        if (m_ExcludedDimension < dim)
          dvec[m_ExcludedDimension] = 0.0;
        float dist = dvec.magnitude();
        if (dist > rad_left)
          rad_left = dist;
      }
      else
      {
        nodeLabels[index] = k_right;
        VectorType dvec = mean_right - samples.get_row(index);
        if (m_ExcludedDimension < dim)
          dvec[m_ExcludedDimension] = 0.0;
        float dist = dvec.magnitude();
        if (dist > rad_right)
          rad_right = dist;
      }
    }

    // Allocate extra space if needed
    unsigned int treeSize = m_TreeCodes.GetSize();
    if (treeSize <= k_right)
    {
      for (unsigned int j = 0; j < (k_right-treeSize+1); j++)
      {
        m_TreeCodes.Append(VectorType(dim, 0.0));
        m_TreeRadii.Append(0.0);
      }
    }

    // Assign centroid as code words
    m_TreeCodes[k_left] = mean_left;
    m_TreeCodes[k_right] = mean_right;

    // Assign radius
    m_TreeRadii[k_left] = rad_left;
    m_TreeRadii[k_right] = rad_right;
  }

}

TreeStructuredVectorQuantizer::VectorType
TreeStructuredVectorQuantizer
::GetNearestMatch(const VectorType& x)
{

  // Use overall mean as initial code
  VectorType code = m_TreeCodes[0];

  unsigned int dim = code.size();

  unsigned int numCodes = m_TreeCodes.GetSize();

  Heap<TaggedFloat> distHeap;

  // Traverse tree
  unsigned int k = 0;
  while (k < numCodes)
  {
//std::cout << "TSVQ search k = " << k << " code = " << code << std::endl;
    unsigned int k_left = TREE_LEFT(k);
    unsigned int k_right = TREE_RIGHT(k);

    // Stop when we get to a leaf node
    if (k_left >= numCodes && k_right >= numCodes)
      break;

    if (k_left < numCodes)
    {
      VectorType dvec_left = m_TreeCodes[k_left] - x;
      if (m_ExcludedDimension < dim)
        dvec_left[m_ExcludedDimension] = 0.0;
      float rad_left = m_TreeRadii[k_left];
      float dist_left = dvec_left.magnitude() - m_Lambda*rad_left;

      TaggedFloat leftT;
      leftT.tag = k_left;
      leftT.value = dist_left;

      distHeap.Insert(leftT);
    }

    if (k_right < numCodes)
    {
      VectorType dvec_right = m_TreeCodes[k_right] - x;
      if (m_ExcludedDimension < dim)
        dvec_right[m_ExcludedDimension] = 0.0;
      float rad_right = m_TreeRadii[k_right];
      float dist_right = dvec_right.magnitude() - m_Lambda*rad_right;

      TaggedFloat rightT;
      rightT.tag = k_right;
      rightT.value = dist_right;

      distHeap.Insert(rightT);
    }

    // Select code w/ shortest distance from test set, can backtrack 
    TaggedFloat minT = distHeap.ExtractMinimum();
    k = minT.tag;
    code = m_TreeCodes[k];

  } // while k

  return code;
}

/*
TreeStructuredVectorQuantizer::VectorType
TreeStructuredVectorQuantizer
::GetNearestMatch(const VectorType& x)
{
  VectorType nearest;
  DynArray<unsigned int> nearest_ids;

  this->GetNearestMatchAndSampleID(x, nearest, nearest_ids);

  return nearest;
}

DynArray<unsigned int>
TreeStructuredVectorQuantizer
::GetNearestSampleID(const VectorType& x)
{
  VectorType nearest;
  DynArray<unsigned int> nearest_ids;

  this->GetNearestMatchAndSampleID(x, nearest, nearest_ids);

  return nearest_ids;
}
*/
