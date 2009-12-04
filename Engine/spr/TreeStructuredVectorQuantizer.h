
////////////////////////////////////////////////////////////////////////////////
//
// Tree structured vector quantization
//
// Build tree with centroids in each node, allows for fast nearest point
// search
//
// Implementation of the method described in
// J-Y. Chen, C. A. Bouman, and J. P. Allebach. Fast Image Database Search
// using Tree-Structured VQ. Proc ICIP 1997, Vol 2, Pages 827-830.
//
////////////////////////////////////////////////////////////////////////////////

// prastawa@cs.unc.edu 3/2006

#ifndef _TreeStructuredVectorQuantizer_h
#define _TreeStructuredVectorQuantizer_h

#include "DynArray.h"

#include "vnl/vnl_matrix.h"
#include "vnl/vnl_vector.h"

class TreeStructuredVectorQuantizer
{

public:

  typedef double ScalarType;
  typedef vnl_matrix<ScalarType> MatrixType;
  typedef vnl_vector<ScalarType> VectorType;

  TreeStructuredVectorQuantizer();
  ~TreeStructuredVectorQuantizer();

  void ConstructTree(MatrixType& samples);

  inline void SetMaxTreeDepth(unsigned int d) { m_MaxTreeDepth = d; }
  inline unsigned int GetMaxTreeDepth() const { return m_MaxTreeDepth; }

  // The minimum number of elements represented by a tree node
  inline void SetMinNodeSize(unsigned int m) { m_MinNodeSize = m; }
  inline unsigned int GetMinNodeSize() const { return m_MinNodeSize; }

  // Exclude a particular dimension for distance computations
  inline void SetExcludedDimension(unsigned int d) { m_ExcludedDimension = d; }
  inline unsigned int GetExcludedDimension() { return m_ExcludedDimension; }

  // The lambda parameter, must be between 0 and 1
  // A value of 1 yields optimum match, but results in more backtracking
  inline void SetLambda(double f) { m_Lambda = f; }
  inline double GetLambda() const { return m_Lambda; }

  // Obtain the (approximate) nearest match from the tree
  VectorType GetNearestMatch(const VectorType& v);
/*
//TODO
  void GetNearestMatchAndSampleIDs(
    const VectorType& v,
    VectorType& nearest, DynArray<unsigned int>& nearestids);
  VectorType GetNearestMatch(const VectorType& v);
  DynArray<unsigned int> GetNearestSampleIDs(const VectorType& v);
*/

  void Update();

private:

  // Code words for each tree node
  DynArray<VectorType> m_TreeCodes;

  // Radius of the nodes (radius = max dist between centroid and elements
  // that comprise the node)
  DynArray<double> m_TreeRadii;

  unsigned int m_MaxTreeDepth;

  unsigned int m_MinNodeSize;

  unsigned int m_ExcludedDimension;

  double m_Lambda;

};

#endif
