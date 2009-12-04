
////////////////////////////////////////////////////////////////////////////////
//
// Clustering using Minimum Spanning Tree edge breaking
// MST constructed using Kruskal's algorithm (best for sparse graphs)
//
// Designed to be used iteratively
//
// Follows the heuristic described in:
// Cocosco, C.A., Zijdenbos, A.P., Evans, A.C.: A fully automatic and robust
// brain MRI tissue classification method. Medical Image Analysis 7 (2003)
// 513-527
//
////////////////////////////////////////////////////////////////////////////////

// prastawa@cs.unc.edu 3/2004

#ifndef _KruskalMSTClusteringProcess_h
#define _KruskalMSTClusteringProcess_h

#include "DynArray.h"
#include "MSTEdge.h"

#include "vnl/vnl_vector.h"

class KruskalMSTClusteringProcess
{

public:

  KruskalMSTClusteringProcess();
  ~KruskalMSTClusteringProcess();

  typedef vnl_vector<float> VertexType;
  typedef MSTEdge EdgeType;

  typedef DynArray<VertexType> VertexList;

  void SetInputVertices(const VertexList& l);

  // Break edges with threshold value T, and then cluster based on MST edge
  // connectivity
  //
  // Returns the number of clusters and fills the cluster map array
  unsigned int GetClusters(unsigned int* maps, float T);

  inline void SortOn() { m_SortFlag = true; }
  inline void SortOff() { m_SortFlag = false; }

private:

  unsigned int m_NumberOfVertices;

  MSTEdge* m_MSTEdges;

  float* m_NodeAverages;

  bool m_SortFlag;

};


#endif
