
////////////////////////////////////////////////////////////////////////////////
//
// Clustering using Minimum Spanning Tree edge breaking
// MST constructed using Prim's algorithm (best for dense graphs)
//
// Designed to be used iteratively
//
// Follows the heuristic described in:
// Cocosco, C.A., Zijdenbos, A.P., Evans, A.C.: A fully automatic and robust
// brain MRI tissue classification method. Medical Image Analysis 7 (2003)
// 513-527
//
////////////////////////////////////////////////////////////////////////////////

// prastawa@cs.unc.edu 8/2004

#ifndef _PrimMSTClusteringProcess_h
#define _PrimMSTClusteringProcess_h

#include "DynArray.h"
#include "MSTEdge.h"

#include "vnl/vnl_vector.h"

class PrimMSTClusteringProcess
{

public:

  PrimMSTClusteringProcess();
  ~PrimMSTClusteringProcess();

  typedef vnl_vector<double> VertexType;
  typedef MSTEdge EdgeType;

  typedef DynArray<VertexType> VertexList;

  void SetInputVertices(const VertexList& l);

  // Break edges with threshold value T, and then cluster based on MST edge
  // connectivity
  //
  // Returns the number of clusters and fills the cluster map array
  unsigned int GetClusters(unsigned int* maps, double T);

  // Sort clusters based on size?
  inline void SortOn() { m_SortFlag = true; }
  inline void SortOff() { m_SortFlag = false; }

private:

  unsigned int m_NumberOfVertices;

  double* m_NodeAverages;

  MSTEdge* m_MSTEdges;

  bool m_SortFlag;

};


#endif
