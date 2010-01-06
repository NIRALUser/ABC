
////////////////////////////////////////////////////////////////////////////////
//
// Robust estimation using the fast MCD algorithm
//
// Rousseeuw, P.J., Van Driessen, K.: A fast algorithm for the minimum
// covariance determinant estimator. Technometrics 41 (1999) 212-223
//
////////////////////////////////////////////////////////////////////////////////

// prastawa@cs.unc.edu 3/2004

#ifndef _FastMCDSampleFilter_h
#define _FastMCDSampleFilter_h

#include "vnl/vnl_matrix.h"
#include "vnl/vnl_vector.h"
#include "vnl/algo/vnl_matrix_inverse.h"

#include "DynArray.h"

class FastMCDSampleFilter
{

public:

  typedef float SampleScalarType;
  typedef vnl_matrix<SampleScalarType> SampleMatrixType;
  typedef vnl_vector<SampleScalarType> SampleVectorType;

  typedef float ScalarType;
  typedef vnl_matrix<ScalarType> MatrixType;
  typedef vnl_vector<ScalarType> VectorType;
  typedef vnl_matrix_inverse<ScalarType> MatrixInverseType;

  typedef DynArray<unsigned int> IndexList;

  FastMCDSampleFilter();
  ~FastMCDSampleFilter();

  SampleMatrixType GetInliers(const SampleMatrixType& samples, double mahaThres);

  void GetRobustEstimate(MatrixType& mean, MatrixType& covariance,
    const SampleMatrixType& samples);
  void GetRobustEstimate(VectorType& mean, MatrixType& covariance,
    const SampleMatrixType& samples);

  void SetChangeTolerance(double f) { m_ChangeTolerance = f; }

  // Defaults is 0.5
  void SetCoverFraction(double f) { m_CoverFraction = f; }

  void SetMaxCStepIterations(unsigned int n);

  void SetNumberOfStarts(unsigned int n) { m_NumberOfStarts = n; }

private:

  double _CSteps(const SampleMatrixType& samples, const IndexList& indices,
    unsigned int maxIters);

  void _ComputeEllipsoid(MatrixType& mean, MatrixType& covariance,
    const SampleMatrixType& samples, const IndexList& indices);

  double m_ChangeTolerance;

  double m_CoverFraction;

  unsigned int m_MaxCStepIterations;

  unsigned int m_NumberOfStarts;

};

#endif
