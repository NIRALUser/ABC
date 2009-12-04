
#include "DynArray.h"
#include "DiagonalKernelDensityEstimator.h"
#include "KMeansEstimator.h"
#include "KNNClassifier.h"

#include "MersenneTwisterRNG.h"

#include <iostream>
#include <math.h>
#include <stdlib.h>

int
_real_main(int argc, char** argv)
{

  //srand(time(NULL));

  MersenneTwisterRNG* rng = MersenneTwisterRNG::GetGlobalInstance();

  unsigned int n = 400;

  // The diagonal non-spherical version
  DiagonalKernelDensityEstimator diagEst;

  DiagonalKernelDensityEstimator::SampleMatrixType hInput2(200, 2);
  for (int i = 0; i < hInput2.rows(); i++)
  {
    hInput2(i, 0) = rng->GenerateNormal(0, 3.0);
    hInput2(i, 1) = rng->GenerateNormal(0, 1.0);
  }
  //std::cout << "Input for diag kernel test = \n" << hInput2 << std::endl;

  diagEst.SetInputSamples(hInput2);
  //diagEst.SetMinClusterCount(hInput2.rows() / 20 + 5);
  diagEst.SetMinClusterCount(10);
  diagEst.SetTruncationDistance(4.0);
  diagEst.SetMinWidth(0.01);
  diagEst.SetBurnInIterations(10);
  diagEst.SetMaximumIterations(100);

  DiagonalKernelDensityEstimator::VectorType hh1(2);
  hh1[0] = 0.1;
  hh1[1] = 0.1;
  DiagonalKernelDensityEstimator::VectorType hh2(2);
  hh2[0] = 1.0;
  hh2[1] = 1.0;
  DiagonalKernelDensityEstimator::VectorType hh3(2);
  hh3[0] = 5.0;
  hh3[1] = 5.0;
  std::cout << "h = " << diagEst.ComputeKernelBandwidth(hh1, 2.0) << std::endl;
  std::cout << "h = " << diagEst.ComputeKernelBandwidth(hh2, 2.0) << std::endl;
  std::cout << "h = " << diagEst.ComputeKernelBandwidth(hh3, 2.0) << std::endl;

  hh2[0] = 1.3;
  hh2[1] = 0.5;
  for (unsigned int i = 0; i < 5; i++)
  {
    DiagonalKernelDensityEstimator::SampleVectorType x(2);
    x[0]  = rng->GenerateNormal(0, 4.0);
    x[1]  = rng->GenerateNormal(0, 2.0);

    std::cout << "p of " << x << std::endl;
    std::cout << "  full = " << diagEst.ComputeDensity(hh2, x) << std::endl;
    std::cout << "  trunc = " << diagEst.ComputeTruncatedDensity(hh2, x) << std::endl;
  }

  double sumDIFF = 0;
  for (unsigned int i = 0; i < 100; i++)
  {
    DiagonalKernelDensityEstimator::SampleVectorType x(2);
    x[0]  = rng->GenerateNormal(0, 4.0);
    x[1]  = rng->GenerateNormal(0, 2.0);

    double pfull = diagEst.ComputeDensity(hh2, x);
    double ptrunc = diagEst.ComputeTruncatedDensity(hh2, x);
    sumDIFF += fabs(pfull - ptrunc);
  }

  std::cout << "Sum trunc diff = " << sumDIFF << std::endl;
  std::cout << "Average trunc diff = " << sumDIFF / 100 << std::endl;

  return 0;
  

}

int
main(int argc, char** argv)
{

  try
  {
    int r = _real_main(argc, argv);
    return r;
  }
/*
  catch (itk::ExceptionObject& e)
  {
    std::cerr << e << std::endl;
    return -1;
  }
*/
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << std::endl;
    return -1;
  }
  catch (std::string& s)
  {
    std::cerr << "Exception: " << s << std::endl;
    return -1;
  }
  catch (...)
  {
    std::cerr << "Unknown exception" << std::endl;
    return -1;
  }

  return 0;

}
