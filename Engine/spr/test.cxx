
#include "DynArray.h"

#include "ReducedSetDensityEstimator.h"
#include "DiagonalKernelDensityEstimator.h"
#include "SphericalKernelDensityEstimator.h"

#include "MultivariateGaussian.h"

#include "KMeansEstimator.h"
#include "KNNClassifier.h"
#include "TreeStructuredVectorQuantizer.h"

#include "MersenneTwisterRNG.h"

#include <iostream>
#include <math.h>
#include <stdlib.h>

int
_real_main(int argc, char** argv)
{

  //srand(time(NULL));

  unsigned int n = 400;

  KNNClassifier::VectorType u1(2);
  KNNClassifier::VectorType u2(2);

  u1[0] = -2;
  u1[1] = -2;

  u2[0] = 2;
  u2[1] = 2;

  unsigned int nhalf = n / 2;

  // Create input set (two gaussians)
  KNNClassifier::MatrixType set(n, 2);
  DynArray<unsigned char> labels;
  labels.Initialize(n, 0);
  for (unsigned int i = 0; i < nhalf; i++)
  {
    float r1 = (float)rand() / (float)RAND_MAX - 0.5;
    float r2 = (float)rand() / (float)RAND_MAX - 0.5;

    KNNClassifier::VectorType s = u1;
    s[0] += r1;
    s[1] += r2;

    set.set_row(i, s);
    labels[i] = 1;
  }
  for (unsigned int i = nhalf; i < n; i++)
  {
    float r1 = (float)rand() / (float)RAND_MAX - 0.5;
    float r2 = (float)rand() / (float)RAND_MAX - 0.5;

    KNNClassifier::VectorType s = u2;
    s[0] += r1;
    s[1] += r2;

    set.set_row(i, s);
    labels[i] = 2;
  }

  KMeansEstimator kmeans;
  kmeans.SetNumberOfClusters(2);
  kmeans.SetNumberOfStarts(20);
  kmeans.UseKdTreeOn();
  kmeans.SetMaximumIterations(100);
  kmeans.SetInput(set);

  KMeansEstimator::MatrixType kmu = kmeans.GetMeans();

  std::cout << "K-means: " << std::endl;
  std::cout << kmu.get_row(0) << " and " << kmu.get_row(1) << std::endl;

  // Create input set (one gaussians)
  KNNClassifier::MatrixType set2(n, 2);
  for (unsigned int i = 0; i < n; i++)
  {
    float r1 = (float)rand() / (float)RAND_MAX;
    float r2 = (float)rand() / (float)RAND_MAX;

    set2(i, 0) = r1;
    set2(i, 1) = r2;
  }

  // Create test set
  KNNClassifier::MatrixType testset(7, 2);

  testset(0, 0) = 0;
  testset(0, 1) = 0;

  testset.set_row(1, u1);
  testset.set_row(2, u2);

  testset(3, 0) = -5;
  testset(3, 1) = -5;

  testset(4, 0) = 5;
  testset(4, 1) = 5;

  testset(5, 0) = -0.5;
  testset(5, 1) = 3.0;

  testset(6, 0) = 0.5;
  testset(6, 1) = -1.0;

  unsigned int q = 10;
  KNNClassifier::MatrixType testset2(q, 2);
  for (unsigned int i = 0; i < q; i++)
  {
    float r1 = (float)rand() / (float)RAND_MAX;
    float r2 = (float)rand() / (float)RAND_MAX;

    testset2(i, 0) = r1;
    testset2(i, 1) = r2;
  }

  // KNN test
  KNNClassifier knn;
  knn.SetDimension(2);
  knn.SetKNeighbors(3);
  knn.SetTrainingData(set, labels);
  for (unsigned int i = 0; i < testset.rows(); i++)
  {
    KNNClassifier::VectorType x = testset.get_row(i);
    unsigned int c1 = knn.Classify(x);
    unsigned int c2 = knn.ClassifyWithoutCondense(x);
    std::cout << x << ":\t" << c1 << " || " << c2 << std::endl;
  }

  KNNClassifier::VectorType kx(2);
  kx[0] = 0.0;
  kx[1] = 0.0;
  DynArray<double> kprobs = knn.ComputeProbabilities(kx);
  std::cout << "KNN probs for " << kx << " is: ";
  for (unsigned int i = 0; i < kprobs.GetSize(); i++)
    std::cout << " " << kprobs[i];
  std::cout << std::endl;

  // RSDE test
  ReducedSetDensityEstimator rsde;
  rsde.SetDimension(2);
  rsde.SetInputSet(set2);
  rsde.SetKernelWidth(0.2);
  //float s = 1.0 / pow(2*M_PI, d/2.0); 
  //float s = 1.0 / (2*M_PI);
  for (unsigned int i = 0; i < testset2.rows(); i++)
  {
    KNNClassifier::VectorType x = testset2.get_row(i);
    float f1 = rsde.Evaluate(x);
    float f2 = rsde.EvaluateWithoutReduce(x);

    //float fideal = s*exp(-0.5 * x.squared_magnitude());
    float fideal = 0;
    if (x[0] >= 0 && x[0] <= 1 && x[1] >= 0 && x[1] <= 1)
      fideal = 1.0;

/*
    std::cout << x << ":\t" <<  f1 << "\t|| " << f2 << "\t|| fideal = " << fideal << std::endl;
    std::cout << "DIFF =\t" << fabs(f1-f2) << "\t|| ERR1 = " << fabs(f1-fideal) << "\t|| ERR2 = " << fabs(f2-fideal) << std::endl;
*/
    std::cout << "DIFF =\t" << fabs(f1-f2) << std::endl;
  }

  // Kernel density estimator test
  MersenneTwisterRNG* rng = MersenneTwisterRNG::GetGlobalInstance();

  SphericalKernelDensityEstimator sphEst;

  SphericalKernelDensityEstimator::SampleMatrixType hInput(200, 2);
  for (int i = 0; i < hInput.rows(); i++)
  {
    hInput(i, 0) = rng->GenerateNormal(0, 2.0);
    hInput(i, 1) = rng->GenerateNormal(0, 2.0);
  }
  //std::cout << "Input for width est = \n" << hInput << std::endl;

  std::cout << "Spherical bandwidths: " << std::endl;
  std::cout << "h = " << sphEst.ComputeKernelBandwidth(hInput, 0.1) << std::endl;
  std::cout << "h = " << sphEst.ComputeKernelBandwidth(hInput, 1.0) << std::endl;
  std::cout << "h = " << sphEst.ComputeKernelBandwidth(hInput, 4.0) << std::endl;

  // The diagonal non-spherical version
  DiagonalKernelDensityEstimator diagEst;

  DiagonalKernelDensityEstimator::SampleMatrixType hInput2(200, 2);
  for (int i = 0; i < hInput2.rows(); i++)
  {
    hInput2(i, 0) = rng->GenerateNormal(0, 3.0);
    hInput2(i, 1) = rng->GenerateNormal(0, 1.0);
  }
  //std::cout << "Input for diag kernel test = \n" << hInput2 << std::endl;

std::cout << "Build diag density est" << std::endl;
  diagEst.SetMinClusterCount(hInput2.rows() / 20 + 5);
  diagEst.SetTruncationDistance(3.75);
  diagEst.SetMinWidth(0.01);
  diagEst.SetBurnInIterations(200);
  diagEst.SetMaximumIterations(2000);
  diagEst.SetInputSamples(hInput2);

  DiagonalKernelDensityEstimator::VectorType hh1(2);
  hh1[0] = 0.1;
  hh1[1] = 0.1;
  DiagonalKernelDensityEstimator::VectorType hh2(2);
  hh2[0] = 1.0;
  hh2[1] = 1.0;
  DiagonalKernelDensityEstimator::VectorType hh3(2);
  hh3[0] = 5.0;
  hh3[1] = 5.0;
  std::cout << "Diagonal bandwidths: " << std::endl;
  std::cout << "h = " << diagEst.ComputeKernelBandwidth(hh1, 2.0) << std::endl;
  std::cout << "h = " << diagEst.ComputeKernelBandwidth(hh2, 2.0) << std::endl;
  std::cout << "h = " << diagEst.ComputeKernelBandwidth(hh3, 2.0) << std::endl;

  hh2[0] = 1.3;
  hh2[1] = 0.5;
  diagEst.SetDefaultBandwidth(hh2);

  for (unsigned int i = 0; i < 5; i++)
  {
    DiagonalKernelDensityEstimator::SampleVectorType x(2);
    x[0]  = rng->GenerateNormal(0, 4.0);
    x[1]  = rng->GenerateNormal(0, 2.0);

    std::cout << "p of " << x << std::endl;
    std::cout << "  full = " << diagEst.EvaluateDensity(x) << std::endl;
    std::cout << "  trunc = " << diagEst.EvaluateTruncatedDensity(hh2, x) << std::endl;
  }

  double sumDIFF = 0;
  for (unsigned int i = 0; i < 100; i++)
  {
    DiagonalKernelDensityEstimator::SampleVectorType x(2);
    x[0]  = rng->GenerateNormal(0, 4.0);
    x[1]  = rng->GenerateNormal(0, 2.0);

    double pfull = diagEst.EvaluateDensity(x);
    double ptrunc = diagEst.EvaluateTruncatedDensity(hh2, x);
    sumDIFF += fabs(pfull - ptrunc);
  }

  std::cout << "Sum trunc diff = " << sumDIFF << std::endl;
  std::cout << "Average trunc diff = " << sumDIFF / 100 << std::endl;

  std::cout << "TSVQ test" << std::endl;

  TreeStructuredVectorQuantizer::MatrixType tsvqInput(100, 2);
  for (int i = 0; i < tsvqInput.rows(); i++)
  {
    float mu_x = 0;
    float mu_y = 0;

    if (i < (tsvqInput.rows()/3))
      mu_x = mu_y = 10;
    if (i >= (tsvqInput.rows()/3) && i <= (tsvqInput.rows()*2/3))
      mu_x = mu_y = -10;

    tsvqInput(i, 0) = rng->GenerateNormal(0, 1.0) + mu_x;
    tsvqInput(i, 1) = rng->GenerateNormal(0, 1.0) + mu_y;
  }

  TreeStructuredVectorQuantizer tsvq;

  tsvq.SetMaxTreeDepth(5);
  tsvq.ConstructTree(tsvqInput);

  TreeStructuredVectorQuantizer::VectorType testv(2, 0);

  testv[0] = 1;
  testv[1] = 1.2;
std::cout << "Code for " << testv << " is " << tsvq.GetNearestMatch(testv) << std::endl;

  testv[0] = 8;
  testv[1] = 14.2;
std::cout << "Code for " << testv << " is " << tsvq.GetNearestMatch(testv) << std::endl;

  testv[0] = -8;
  testv[1] = 8;
std::cout << "Code for " << testv << " is " << tsvq.GetNearestMatch(testv) << std::endl;

  testv[0] = -10.5;
  testv[1] = -8.9;
std::cout << "Code for " << testv << " is " << tsvq.GetNearestMatch(testv) << std::endl;

  MultivariateGaussian normDist(2);

  MultivariateGaussian::SampleMatrixType nsamples(20, 2);
  for (unsigned int i = 0; i < nsamples.rows(); i++)
  {
    nsamples(i, 0) = rng->GenerateNormal(0, 2.0) + 1.0;
    nsamples(i, 1) = rng->GenerateNormal(0, 1.0) + 2.0;
  }
  nsamples(0, 0) = 1000.0;
  nsamples(0, 1) = -20.0;

  normDist.SetParametersFromSamples(nsamples, true);

  std::cout << "Norm mean = " << normDist.GetMean() << std::endl;
  std::cout << "Norm cov = " << normDist.GetCovariance() << std::endl;

  MultivariateGaussian::SampleVectorType nx(2);
  nx[0] = 0; nx[1] = 0;
  std::cout << "Norm prob of " << nx << " = " << normDist.EvaluateDensity(nx) << std::endl;
  nx[0] = 2; nx[1] = 1;
  std::cout << "Norm prob of " << nx << " = " << normDist.EvaluateDensity(nx) << std::endl;
  nx[0] = 8; nx[1] = 8;
  std::cout << "Norm prob of " << nx << " = " << normDist.EvaluateDensity(nx) << std::endl;

  std::cout << "Norm gen " << normDist.GenerateRandomVariate() << std::endl;
  std::cout << "Norm gen " << normDist.GenerateRandomVariate() << std::endl;

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
