
#ifndef _PairRegistrationMethod_txx
#define _PairRegistrationMethod_txx

#include "PairRegistrationMethod.h"

#include "itkCommand.h"
#include "itkDemonsRegistrationFilter.h"
#include "itkHistogramMatchingImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageMaskSpatialObject.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkMattesMutualInformationImageToImageMetric.h"
#include "itkMultiResolutionImageRegistrationMethod.h"
#include "itkRescaleIntensityImageFilter.h"

#include "itkLBFGSBOptimizer.h"

#include "vnl/vnl_math.h"

#include "AmoebaOptimizer.h"
#include "PairRegistrationMethod.h"
#include "NegativeHCImageMatchMetric.h"
#include "NegativeMIImageMatchMetric.h"
#include "GradientDescentOptimizer.h"
#include "PowellOptimizer.h"
#include "RegistrationParameters.h"
#include "SimulatedAnnealingOptimizer.h"

#include "DynArray.h"
#include "Log.h"
#include "muException.h"

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

#include <math.h>
#include <stdlib.h>

#define PAIRREG_LINE_MAX 1024

// Observer for the affine optimizer iterations
class AffineIterationUpdate : public itk::Command
{
public:
  typedef  AffineIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef  itk::SmartPointer<Self>  Pointer;
  itkNewMacro( Self );
protected:
  AffineIterationUpdate() { }
public:
  void Execute(itk::Object *caller, const itk::EventObject & event)
  {
    Execute( (const itk::Object *)caller, event);
  }
  void Execute(const itk::Object * object, const itk::EventObject & event)
  {
    if (!itk::IterationEvent().CheckEvent(&event))
      return;
    const PowellOptimizer* powell =
      dynamic_cast<const PowellOptimizer*>(object);
    if (powell != 0)
    {
      muLogMacro(
        << "  Iter: " << powell->GetCurrentIteration() << " ||  "
        << "-MI: " << powell->GetValue() << "\n");
      muLogMacro(<< "  " << powell->GetCurrentPosition() << "\n");
      return;
    }
    const AmoebaOptimizer* amoeba =
      dynamic_cast<const AmoebaOptimizer*>(object);
    if (amoeba != 0)
    {
      muLogMacro(
        << "  Iter: " << amoeba->GetCurrentIteration() << " ||  "
        << "-MI: " << amoeba->GetValue() << "\n");
      muLogMacro(<< "  " << amoeba->GetCurrentPosition() << "\n");
      return;
    }
    const GradientDescentOptimizer* descent =
      dynamic_cast<const GradientDescentOptimizer*>(object);
    if (descent != 0)
    {
      muLogMacro(
        << "  Iter: " << descent->GetCurrentIteration() << " ||  "
        << "-MI: " << descent->GetValue() << "\n");
      muLogMacro(<< "  " << descent->GetCurrentPosition() << "\n");
      return;
    }
    const SimulatedAnnealingOptimizer* anneal =
      dynamic_cast<const SimulatedAnnealingOptimizer*>(object);
    if (anneal != 0)
    {
      muLogMacro(
        << "  Iter: " << anneal->GetCurrentIteration() << " ||  "
        << "-MI: " << anneal->GetValue() << "\n");
      muLogMacro(<< "  " << anneal->GetCurrentPosition() << "\n");
      return;
    }
  }
};

// Update parameters at change of resolution level
template <class TRegistration>
class AffineLevelUpdate : public itk::Command
{
public:
  typedef  AffineLevelUpdate   Self;
  typedef  itk::Command                   Superclass;
  typedef  itk::SmartPointer<Self>        Pointer;
  itkNewMacro( Self );
protected:
  AffineLevelUpdate() {};
public:
  typedef   TRegistration                              RegistrationType;
  typedef   RegistrationType *                         RegistrationPointer;
  //typedef   itk::RegularStepGradientDescentOptimizer   OptimizerType;
  //typedef   itk::LBFGSBOptimizer   OptimizerType;
  typedef   PowellOptimizer   OptimizerType;
  //typedef   GradientDescentOptimizer   OptimizerType;
  typedef   OptimizerType *                            OptimizerPointer;
  void Execute(itk::Object * object, const itk::EventObject & event)
  {
    if( typeid( event ) != typeid( itk::IterationEvent ) )
    {
      return;
    }
    RegistrationPointer registration =
                        dynamic_cast<RegistrationPointer>( object );
    //OptimizerPointer optimizer = dynamic_cast< OptimizerPointer >(
    //                   registration->GetOptimizer() );

    muLogMacro(<< "  Registration at level "
      << registration->GetCurrentLevel() + 1 << "\n");

    if ( registration->GetCurrentLevel() == 0 )
    {
      //optimizer->SetMaximumStepLength(10.0);
      //optimizer->SetMinimumStepLength(1.0);
    }
    else
    {
      //optimizer->SetLearningRate(optimizer->GetLearningRate() / 2.0);
      //optimizer->SetNumberOfIterations(
      //  optimizer->GetNumberOfIterations() +
      //  registration->GetCurrentLevel()*500);
      //optimizer->SetMaximumStepLength(optimizer->GetCurrentStepLength());
      //optimizer->SetMinimumStepLength(optimizer->GetMinimumStepLength() / 10.0);
      //optimizer->SetStepLength(optimizer->GetStepLength() / 10.0);
    }
  }
  void Execute(const itk::Object * , const itk::EventObject & )
  { return; }
};

// Observer for the BSpline deformable registration
class BSplineIterationUpdate : public itk::Command
{
public:
  typedef  BSplineIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef itk::SmartPointer<Self>  Pointer;
  itkNewMacro( Self );
protected:
  BSplineIterationUpdate() {};
public:
  typedef itk::LBFGSBOptimizer     OptimizerType;
  typedef const OptimizerType*    OptimizerPointer;

  void Execute(itk::Object *caller, const itk::EventObject & event)
  {
    Execute( (const itk::Object *)caller, event);
  }

  void Execute(const itk::Object * object, const itk::EventObject & event)
  {
    if( typeid( event ) != typeid( itk::IterationEvent ) )
    {
      return;
    }
    OptimizerPointer optimizer =
      dynamic_cast< OptimizerPointer >( object );
if (optimizer != 0)
{
    muLogMacro(
      << "  B-spline iter: " << optimizer->GetCurrentIteration()+1 << " ||  "
      << "-MI: " << optimizer->GetValue());
    muLogMacro(
      << " || max(grad) = " << optimizer->GetInfinityNormOfProjectedGradient()
      << "\n");
}
const AmoebaOptimizer* amoeba = dynamic_cast<const AmoebaOptimizer*>(object);
if (amoeba != 0)
  muLogMacro(
    << "  B-spline iter: " << amoeba->GetCurrentIteration()+1 << " ||  "
    << "-MI: " << amoeba->GetValue() << "\n");
  }
};


// Observer for the Demons deformable registration
class DemonsIterationUpdate : public itk::Command
{
public:
  typedef  DemonsIterationUpdate   Self;
  typedef  itk::Command             Superclass;
  typedef  itk::SmartPointer<Self>  Pointer;
  itkNewMacro(Self);
protected:
  DemonsIterationUpdate() { m_Iterations = 0; };

  typedef itk::Image<float, 3> InternalImageType;
  typedef itk::Vector<float, 3> VectorPixelType;
  typedef itk::Image<VectorPixelType, 3> DeformationFieldType;

  typedef itk::DemonsRegistrationFilter<
    InternalImageType,
    InternalImageType,
    DeformationFieldType> RegistrationFilterType;

private:
  unsigned int m_Iterations;

public:
  void Execute(itk::Object *caller, const itk::EventObject & event)
  {
    Execute( (const itk::Object *)caller, event);
  }
  void Execute(const itk::Object * object, const itk::EventObject & event)
  {
    const RegistrationFilterType * filter =
      dynamic_cast< const RegistrationFilterType * >( object );
    if( typeid( event ) != typeid( itk::IterationEvent ) )
    {
      return;
    }
    m_Iterations++;
    muLogMacro(
      << "  Demons iteration " << m_Iterations << ", metric = "
      << filter->GetMetric() << "\n");
  }
};

template <class TPixel>
void
PairRegistrationMethod<TPixel>
::ReadNextLine(char* s, std::ifstream& infile)
{
  while (true)
  {
    infile.getline(s, PAIRREG_LINE_MAX);

    if (infile.fail())
      break;

    if (s[0] != '#')
      break;
  }
}

template <class TPixel>
PairRegistrationMethod<TPixel>::AffineTransformType::Pointer
PairRegistrationMethod<TPixel>
::RegisterAffine(ImageType* fixedImg, ImageType* movingImg,
  QuantizationOption qopt)
{
  if (fixedImg == NULL || movingImg == NULL)
    muExceptionMacro(<< "One of input images is NULL");

  // Get image info
  typename ImageType::SizeType fixedSize =
    fixedImg->GetLargestPossibleRegion().GetSize();
  typename ImageType::SizeType movingSize =
    movingImg->GetLargestPossibleRegion().GetSize();

  typename ImageType::PointType fixedOrigin = fixedImg->GetOrigin();
  typename ImageType::PointType movingOrigin = movingImg->GetOrigin();

  typename ImageType::SpacingType fixedSpacing = fixedImg->GetSpacing();
  typename ImageType::SpacingType movingSpacing = movingImg->GetSpacing();

  typename ImageType::IndexType fixedCenterIndex;
  for (unsigned int dim = 0; dim < 3; dim++)
    fixedCenterIndex[dim] = (fixedSize[dim]-1) / 2;
  typename ImageType::IndexType movingCenterIndex;
  for (unsigned int dim = 0; dim < 3; dim++)
    movingCenterIndex[dim] = (movingSize[dim]-1) / 2;

  AffineTransformType::CenterType fixedCenter;
  fixedImg->TransformIndexToPhysicalPoint(fixedCenterIndex, fixedCenter);

  AffineTransformType::CenterType movingCenter;
  movingImg->TransformIndexToPhysicalPoint(movingCenterIndex, movingCenter);

  // Adjust parameters relative to fixed image size
  double minSpacing = fixedSpacing[0];
  for (unsigned int dim = 1; dim < 3; dim++)
    if (fixedSpacing[dim] < minSpacing)
      minSpacing = fixedSpacing[dim];

  // Define framework
  typedef itk::LinearInterpolateImageFunction<
    ImageType, double> InterpolatorType;
  //typedef NegativeMIImageMatchMetric<ImageType, ImageType> MetricType;    
  typedef NegativeHCImageMatchMetric<ImageType, ImageType> MetricType;    
  //typedef itk::MattesMutualInformationImageToImageMetric<
  //  ImageType, ImageType> MetricType;
  
  typedef itk::MultiResolutionImageRegistrationMethod<
    ImageType, ImageType> RegistrationType;

  // Create objects
  PowellOptimizer::Pointer powell = PowellOptimizer::New();
  AmoebaOptimizer::Pointer amoeba = AmoebaOptimizer::New();
  SimulatedAnnealingOptimizer::Pointer anneal = SimulatedAnnealingOptimizer::New();
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
  typename RegistrationType::Pointer registration = RegistrationType::New();
  typename MetricType::Pointer metric = MetricType::New();

  registration->SetOptimizer(powell);
  registration->SetInterpolator(interpolator);
  registration->SetMetric(metric);

  interpolator->SetInputImage(movingImg);

  // Initial affine transform (identity, centered at mid-image)
  AffineTransformType::Pointer affine = AffineTransformType::New();
  affine->SetSourceCenter(fixedCenter[0], fixedCenter[1], fixedCenter[2]);
  affine->SetTargetCenter(movingCenter[0], movingCenter[1], movingCenter[2]);

  registration->SetTransform(affine);
  registration->SetInitialTransformParameters(affine->GetParameters());

  registration->SetFixedImage(fixedImg);
  registration->SetMovingImage(movingImg);
  registration->SetFixedImageRegion(fixedImg->GetLargestPossibleRegion());

  // Set steps for optimization
  PowellOptimizer::ParametersType steps(12);
  steps[0] = MU_AFFINE_STEP_TRANSLATE * minSpacing;
  steps[1] = MU_AFFINE_STEP_TRANSLATE * minSpacing;
  steps[2] = MU_AFFINE_STEP_TRANSLATE * minSpacing;
  steps[3] = MU_AFFINE_STEP_ROTATE;
  steps[4] = MU_AFFINE_STEP_ROTATE;
  steps[5] = MU_AFFINE_STEP_ROTATE;
  steps[6] = MU_AFFINE_STEP_SCALE;
  steps[7] = MU_AFFINE_STEP_SCALE;
  steps[8] = MU_AFFINE_STEP_SCALE;
  steps[9] = MU_AFFINE_STEP_SKEW;
  steps[10] = MU_AFFINE_STEP_SKEW;
  steps[11] = MU_AFFINE_STEP_SKEW;

  PowellOptimizer::OrderType order(12);
  order[0] = MU_AFFINE_ORDER0;
  order[1] = MU_AFFINE_ORDER1;
  order[2] = MU_AFFINE_ORDER2;
  order[3] = MU_AFFINE_ORDER3;
  order[4] = MU_AFFINE_ORDER4;
  order[5] = MU_AFFINE_ORDER5;
  order[6] = MU_AFFINE_ORDER6;
  order[7] = MU_AFFINE_ORDER7;
  order[8] = MU_AFFINE_ORDER8;
  order[9] = MU_AFFINE_ORDER9;
  order[10] = MU_AFFINE_ORDER10;
  order[11] = MU_AFFINE_ORDER11;

  powell->SetInitialSteps(steps);
  powell->SetOrder(order);
  powell->SetMaximumIterations(10);
  powell->SetBracketMaxStep(10.0);
  powell->SetUseNewDirections(false);

  amoeba->SetMaxIterations(300);
  amoeba->SetInitialSimplexDeltas(steps);
  amoeba->SetParameterTolerance(1e-3);
  amoeba->SetFunctionTolerance(1e-3);

  anneal->SetRandomWalkSteps(steps);
  anneal->SetBurnInIterations(20);
  anneal->SetMaxIterations(220);

  typename MetricType::ParametersType derivSteps(12);
  derivSteps.Fill(1e-8);
  for (int i = 0; i < 3; i++)
    derivSteps[i] = 1e-4; // Translation steps
  metric->SetDerivativeStepLengths(derivSteps);

  metric->SetNumberOfBins(200);

/*
  // Not needed for Havrda-Charvat
  metric->SetNormalized(true);
  if (qopt == QuantizeFixed || qopt == QuantizeBoth)
    metric->QuantizeFixedImageOn();
  else
    metric->QuantizeFixedImageOff();
  if (qopt == QuantizeMoving || qopt == QuantizeBoth)
    metric->QuantizeMovingImageOn();
  else
    metric->QuantizeMovingImageOff();
*/

/*
  // ITK's MI metric
  metric->SetNumberOfHistogramBins(200);
  unsigned int numSamples =
    fixedImg->GetLargestPossibleRegion().GetNumberOfPixels() / 5;
  if (numSamples > 500000)
    numSamples = 500000;
//TODO: need to be adjusted based on pyramid level??
  metric->SetNumberOfSpatialSamples(numSamples);
  metric->ReinitializeSeed( 76926294 );
*/

  // Create the Command observer and register it with the optimizers
  powell->AddObserver(itk::IterationEvent(), AffineIterationUpdate::New());
  amoeba->AddObserver(itk::IterationEvent(), AffineIterationUpdate::New());
  anneal->AddObserver(itk::IterationEvent(), AffineIterationUpdate::New());

  // Create the Command interface observer and register it with the
  // ITK registration wrapper
  typedef AffineLevelUpdate<RegistrationType> LevelUpdaterType;
  typename LevelUpdaterType::Pointer levelUpd = LevelUpdaterType::New();
  registration->AddObserver(itk::IterationEvent(), levelUpd);

  muLogMacro(<< "Beginning affine registration...\n");
#if 0
  // Use ITK framework to handle multi resolution registration
  //metric->SetSampleSpacing(minSpacing);

  registration->SetOptimizer(amoeba);

  registration->SetNumberOfLevels(3);
  registration->StartRegistration();

  affine->SetParameters(registration->GetLastTransformParameters());
#else
  // Manage the multi resolution registration here
  metric->SetFixedImage(fixedImg);
  metric->SetMovingImage(movingImg);
  metric->SetTransform(affine);

  // Start with amoeba (slow, less prone to local minima)
  muLogMacro(<< "Registering at [4x4x4]...\n");
  metric->SetSampleSpacing(4.0*minSpacing);
  amoeba->SetCostFunction(metric);
  amoeba->SetInitialPosition(affine->GetParameters());
  amoeba->StartOptimization();
  //anneal->SetCostFunction(metric);
  //anneal->SetInitialPosition(affine->GetParameters());
  //anneal->StartOptimization();

  muLogMacro(<< "Registering at [2x2x2]...\n");
  metric->SetSampleSpacing(2.0*minSpacing);
  amoeba->SetInitialPosition(amoeba->GetCurrentPosition());
  amoeba->StartOptimization();

  // Refine results using Powell's method
  muLogMacro(<< "Refining registration at [2x2x2]...\n");
  powell->SetCostFunction(metric);
  powell->SetInitialPosition(amoeba->GetCurrentPosition());
  powell->StartOptimization();

  affine->SetParameters(powell->GetCurrentPosition());

  //affine->SetParameters(amoeba->GetCurrentPosition());

/*
  // Powell only?
  muLogMacro(<< "Registering at [2x2x2]...\n");
  metric->SetSampleSpacing(2.0*minSpacing);
  powell->SetCostFunction(metric);
  powell->SetInitialPosition(affine->GetParameters());
  powell->SetMaximumIterations(10);
  powell->StartOptimization();

  muLogMacro(<< "Registering at [2x2x2]...\n");
  metric->SetSampleSpacing(2.0*minSpacing);
  powell->SetCostFunction(metric);
  powell->SetInitialPosition(powell->GetCurrentPosition());
  powell->SetMaximumIterations(5);
  powell->StartOptimization();

  affine->SetParameters(powell->GetCurrentPosition());
*/

#endif

  muLogMacro(<< "Done with affine registration\n");

  return affine;
}

template <class TPixel>
PairRegistrationMethod<TPixel>::AffineTransformType::Pointer
PairRegistrationMethod<TPixel>
::RegisterRigid(ImageType* fixedImg, ImageType* movingImg,
  QuantizationOption qopt)
{
  if (fixedImg == NULL || movingImg == NULL)
    muExceptionMacro(<< "One of input images is NULL");

  AffineTransformType::Pointer affine = RegisterAffine(fixedImg, movingImg);

  AffineTransformType::ParametersType p = affine->GetParameters();

  // Unit scale
  p[6] = 1.0;
  p[7] = 1.0;
  p[8] = 1.0;

  // Zero skew
  p[9] = 0.0;
  p[10] = 0.0;
  p[11] = 0.0;

  affine->SetParameters(p);

  return affine;
}

template <class TPixel>
PairRegistrationMethod<TPixel>::BSplineTransformType::Pointer
PairRegistrationMethod<TPixel>
::RegisterBSpline(ImageType* fixedImg, ImageType* movingImg,
  unsigned int nx, unsigned int ny, unsigned int nz,
  QuantizationOption qopt,
  MaskImageType* fixedMask)
{
  if (nx < 1 || ny < 1 || nz < 1)
    muExceptionMacro(<< "Grid size in any dimension must be >= 1");

  // Quantize images to improve histogram MI estimates
  typedef itk::Image<short, 3> IndexImageType;

  typedef KMeansQuantizeImageFilter<ImageType, IndexImageType>
    QuantizerType;
  typedef itk::RescaleIntensityImageFilter<ImageType, IndexImageType>
    RescalerType;

  unsigned int numHistogramBins = 200;

  IndexImageType::Pointer qdFixedImg;
  if (qopt == QuantizeFixed || qopt == QuantizeBoth)
  {
    typename QuantizerType::Pointer qfilter = QuantizerType::New();
    qfilter->SetInput(fixedImg);
    qfilter->SetNumberOfBins(numHistogramBins);
    qfilter->SetTrimFraction(0.01);
    qfilter->TrimAboveOff();
    qfilter->TrimBelowOff();
    qfilter->SetTrimAboveValue(numHistogramBins);
    qfilter->SetTrimBelowValue(0);
    qfilter->Update();

    qdFixedImg = qfilter->GetOutput();
  }
  else
  {
    typename RescalerType::Pointer rescaler = RescalerType::New();
    rescaler->SetInput(fixedImg);
    rescaler->SetOutputMinimum(1);
    rescaler->SetOutputMaximum(numHistogramBins);
    rescaler->Update();

    qdFixedImg = rescaler->GetOutput();
  }

  IndexImageType::Pointer qdMovingImg;
  if (qopt == QuantizeMoving || qopt == QuantizeBoth)
  {
    typename QuantizerType::Pointer qfilter = QuantizerType::New();
    qfilter->SetInput(movingImg);
    qfilter->SetNumberOfBins(numHistogramBins);
    qfilter->SetTrimFraction(0.001);
    qfilter->TrimAboveOn();
    qfilter->TrimBelowOn();
    qfilter->SetTrimAboveValue(numHistogramBins);
    qfilter->SetTrimBelowValue(0);
    qfilter->Update();

    qdMovingImg = qfilter->GetOutput();
  }
  else
  {
    typename RescalerType::Pointer rescaler = RescalerType::New();
    rescaler->SetInput(movingImg);
    rescaler->SetOutputMinimum(1);
    rescaler->SetOutputMaximum(numHistogramBins);
    rescaler->Update();

    qdMovingImg = rescaler->GetOutput();
  }

  typedef itk::MattesMutualInformationImageToImageMetric<
    IndexImageType, IndexImageType> MIMetricType;
  //typedef NegativeMIImageMatchMetric<IndexImageType, IndexImageType>
  //  MIMetricType;

  typedef itk::LinearInterpolateImageFunction<IndexImageType, double>
    InterpolatorType;

  typedef itk::LBFGSBOptimizer OptimizerType;

  typedef itk::MultiResolutionImageRegistrationMethod<
    IndexImageType, IndexImageType> RegistrationType;

  typename MIMetricType::Pointer metric = MIMetricType::New();
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
  OptimizerType::Pointer optimizer = OptimizerType::New();

  BSplineTransformType::Pointer btrafo = BSplineTransformType::New();

  typename RegistrationType::Pointer registration = RegistrationType::New();
  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);
  registration->SetInterpolator(interpolator);

  interpolator->SetInputImage(qdMovingImg);

  registration->SetTransform(btrafo);

  registration->SetFixedImage(qdFixedImg);
  registration->SetMovingImage(qdMovingImg);

  registration->SetFixedImageRegion(qdFixedImg->GetLargestPossibleRegion());

  unsigned int gridSize[3] = {nx, ny, nz};

  typedef BSplineTransformType::RegionType RegionType;
  RegionType bsplineRegion;
  // Pad grid for border regions
  BSplineTransformType::SizeType totalGridSize;
  for (unsigned int i = 0; i < 3; i++)
    totalGridSize[i] = gridSize[i] + SplineOrder;
  bsplineRegion.SetSize(totalGridSize);

  typedef BSplineTransformType::DirectionType DirectionType;
  DirectionType direction = fixedImg->GetDirection();

  typedef BSplineTransformType::OriginType OriginType;
  OriginType origin = fixedImg->GetOrigin();

  typedef BSplineTransformType::SpacingType SpacingType;
  SpacingType spacing = fixedImg->GetSpacing();

  SpacingType shift;

  SpacingType gridSpacing;

  BSplineTransformType::SizeType fixedImgSize =
    fixedImg->GetLargestPossibleRegion().GetSize();

  // Scale spacing by image and grid size
  for (unsigned int i = 0; i < 3; i++)
  {
    gridSpacing[i] =
      spacing[i] * static_cast<double>(fixedImgSize[i])  /
      static_cast<double>(gridSize[i]);
    shift[i] = floor(0.5 * SplineOrder) * gridSpacing[i] + spacing[i]*0.5 ;
  }

  // Adjust origin
  OriginType gridOrigin = origin - direction*shift;

  btrafo->SetGridDirection(direction);
  btrafo->SetGridOrigin(gridOrigin);
  btrafo->SetGridSpacing(gridSpacing);
  btrafo->SetGridRegion(bsplineRegion);

  typedef BSplineTransformType::ParametersType ParametersType;

  unsigned int numParams = btrafo->GetNumberOfParameters();

  ParametersType initp(numParams);
  initp.Fill(0.0);

  // Force assignment by value, otherwise will need to maintain p
  // outside of this function
  btrafo->SetParametersByValue(initp);

  OptimizerType::BoundSelectionType boundSelect(numParams);
  OptimizerType::BoundValueType upperBound(numParams);
  OptimizerType::BoundValueType lowerBound(numParams);

  boundSelect.Fill( 0 ); // No bounds
  //boundSelect.Fill( 2 ); // Both lower and upper bounds
  upperBound.Fill( 10.0 );
  lowerBound.Fill( -10.0 );

  optimizer->SetBoundSelection( boundSelect );
  optimizer->SetUpperBound( upperBound );
  optimizer->SetLowerBound( lowerBound );

  optimizer->SetCostFunctionConvergenceFactor(1e+7);
  optimizer->SetProjectedGradientTolerance(1e-5);
  optimizer->SetMaximumNumberOfIterations(20);
  optimizer->SetMaximumNumberOfEvaluations(100);
  optimizer->SetMaximumNumberOfCorrections(8);

  BSplineIterationUpdate::Pointer obs = BSplineIterationUpdate::New();
  optimizer->AddObserver(itk::IterationEvent(), obs);

#if 1
  unsigned int numSamples =
    fixedImg->GetLargestPossibleRegion().GetNumberOfPixels() / 5;
  if (numSamples > 500000)
    numSamples = 500000;

  registration->SetInitialTransformParameters(initp);

  metric->SetNumberOfHistogramBins(numHistogramBins);
  metric->SetNumberOfSpatialSamples(numSamples);

  metric->UseAllPixelsOn();

  // Set up mask for MI
  if (fixedMask != 0)
  {
    // Use image mask spatial object, evaluates and checks for non-zero voxels
    typedef itk::ImageMaskSpatialObject<3> ImageMaskSpatialObject;
    ImageMaskSpatialObject::Pointer imso = ImageMaskSpatialObject::New();
    imso->SetImage(fixedMask);
    metric->SetFixedImageMask(imso);
  }

  metric->ReinitializeSeed( 76926294 );

  registration->SetNumberOfLevels(1);
  registration->StartRegistration();

  btrafo->SetParametersByValue(registration->GetLastTransformParameters());

#else

  metric->SetFixedImage(qdFixedImg);
  metric->SetMovingImage(qdMovingImg);
  metric->SetTransform(btrafo);

  typename MIMetricType::ParametersType derivSteps(numParams);
  derivSteps.Fill(0.5);
  metric->SetDerivativeStepLengths(derivSteps);

  metric->SetNumberOfBins(numHistogramBins);
  metric->SetNormalized(true);

  metric->SetSampleSpacing(2.0*minSpacing);
  optimizer->SetCostFunction(metric);
  optimizer->SetInitialPosition(initp);
  optimizer->StartOptimization();

  btrafo->SetParametersByValue(optimizer->GetCurrentPosition());

/*
  AmoebaOptimizer::Pointer amoeba = AmoebaOptimizer::New();
  amoeba->SetMaxIterations(10);
  amoeba->SetParameterTolerance(1e-2);
  amoeba->SetFunctionTolerance(1e-3);

  typename MIMetricType::ParametersType deltas(numParams);
  deltas.Fill(1.0);
  amoeba->SetInitialSimplexDeltas(deltas);

  amoeba->AddObserver(itk::IterationEvent(), BSplineIterationUpdate::New());

  metric->SetSampleSpacing(4.0*minSpacing);
  amoeba->SetCostFunction(metric);
  amoeba->SetInitialPosition(initp);
  amoeba->StartOptimization();

  btrafo->SetParametersByValue(amoeba->GetCurrentPosition());
*/
#endif

  return btrafo;
}

template <class TPixel>
PairRegistrationMethod<TPixel>::DeformationFieldType::Pointer
PairRegistrationMethod<TPixel>
::RegisterDemons(
  ImageType* fixedImg, ImageType* movingImg, unsigned int numIters)
{
  typedef itk::HistogramMatchingImageFilter<ImageType, ImageType>
    MatchFilterType;
  typename MatchFilterType::Pointer matcher = MatchFilterType::New();

  matcher->SetInput(movingImg);
  matcher->SetReferenceImage(fixedImg);

  matcher->SetNumberOfHistogramLevels(1024);
  matcher->SetNumberOfMatchPoints(8);
  matcher->ThresholdAtMeanIntensityOn();

  matcher->Update();

  typedef itk::DemonsRegistrationFilter<
    ImageType, ImageType, DeformationFieldType> DemonsType;

  typename DemonsType::Pointer demons = DemonsType::New();

  demons->AddObserver(itk::IterationEvent(), DemonsIterationUpdate::New());

  demons->SetFixedImage(fixedImg);
  demons->SetMovingImage(matcher->GetOutput());

  demons->SetNumberOfIterations(numIters);
  demons->SetStandardDeviations(1.0);

  demons->Update();

  return demons->GetOutput();
}

template <class TPixel>
PairRegistrationMethod<TPixel>::AffineTransformType::Pointer
PairRegistrationMethod<TPixel>
::ReadAffineTransform(const char* fn)
{
  std::ifstream infile;
  infile.open(fn);

  if (infile.fail())
    muExceptionMacro(<< "Failed opening " << fn);

  AffineTransformType::Pointer trafo = AffineTransformType::New();

  AffineTransformType::ParametersType p = trafo->GetParameters();
  for (unsigned int i = 0; i < 12; i++)
  {
    char s[PAIRREG_LINE_MAX];
    ReadNextLine(s, infile);
    p[i] = atof(s);
  }

  AffineTransformType::CenterType sourceC;
  for (unsigned int i = 0; i < 3; i++)
  {
    char s[PAIRREG_LINE_MAX];
    ReadNextLine(s, infile);
    sourceC[i] = atof(s);
  }

  AffineTransformType::CenterType targetC;
  for (unsigned int i = 0; i < 3; i++)
  {
    char s[PAIRREG_LINE_MAX];
    ReadNextLine(s, infile);
    targetC[i] = atof(s);
  }

  bool isForwardOrder;
  {
    char s[PAIRREG_LINE_MAX];
    ReadNextLine(s, infile);
    isForwardOrder= atoi(s);
  }

  infile.close();

  trafo->SetAllParameters(
    p,
    sourceC[0], sourceC[1], sourceC[2],
    targetC[0], targetC[1], targetC[2],
    isForwardOrder);

  return trafo;
}

template <class TPixel>
void
PairRegistrationMethod<TPixel>
::WriteAffineTransform(const char* fn, const AffineTransformType* affine)
{
  std::ofstream outfile;
  outfile.open(fn);
  outfile.setf(std::ios::fixed, std::ios::floatfield);
  outfile.precision(50);

  if (outfile.fail())
    muExceptionMacro(<< "Failed opening " << fn);

  outfile << "# Chained affine transform" << std::endl;
  outfile << "# Generated using mu::register module version " <<
    MU_REGISTER_MAJORVER << "." << MU_REGISTER_MINORVER << std::endl;
  outfile << "# Compiled on " << __DATE__ << std::endl;
  outfile << "#" << std::endl;

  DynArray<std::string> commentsList;
  commentsList.Append("Translations:");
  commentsList.Append("Rotations:");
  commentsList.Append("Scalings:");
  commentsList.Append("Skews:");

  AffineTransformType::ParametersType p = affine->GetParameters();
  for (unsigned int i = 0; i < 12; i++)
  {
    if ((i % 3) == 0)
      outfile << "# " << commentsList[i/3] << std::endl;
    outfile << p[i] << std::endl;
  }

  outfile << "# Center of rotation (source): " << std::endl;
  AffineTransformType::CenterType sourceC = affine->GetSourceCenter();
  for (unsigned int i = 0; i < 3; i++)
    outfile << sourceC[i] << std::endl;

  outfile << "# Center of rotation (target): " << std::endl;
  AffineTransformType::CenterType targetC = affine->GetTargetCenter();
  for (unsigned int i = 0; i < 3; i++)
    outfile << targetC[i] << std::endl;

  outfile << "# Forward composition order?" << std::endl;
  outfile << (int)affine->IsForwardEvaluation() << std::endl;

  outfile.close();
}

template <class TPixel>
PairRegistrationMethod<TPixel>::BSplineTransformType::Pointer
PairRegistrationMethod<TPixel>
::ReadBSplineTransform(const char* fn)
{
  std::ifstream infile;
  infile.open(fn);

  if (infile.fail())
    muExceptionMacro(<< "Failed reading " << fn);

//TODO version check?

  BSplineTransformType::Pointer btrafo = BSplineTransformType::New();

  typedef BSplineTransformType::RegionType RegionType;
  RegionType bsplineRegion;

  // Read direction
//TODO

  // Read size
  BSplineTransformType::SizeType size;
  for (unsigned int i = 0; i < 3; i++)
  {
    char s[PAIRREG_LINE_MAX];
    ReadNextLine(s, infile);
    size[i] = atoi(s);
  }

  // Pad
  for (unsigned int i = 0; i < 3; i++)
  {
    size[i] += 2;
  }

  bsplineRegion.SetSize(size);

  // Read spacing
  BSplineTransformType::SpacingType spacing;
  for (unsigned int i = 0; i < 3; i++)
  {
    char s[PAIRREG_LINE_MAX];
    ReadNextLine(s, infile);
    spacing[i] = atof(s);
  }

  // Assume default origin
  typedef BSplineTransformType::OriginType OriginType;
  OriginType origin;
  for (unsigned int i = 0; i < 3; i++)
    origin[i] = -spacing[i];

// TODO
// Set direction
  btrafo->SetGridSpacing(spacing);
  btrafo->SetGridOrigin(origin);
  btrafo->SetGridRegion(bsplineRegion);

  // Read parameters
  unsigned int numParams = btrafo->GetNumberOfParameters();

  BSplineTransformType::ParametersType p(numParams);
  for (unsigned int i = 0; i < numParams; i++)
  {
    char s[PAIRREG_LINE_MAX];
    ReadNextLine(s, infile);
    p[i] = atof(s);
  }
  btrafo->SetParametersByValue(p);

  infile.close();

  return btrafo;
}

template <class TPixel>
void
PairRegistrationMethod<TPixel>
::WriteBSplineTransform(const char* fn, const BSplineTransformType* btrafo)
{
  if (btrafo == 0)
    muExceptionMacro(<< "NULL B-spline");

  if (fn == 0)
    muExceptionMacro(<< "NULL file name");

  std::ofstream outfile;
  outfile.open(fn);
  outfile.setf(std::ios::fixed, std::ios::floatfield);
  outfile.precision(50);

  if (outfile.fail())
    muExceptionMacro(<< "Error writing file: " << fn);

  outfile << "# B-spline warp transform" << std::endl;
  outfile << "# Generated using mu::register module version " <<
    MU_REGISTER_MAJORVER << "." << MU_REGISTER_MINORVER << std::endl;
  outfile << "# Compiled on " << __DATE__ << std::endl;
  outfile << "#" << std::endl;

  // Write direction
//TODO

  // Write size
  outfile << "# Grid size:" << std::endl;
  BSplineTransformType::SizeType size = btrafo->GetGridRegion().GetSize();
  for (unsigned int i = 0; i < 3; i++)
  {
    outfile << size[i]-2 << std::endl;
  }

  // Write spacing
  outfile << "# Grid spacings:" << std::endl;
  BSplineTransformType::SpacingType spacing = btrafo->GetGridSpacing();
  for (unsigned int i = 0; i < 3; i++)
  {
    outfile << spacing[i] << std::endl;
  }

  outfile << "# B-spline coefficients:" << std::endl;
  BSplineTransformType::ParametersType p = btrafo->GetParameters();
  for (unsigned int i = 0; i < p.GetSize(); i++)
  {
    outfile << p[i] << std::endl;
  }

  outfile.close();
}

template <class TPixel>
PairRegistrationMethod<TPixel>::DeformationFieldType::Pointer
PairRegistrationMethod<TPixel>
::ReadDeformationField(const char* fn)
{
  typedef itk::ImageFileReader<DeformationFieldType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();

  reader->SetFileName(fn);
  reader->Update();

  DeformationFieldType::Pointer ret = reader->GetOutput();

  return ret;
}

template <class TPixel>
void
PairRegistrationMethod<TPixel>
::WriteDeformationField(const char* fn, const DeformationFieldType* def)
{
  typedef itk::ImageFileWriter<DeformationFieldType> WriterType;
  WriterType::Pointer writer = WriterType::New();

  writer->SetInput(def);
  writer->SetFileName(fn);
  writer->Update();
}


#endif
