

/*******************************************************************************

  Simplified greedy fluid warping that maximizes data likelihood

*******************************************************************************/
// prastawa@sci.utah.edu 7/2013

#ifndef _MaxLikelihoodFluidWarpEstimator_h
#define _MaxLikelihoodFluidWarpEstimator_h

#include "itkImage.h"
#include "itkVector.h"

#include "DynArray.h"

template <class TPixel, unsigned int Dimension>
class MaxLikelihoodFluidWarpEstimator : public itk::Object
{

public:

  /** Standard class typedefs. */
  typedef MaxLikelihoodFluidWarpEstimator  Self;
  typedef itk::SmartPointer<Self>  Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  typedef itk::Image<TPixel, Dimension> ImageType;
  typedef typename ImageType::IndexType ImageIndexType;
  typedef typename ImageType::Pointer ImagePointer;
  typedef typename ImageType::PointType ImagePointType;
  typedef typename ImageType::RegionType ImageRegionType;
  typedef typename ImageType::SizeType ImageSizeType;
  typedef typename ImageType::SpacingType ImageSpacingType;

  typedef itk::Vector<float, Dimension> VectorType;
  typedef itk::Image<VectorType, Dimension> VectorFieldType;
  typedef typename VectorFieldType::Pointer VectorFieldPointer;

  typedef itk::Image<unsigned char, Dimension> MaskType;
  typedef typename MaskType::Pointer MaskPointer;

  void SetLikelihoodImages(const DynArray<ImagePointer>& images);
  void SetPriorImages(const DynArray<ImagePointer>& images);

  void SetMask(MaskPointer m);

  itkSetMacro(NumberOfScales, unsigned int);

  itkSetMacro(Iterations, unsigned int);
  itkSetMacro(MaxStep, double);

  itkSetMacro(KernelWidth, double);

  itkSetMacro(RegularityWeight, double);

  void SetInitialMomenta(VectorFieldPointer A)
  { m_InitialMomenta = A; }

  VectorFieldPointer GetMomenta()
  { if (m_Modified) this->Update(); return m_Momenta; }
  VectorFieldPointer GetVelocity()
  { if (m_Modified) this->Update(); return m_Velocity; }

  // TODO GetInverseMapping ?

  DynArray<ImagePointer> GetWarpedPriorImages()
  { if (m_Modified) this->Update(); return m_WarpedPriorImages; }

  void Update();

protected:

  MaxLikelihoodFluidWarpEstimator();
  ~MaxLikelihoodFluidWarpEstimator();

  double ComputeObjective(VectorFieldPointer v);
  void ComputeGradient();

  //VectorFieldPointer DeformationToDisplacement(VectorFieldPointer h);

  void ScaleLikelihoodImages();

  VectorFieldPointer ApplyKernel(VectorFieldPointer img);
  VectorFieldPointer ApplyKernelFlip(VectorFieldPointer img);

  ImagePointer DownsampleImage(ImagePointer img, ImageSizeType sz, ImageSpacingType sp);
  ImagePointer UpsampleImage(ImagePointer img, ImageSizeType sz, ImageSpacingType sp);

  //MaskPointer DownsampleMask(MaskPointer img, ImageSizeType sz, ImageSpacingType sp);
  //MaskPointer UpsampleMask(MaskPointer img, ImageSizeType sz, ImageSpacingType sp);

  VectorFieldPointer DownsampleDisplacement(VectorFieldPointer img, ImageSizeType sz, ImageSpacingType sp);

  VectorFieldPointer UpsampleDisplacement(VectorFieldPointer img, ImageSizeType sz, ImageSpacingType sp);

  bool Step();

  unsigned int m_Iterations;
  double m_MaxStep;

  double m_KernelWidth;

  double m_CurrentKernelWidth;

  double m_RegularityWeight;

  double m_Delta;

  DynArray<ImagePointer> m_LikelihoodImages;
  DynArray<ImagePointer> m_PriorImages;

  DynArray<ImagePointer> m_DownLikelihoodImages;
  DynArray<ImagePointer> m_DownPriorImages;

  MaskPointer m_Mask;

  DynArray<ImagePointer> m_WarpedPriorImages;

  VectorFieldPointer m_Momenta;

  VectorFieldPointer m_InitialMomenta;

  VectorFieldPointer m_Velocity;

  VectorFieldPointer m_GradientMomenta;

  bool m_Modified;

  unsigned int m_NumberOfScales;
  DynArray<ImageSizeType> m_MultiScaleSizes;
  DynArray<ImageSpacingType> m_MultiScaleSpacings;

};

#ifndef MU_MANUAL_INSTANTIATION
#include "MaxLikelihoodFluidWarpEstimator.txx"
#endif

#endif
