

/*******************************************************************************

  Simplified greedy fluid warping with L2 norm
  Can handle multimodal images, assuming the intensities are normalized

*******************************************************************************/
// prastawa@sci.utah.edu 1/2010

#ifndef _SimpleGreedyFluidRegistration_h
#define _SimpleGreedyFluidRegistration_h

#include "itkImage.h"
#include "itkVector.h"

#include "DynArray.h"

template <class TPixel, unsigned int Dimension>
class SimpleGreedyFluidRegistration : public itk::Object
{

public:

  /** Standard class typedefs. */
  typedef SimpleGreedyFluidRegistration  Self;
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

  typedef itk::Vector<float, Dimension> DisplacementType;
  typedef itk::Image<DisplacementType, Dimension> DeformationFieldType;
  typedef typename DeformationFieldType::Pointer DeformationFieldPointer;

  typedef itk::Image<unsigned char, Dimension> MaskType;
  typedef typename MaskType::Pointer MaskPointer;

  void SetFixedImages(const DynArray<ImagePointer>& images);
  void SetMovingImages(const DynArray<ImagePointer>& images);

  void SetMask(MaskPointer m);

  itkSetMacro(NumberOfScales, uint);

  itkSetMacro(Iterations, uint);
  itkSetMacro(MaxStep, double);

  itkSetMacro(KernelWidth, double);

  void SetInitialDisplacementField(DeformationFieldPointer def)
  { m_InitialDisplacementField = def; }

  DeformationFieldPointer GetDeformationField()
  { if (m_Modified) this->Update(); return m_DeformationField; }
  DeformationFieldPointer GetDisplacementField()
  { if (m_Modified) this->Update(); return m_DisplacementField; }

  DynArray<ImagePointer> GetOutputImages()
  { if (m_Modified) this->Update(); return m_OutputImages; }

  void Update();

protected:

  SimpleGreedyFluidRegistration();
  ~SimpleGreedyFluidRegistration();

  //DeformationFieldPointer DeformationToDisplacement(DeformationFieldPointer h);

  ImagePointer DownsampleImage(ImagePointer img, ImageSizeType sz, ImageSpacingType sp);
  ImagePointer UpsampleImage(ImagePointer img, ImageSizeType sz, ImageSpacingType sp);

  DeformationFieldPointer DownsampleDeformation(DeformationFieldPointer img, ImageSizeType sz, ImageSpacingType sp);

  DeformationFieldPointer UpsampleDeformation(DeformationFieldPointer img, ImageSizeType sz, ImageSpacingType sp);
  DeformationFieldPointer UpsampleDisplacement(DeformationFieldPointer img, ImageSizeType sz, ImageSpacingType sp);

  bool Step();

  uint m_Iterations;
  double m_MaxStep;

  double m_KernelWidth;

  double m_Delta;

  DynArray<ImagePointer> m_FixedImages;
  DynArray<ImagePointer> m_MovingImages;

  DynArray<ImagePointer> m_DownFixedImages;
  DynArray<ImagePointer> m_DownMovingImages;

  MaskPointer m_Mask;

  DynArray<ImagePointer> m_OutputImages;

  DeformationFieldPointer m_DeformationField;
  DeformationFieldPointer m_DisplacementField;

  DeformationFieldPointer m_InitialDisplacementField;

  bool m_Modified;

  uint m_NumberOfScales;
  DynArray<ImageSizeType> m_MultiScaleSizes;
  DynArray<ImageSpacingType> m_MultiScaleSpacings;

};

#ifndef MU_MANUAL_INSTANTIATION
#include "SimpleGreedyFluidRegistration.txx"
#endif

#endif
