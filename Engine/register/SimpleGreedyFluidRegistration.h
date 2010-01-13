

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
  typedef typename ImageType::SpacingType ImageSpacingType;

  typedef itk::Vector<float, Dimension> DisplacementType;
  typedef itk::Image<DisplacementType, Dimension> DeformationFieldType;
  typedef typename DeformationFieldType::Pointer DeformationFieldPointer;

  typedef itk::Image<unsigned char, Dimension> MaskType;
  typedef typename MaskType::Pointer MaskPointer;

  void SetFixedImages(const DynArray<ImagePointer>& images);
  void SetMovingImages(const DynArray<ImagePointer>& images);

  void SetMask(MaskPointer m);

  itkSetMacro(KernelWidth, double);
  itkSetMacro(Iterations, unsigned int);
  itkSetMacro(MaxStep, double);

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

  void Step();

  unsigned int m_Iterations;
  double m_MaxStep;
  double m_KernelWidth;

  double m_Delta;

  DynArray<ImagePointer> m_FixedImages;
  DynArray<ImagePointer> m_MovingImages;

  MaskPointer m_Mask;

  DynArray<ImagePointer> m_OutputImages;

  DeformationFieldPointer m_DeformationField;
  DeformationFieldPointer m_DisplacementField;

  bool m_Modified;

};

#ifndef MU_MANUAL_INSTANTIATION
#include "SimpleGreedyFluidRegistration.txx"
#endif

#endif
