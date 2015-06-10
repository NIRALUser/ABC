
#ifndef _SimpleGreedyFluidRegistration_txx
#define _SimpleGreedyFluidRegistration_txx

#include "itkAddImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkSubtractImageFilter.h"

#include "itkBSplineInterpolateImageFunction.h"
#include "itkComposeImageFilter.h"
#include "itkDerivativeImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkVectorResampleImageFilter.h"
#include "itkWarpImageFilter.h"
#include "itkWarpVectorImageFilter.h"

//#include "itkDiscreteGaussianImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"
#include "VectorBlurImageFilter.h"

#include "itkImageFileWriter.h"

template <class TPixel, unsigned int Dimension>
SimpleGreedyFluidRegistration<TPixel, Dimension>
::SimpleGreedyFluidRegistration()
{
  m_Iterations = 10;
  m_MaxStep = 0.5;
  m_Delta = 0.0;
  m_KernelWidth = 1.0;
  m_NumberOfScales = 3;
  m_Modified = false;
}

template <class TPixel, unsigned int Dimension>
SimpleGreedyFluidRegistration<TPixel, Dimension>
::~SimpleGreedyFluidRegistration()
{
}

template <class TPixel, unsigned int Dimension>
void
SimpleGreedyFluidRegistration<TPixel, Dimension>
::SetFixedImages(const DynArray<ImagePointer>& images)
{
  m_FixedImages = images;
  m_Modified = true;
}

template <class TPixel, unsigned int Dimension>
void
SimpleGreedyFluidRegistration<TPixel, Dimension>
::SetMovingImages(const DynArray<ImagePointer>& images)
{
  m_MovingImages = images;
  m_Modified = true;
}

template <class TPixel, unsigned int Dimension>
void
SimpleGreedyFluidRegistration<TPixel, Dimension>
::SetMask(MaskPointer m)
{
  m_Mask = m;
  m_Modified = true;
}

template <class TPixel, unsigned int Dimension>
typename SimpleGreedyFluidRegistration<TPixel, Dimension>::ImagePointer
SimpleGreedyFluidRegistration<TPixel, Dimension>
::DownsampleImage(ImagePointer img, ImageSizeType downsize, ImageSpacingType downspacing)
{
  ImageRegionType region = img->GetLargestPossibleRegion();
  ImageSizeType size = region.GetSize();

  ImageSpacingType spacing = img->GetSpacing();

  double sigma = downspacing[0];
  for (unsigned int dim = 0; dim < Dimension; dim++)
    if (downspacing[dim] < sigma)
      sigma = downspacing[dim];

  typedef itk::SmoothingRecursiveGaussianImageFilter<ImageType, ImageType>
  //typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType>
    BlurType;
  typename BlurType::Pointer blurf = BlurType::New();
  blurf->SetInput(img);
  blurf->SetSigma(sigma);
  //blurf->SetVariance(sigma*sigma);
  blurf->Update();

  typedef itk::ResampleImageFilter<ImageType, ImageType>
    ResamplerType;
  typename ResamplerType::Pointer resf = ResamplerType::New();
  resf->SetInput(blurf->GetOutput());
  resf->SetDefaultPixelValue(0);
  resf->SetOutputDirection(img->GetDirection());
  resf->SetOutputSpacing(downspacing);
  resf->SetOutputOrigin(img->GetOrigin());
  resf->SetSize(downsize);
  resf->Update();

  return resf->GetOutput();
}

template <class TPixel, unsigned int Dimension>
typename SimpleGreedyFluidRegistration<TPixel, Dimension>::ImagePointer
SimpleGreedyFluidRegistration<TPixel, Dimension>
::UpsampleImage(ImagePointer img, ImageSizeType upsize, ImageSpacingType upspacing)
{
  typedef itk::BSplineInterpolateImageFunction<ImageType, double, double>
    SplineInterpolatorType;
  typename SplineInterpolatorType::Pointer splineInt =
    SplineInterpolatorType::New();
  splineInt->SetSplineOrder(3);

  typedef itk::ResampleImageFilter<ImageType, ImageType>
    ResamplerType;
  typename ResamplerType::Pointer resf = ResamplerType::New();
  resf->SetInput(img);
  resf->SetDefaultPixelValue(0);
  resf->SetInterpolator(splineInt);
  resf->SetOutputDirection(img->GetDirection());
  resf->SetOutputSpacing(upspacing);
  resf->SetOutputOrigin(img->GetOrigin());
  resf->SetSize(upsize);
  resf->Update();

  return resf->GetOutput();
}

template <class TPixel, unsigned int Dimension>
typename SimpleGreedyFluidRegistration<TPixel, Dimension>::DeformationFieldPointer
SimpleGreedyFluidRegistration<TPixel, Dimension>
::DownsampleDeformation(DeformationFieldPointer img, ImageSizeType downsize, ImageSpacingType downspacing)
{
  ImageRegionType region = img->GetLargestPossibleRegion();
  ImageSizeType size = region.GetSize();

  ImageSpacingType spacing = img->GetSpacing();

  typedef itk::VectorResampleImageFilter<DeformationFieldType, DeformationFieldType>
    ResamplerType;
  typename ResamplerType::Pointer resf = ResamplerType::New();
  resf->SetInput(img);
  DisplacementType zerov;
  zerov.Fill(0.0);
  resf->SetDefaultPixelValue(zerov);
  resf->SetOutputDirection(img->GetDirection());
  resf->SetOutputSpacing(downspacing);
  resf->SetOutputOrigin(img->GetOrigin());
  resf->SetSize(downsize);
  resf->Update();

  return resf->GetOutput();
}

template <class TPixel, unsigned int Dimension>
typename SimpleGreedyFluidRegistration<TPixel, Dimension>::DeformationFieldPointer
SimpleGreedyFluidRegistration<TPixel, Dimension>
::UpsampleDeformation(DeformationFieldPointer img, ImageSizeType upsize, ImageSpacingType upspacing)
{

  typedef itk::VectorResampleImageFilter<DeformationFieldType, DeformationFieldType>
    ResamplerType;
  typename ResamplerType::Pointer resf = ResamplerType::New();
  resf->SetInput(img);
  DisplacementType zerov;
  zerov.Fill(sqrt(-1.0f));
  resf->SetDefaultPixelValue(zerov);
  resf->SetOutputDirection(img->GetDirection());
  resf->SetOutputSpacing(upspacing);
  resf->SetOutputOrigin(img->GetOrigin());
  resf->SetSize(upsize);
  resf->Update();

  DeformationFieldPointer def = resf->GetOutput();

  // Set mapping to identity if it goes out of image boundaries
  typedef itk::ImageRegionIteratorWithIndex<DeformationFieldType> IteratorType;
  IteratorType it(def, def->GetLargestPossibleRegion());

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    ImageIndexType ind = it.GetIndex();

    DisplacementType h = def->GetPixel(ind);

    ImagePointType p;
    def->TransformIndexToPhysicalPoint(ind, p);

    bool isout = false;
    for (unsigned int dim = 0; dim < Dimension; dim++)
      if (vnl_math_isnan(h[dim]))
      {
        isout = true;
        break;
      }
    if (isout)
    {
      for (unsigned int dim = 0; dim < Dimension; dim++)
        h[dim] = p[dim];
      def->SetPixel(ind, h);
    }
  }

  return def;
}

template <class TPixel, unsigned int Dimension>
typename SimpleGreedyFluidRegistration<TPixel, Dimension>::DeformationFieldPointer
SimpleGreedyFluidRegistration<TPixel, Dimension>
::UpsampleDisplacement(DeformationFieldPointer img, ImageSizeType upsize, ImageSpacingType upspacing)
{

  typedef itk::VectorResampleImageFilter<DeformationFieldType, DeformationFieldType>
    ResamplerType;
  typename ResamplerType::Pointer resf = ResamplerType::New();
  resf->SetInput(img);
  DisplacementType zerov;
  zerov.Fill(0.0);
  resf->SetDefaultPixelValue(zerov);
  resf->SetOutputDirection(img->GetDirection());
  resf->SetOutputSpacing(upspacing);
  resf->SetOutputOrigin(img->GetOrigin());
  resf->SetSize(upsize);
  resf->Update();

  return resf->GetOutput();
}

template <class TPixel, unsigned int Dimension>
void
SimpleGreedyFluidRegistration<TPixel, Dimension>
::Update()
{
  if (!m_Modified)
    return;

  if (m_Iterations < 5)
    itkExceptionMacro(<< "Must have at least 5 iterations");

  if (m_FixedImages.GetSize() != m_MovingImages.GetSize())
    itkExceptionMacro(<< "Number of channels must match");

  unsigned int numChannels = m_FixedImages.GetSize();

  ImageSizeType size = m_FixedImages[0]->GetLargestPossibleRegion().GetSize();
  ImageSpacingType spacing = m_FixedImages[0]->GetSpacing();

  m_MultiScaleSizes.Clear();
  m_MultiScaleSpacings.Clear();

  m_MultiScaleSizes.Append(size);
  m_MultiScaleSpacings.Append(spacing);

  double downF = 2.0;
  for (unsigned int s = 0; s < (m_NumberOfScales-1); s++)
  {
    ImageSizeType downsize;
    for (unsigned int dim = 0; dim < Dimension; dim++)
      downsize[dim] = size[dim] / downF;
    ImageSpacingType downspacing;
    for (unsigned int dim = 0; dim < Dimension; dim++)
      downspacing[dim] =
       spacing[dim] * (double)size[dim] / (double)downsize[dim];
    m_MultiScaleSizes.Append(downsize);
    m_MultiScaleSpacings.Append(downspacing);
    downF *= 2.0;
  }

  // Initialize output images and downsample
  m_OutputImages.Clear();
  for (unsigned int c = 0; c < numChannels; c++)
  {
/*
    typedef itk::ImageDuplicator<ImageType> DuperType;
    typename DuperType::Pointer dupef = DuperType::New();
    dupef->SetInputImage(m_MovingImages[c]);
    dupef->Update();
    m_OutputImages.Append(dupef->GetOutput());
*/
    m_OutputImages.Append(
      this->DownsampleImage(m_MovingImages[c],
        m_MultiScaleSizes[m_NumberOfScales-1],
        m_MultiScaleSpacings[m_NumberOfScales-1]) );

  }

  // Initialize H-field and downsample
  m_DeformationField = DeformationFieldType::New();
  //m_DeformationField->CopyInformation(m_FixedImages[0]);
  m_DeformationField->SetDirection(m_FixedImages[0]->GetDirection());
  m_DeformationField->SetOrigin(m_FixedImages[0]->GetOrigin());
  m_DeformationField->SetSpacing(m_FixedImages[0]->GetSpacing());
  m_DeformationField->SetRegions(m_FixedImages[0]->GetLargestPossibleRegion());
  m_DeformationField->Allocate();

  DisplacementType zerov;
  zerov.Fill(0.0);
  m_DeformationField->FillBuffer(zerov);

  typedef itk::ImageRegionIteratorWithIndex<DeformationFieldType> IteratorType;

  IteratorType h0It(m_DeformationField, m_DeformationField->GetLargestPossibleRegion());
  for (h0It.GoToBegin(); !h0It.IsAtEnd(); ++h0It)
  {
    ImagePointType p;
    m_DeformationField->TransformIndexToPhysicalPoint(h0It.GetIndex(), p);
    DisplacementType v;
    for (unsigned int i = 0; i < Dimension; i++)
      v[i] = p[i];
    h0It.Set(v);
  }
  m_DeformationField =
    this->DownsampleDeformation(
      m_DeformationField,
      m_MultiScaleSizes[m_NumberOfScales-1],
      m_MultiScaleSpacings[m_NumberOfScales-1]);

  // Store displacement field as well
  m_DisplacementField = DeformationFieldType::New();
  //m_DisplacementField->CopyInformation(m_FixedImages[0]);
  m_DisplacementField->SetDirection(m_FixedImages[0]->GetDirection());
  m_DisplacementField->SetOrigin(m_FixedImages[0]->GetOrigin());
  m_DisplacementField->SetSpacing(m_FixedImages[0]->GetSpacing());
  m_DisplacementField->SetRegions(m_FixedImages[0]->GetLargestPossibleRegion());
  m_DisplacementField->Allocate();
  m_DisplacementField->FillBuffer(zerov);
  m_DisplacementField =
    this->DownsampleDeformation(
      m_DisplacementField,
      m_MultiScaleSizes[m_NumberOfScales-1],
      m_MultiScaleSpacings[m_NumberOfScales-1]);

  // Initialize using user-specified deformation if available
  if (!m_InitialDisplacementField.IsNull())
  {
    m_DisplacementField =
      this->DownsampleDeformation(
        m_InitialDisplacementField,
        m_MultiScaleSizes[m_NumberOfScales-1],
        m_MultiScaleSpacings[m_NumberOfScales-1]);

    IteratorType it(m_DeformationField, m_DeformationField->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      ImageIndexType ind = it.GetIndex();

      ImagePointType p;
      m_DeformationField->TransformIndexToPhysicalPoint(ind, p);

      DisplacementType v = m_DisplacementField->GetPixel(ind);

      DisplacementType h;
      for (unsigned int i = 0; i < Dimension; i++)
        h[i] = p[i] + v[i];
      m_DeformationField->SetPixel(ind, h);
    }

    m_OutputImages.Clear();
    for (unsigned int c = 0; c < numChannels; c++)
    {
      typedef itk::WarpImageFilter<
        ImageType, ImageType, DeformationFieldType>
        WarperType;
      typename WarperType::Pointer warpf = WarperType::New();
      warpf->SetInput(m_MovingImages[c]);
      warpf->SetEdgePaddingValue(0.0);
      warpf->SetDisplacementField(m_DisplacementField);
      warpf->SetOutputDirection(m_DisplacementField->GetDirection());
      warpf->SetOutputOrigin(m_DisplacementField->GetOrigin());
      warpf->SetOutputSpacing(m_DisplacementField->GetSpacing());
      warpf->Update();
      m_OutputImages.Append(
        this->DownsampleImage(
          warpf->GetOutput(),
          m_MultiScaleSizes[m_NumberOfScales-1],
          m_MultiScaleSpacings[m_NumberOfScales-1]) );
    }
  } // if h init exist

  for (long s = (m_NumberOfScales-1); s >= 0; s--)
  {
    ImageSizeType currsize = m_MultiScaleSizes[s];
    ImageSpacingType currspacing = m_MultiScaleSpacings[s];

    if (s == 0)
    {
      m_DownFixedImages = m_FixedImages;
      m_DownMovingImages = m_MovingImages;
    }
    else
    {
      m_DownFixedImages.Clear();
      m_DownMovingImages.Clear();
      for (unsigned int c = 0; c < numChannels; c++)
      {
        m_DownFixedImages.Append(
          this->DownsampleImage(m_FixedImages[c], currsize, currspacing) );
        m_DownMovingImages.Append(
          this->DownsampleImage(m_MovingImages[c], currsize, currspacing) );
      }
    }

    if (s < (m_NumberOfScales-1))
    {
      for (unsigned int c = 0; c < numChannels; c++)
        m_OutputImages[c] = this->UpsampleImage(
          m_OutputImages[c], currsize, currspacing); 
      m_DeformationField = this->UpsampleDeformation(
        m_DeformationField, currsize, currspacing);
      m_DisplacementField = this->UpsampleDisplacement(
        m_DisplacementField, currsize, currspacing);
    }

    // Greedy optimization
    m_Delta = 0.0;
    for (unsigned int iter = 1; iter <= m_Iterations; iter++)
    {
      bool converge = this->Step();
      if (converge)
        break;
    }

  }

  // Warp images using final deformation
  m_OutputImages.Clear();
  for (unsigned int ichan = 0; ichan < numChannels; ichan++)
  {
    typedef itk::WarpImageFilter<
      ImageType, ImageType, DeformationFieldType>
      WarperType;
    typename WarperType::Pointer warpf = WarperType::New();
    warpf->SetInput(m_MovingImages[ichan]);
    warpf->SetEdgePaddingValue(0.0);
    warpf->SetDisplacementField(m_DisplacementField);
    warpf->SetOutputDirection(m_DisplacementField->GetDirection());
    warpf->SetOutputOrigin(m_DisplacementField->GetOrigin());
    warpf->SetOutputSpacing(m_DisplacementField->GetSpacing());
    warpf->Update();
    m_OutputImages.Append(warpf->GetOutput());
  }

  m_Modified = false;
}

template <class TPixel, unsigned int Dimension>
bool
SimpleGreedyFluidRegistration<TPixel, Dimension>
::Step()
{
  typedef itk::AddImageFilter<ImageType, ImageType, ImageType> AddFilterType;
  typedef itk::MultiplyImageFilter<ImageType, ImageType, ImageType> MultiplyFilterType;
  typedef itk::SubtractImageFilter<ImageType, ImageType, ImageType> SubtractFilterType;

  unsigned int numChannels = m_DownFixedImages.GetSize();

  // Find scale adjustment
  ImageSpacingType spacing = m_DownFixedImages[0]->GetSpacing();
  double minSpacing = spacing[0];
  for (unsigned int i = 1; i < Dimension; i++)
    if (spacing[i] < minSpacing)
      minSpacing = spacing[i];

  ImageSizeType size = m_DownFixedImages[0]->GetLargestPossibleRegion().GetSize();

  DisplacementType edgev;
  //edgev.Fill(vnl_huge_val(0.0f));
  edgev.Fill(sqrt(-1.0f));
  DisplacementType zerov;
  zerov.Fill(0.0);

  // Velocity field
  // v = sum_c { (fixed_c - moving_c) * grad(moving_c) }

  DynArray<ImagePointer> velocImages;
  for (unsigned int dim = 0; dim < Dimension; dim++)
  {
    ImagePointer tmp = ImageType::New();
    //tmp->CopyInformation(m_DownFixedImages[0]);
    tmp->SetDirection(m_DownFixedImages[0]->GetDirection());
    tmp->SetOrigin(m_DownFixedImages[0]->GetOrigin());
    tmp->SetSpacing(m_DownFixedImages[0]->GetSpacing());
    tmp->SetRegions(m_DownFixedImages[0]->GetLargestPossibleRegion());
    tmp->Allocate();
    tmp->FillBuffer(0);
    velocImages.Append(tmp);
  }

  // Difference images
  DynArray<ImagePointer> diffImages;
  for (unsigned int ichan = 0; ichan < numChannels; ichan++)
  {
    typename SubtractFilterType::Pointer subf = SubtractFilterType::New();
    subf->SetInput1(m_DownFixedImages[ichan]);
    subf->SetInput2(m_OutputImages[ichan]);
    subf->Update();
    diffImages.Append(subf->GetOutput());
  }

  typedef itk::ImageRegionIteratorWithIndex<DeformationFieldType> IteratorType;

  for (unsigned int ichan = 0; ichan < numChannels; ichan++)
  {
    typedef itk::DerivativeImageFilter<ImageType, ImageType>
      DerivativeFilterType;

    for (unsigned int dim = 0; dim < Dimension; dim++)
    {
      typename DerivativeFilterType::Pointer derivf = DerivativeFilterType::New();
      derivf->SetInput(m_OutputImages[ichan]);
      derivf->SetDirection(dim);
      derivf->SetUseImageSpacingOn();
      derivf->Update();

      typename MultiplyFilterType::Pointer mulf = MultiplyFilterType::New();
      mulf->SetInput1(diffImages[ichan]);
      mulf->SetInput2(derivf->GetOutput());
      mulf->Update();

      typename AddFilterType::Pointer addf = AddFilterType::New();
      addf->SetInput1(velocImages[dim]);
      addf->SetInput2(mulf->GetOutput());
      addf->Update();
      velocImages[dim] = addf->GetOutput();
    } // for dim
  } // for ichan

  // Put together accumulated forces into a single vector image
  typedef itk::ComposeImageFilter<ImageType, DeformationFieldType> VectorComposerType;

  typename VectorComposerType::Pointer vecf = VectorComposerType::New();
  for (unsigned int dim = 0; dim < Dimension; dim++)
    vecf->SetInput(dim, velocImages[dim]);
  vecf->Update();
  
  DeformationFieldPointer velocF = vecf->GetOutput();

  // Apply Green's kernel to velocity field
  double adjustedWidth = m_KernelWidth * minSpacing;

  typedef VectorBlurImageFilter<DeformationFieldType, DeformationFieldType>
    DeformationSmootherType;
  typename DeformationSmootherType::Pointer defsmoother = DeformationSmootherType::New();
  defsmoother->SetKernelWidth(adjustedWidth);
  defsmoother->SetInput(velocF);
  defsmoother->Update();

  velocF = defsmoother->GetOutput();

  IteratorType it(velocF, velocF->GetLargestPossibleRegion());

  // Compute max velocity magnitude
  double maxVeloc = 0.0;
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    DisplacementType v = it.Get();
    double d = v.GetNorm();
    if (d > maxVeloc)
      maxVeloc = d;
  }

  // Update delta at initial step
  if (m_Delta == 0.0)
  {
    m_Delta = m_MaxStep * minSpacing / (maxVeloc + 1e-20);
  }

  // Test for convergence
  //if ((maxVeloc*m_Delta) < (1e-5*minSpacing))
  //  return true;

  double adaptDelta = m_Delta;
  //if ((maxVeloc*m_Delta) > (m_MaxStep*minSpacing))
  //  adaptDelta = m_MaxStep * minSpacing / (maxVeloc + 1e-20);

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    it.Set(it.Get() * adaptDelta);
  }

  // Compose velocity field
  // new h(x) <= h( g(x) ) where g(x) = x + v
  typedef itk::WarpVectorImageFilter<
    DeformationFieldType, DeformationFieldType, DeformationFieldType>
    ComposerType;
  typename ComposerType::Pointer compf = ComposerType::New();
  compf->SetInput(m_DeformationField);
  compf->SetDisplacementField(velocF);
  compf->SetEdgePaddingValue(edgev);
  compf->SetOutputDirection(m_DeformationField->GetDirection());
  compf->SetOutputOrigin(m_DeformationField->GetOrigin());
  compf->SetOutputSpacing(m_DeformationField->GetSpacing());
  compf->Update();

  m_DeformationField = compf->GetOutput();

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    ImageIndexType ind = it.GetIndex();

    DisplacementType h = m_DeformationField->GetPixel(ind);

    ImagePointType p;
    m_DeformationField->TransformIndexToPhysicalPoint(ind, p);

    bool isout = false;
    for (unsigned int dim = 0; dim < Dimension; dim++)
      //if (vnl_math_isinf(h[dim]))
      if (vnl_math_isnan(h[dim]))
      {
        isout = true;
        break;
      }
    if (isout)
    {
      for (unsigned int dim = 0; dim < Dimension; dim++)
        h[dim] = p[dim];
      m_DeformationField->SetPixel(ind, h);
    }

    DisplacementType u;
    for (unsigned int i = 0; i < Dimension; i++)
      u[i] = p[i];
    m_DisplacementField->SetPixel(ind, h - u);
  }

  // Warp images
  m_OutputImages.Clear();
  for (unsigned int ichan = 0; ichan < numChannels; ichan++)
  {
    typedef itk::WarpImageFilter<
      ImageType, ImageType, DeformationFieldType>
      WarperType;
    typename WarperType::Pointer warpf = WarperType::New();
    warpf->SetInput(m_DownMovingImages[ichan]);
    warpf->SetEdgePaddingValue(0.0);
    warpf->SetDisplacementField(m_DisplacementField);
    warpf->SetOutputDirection(m_DisplacementField->GetDirection());
    warpf->SetOutputOrigin(m_DisplacementField->GetOrigin());
    warpf->SetOutputSpacing(m_DisplacementField->GetSpacing());
    warpf->Update();
    m_OutputImages.Append(warpf->GetOutput());
  }

  return false;
}

#endif
