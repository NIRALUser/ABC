
#ifndef _SimpleGreedyFluidRegistration_txx
#define _SimpleGreedyFluidRegistration_txx

#include "itkDerivativeImageFilter.h"
#include "itkImageDuplicator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkVectorResampleImageFilter.h"
#include "itkWarpImageFilter.h"
#include "itkWarpVectorImageFilter.h"

#include "VectorBlurImageFilter.h"

template <class TPixel, unsigned int Dimension>
SimpleGreedyFluidRegistration<TPixel, Dimension>
::SimpleGreedyFluidRegistration()
{
  m_Iterations = 10;
  m_MaxStep = 0.5;
  m_Delta = 0.0;
  m_KernelWidth = 1.0;
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
::DownsampleImage(ImagePointer img)
{
  ImageRegionType region = img->GetLargestPossibleRegion();
  ImageSizeType size = region.GetSize();

  ImageSpacingType spacing = img->GetSpacing();

  ImageSizeType downsize;
  for (uint dim = 0; dim < Dimension; dim++)
    downsize[dim] = size[dim] / 2;
  ImageSpacingType downspacing;
  for (uint dim = 0; dim < Dimension; dim++)
    downspacing[dim] = spacing[dim] * (double)size[dim] / (double)downsize[dim];

  typedef itk::ResampleImageFilter<ImageType, ImageType>
    ResamplerType;
  typename ResamplerType::Pointer resf = ResamplerType::New();
  resf->SetInput(img);
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
::UpsampleImage(ImagePointer img)
{
  typedef itk::ResampleImageFilter<ImageType, ImageType>
    ResamplerType;
  typename ResamplerType::Pointer resf = ResamplerType::New();
  resf->SetInput(img);
  resf->SetOutputParametersFromImage(m_FixedImages[0]);
  resf->Update();

  return resf->GetOutput();
}

template <class TPixel, unsigned int Dimension>
typename SimpleGreedyFluidRegistration<TPixel, Dimension>::DeformationFieldPointer
SimpleGreedyFluidRegistration<TPixel, Dimension>
::DownsampleDeformation(DeformationFieldPointer img)
{
  ImageRegionType region = img->GetLargestPossibleRegion();
  ImageSizeType size = region.GetSize();

  ImageSpacingType spacing = img->GetSpacing();

  ImageSizeType downsize;
  for (uint dim = 0; dim < Dimension; dim++)
    downsize[dim] = size[dim] / 2;
  ImageSpacingType downspacing;
  for (uint dim = 0; dim < Dimension; dim++)
    downspacing[dim] = spacing[dim] * (double)size[dim] / (double)downsize[dim];

  typedef itk::VectorResampleImageFilter<DeformationFieldType, DeformationFieldType>
    ResamplerType;
  typename ResamplerType::Pointer resf = ResamplerType::New();
  resf->SetInput(img);
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
::UpsampleDeformation(DeformationFieldPointer img)
{
  typedef itk::VectorResampleImageFilter<DeformationFieldType, DeformationFieldType>
    ResamplerType;
  typename ResamplerType::Pointer resf = ResamplerType::New();
  resf->SetInput(img);
  resf->SetOutputDirection(m_FixedImages[0]->GetDirection());
  resf->SetOutputSpacing(m_FixedImages[0]->GetSpacing());
  resf->SetOutputOrigin(m_FixedImages[0]->GetOrigin());
  resf->SetSize(m_FixedImages[0]->GetLargestPossibleRegion().GetSize());
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

  if (m_FixedImages.GetSize() != m_MovingImages.GetSize())
    itkExceptionMacro(<< "Number of channels must match");

  // Initialize out = input moving
  m_OutputImages.Clear();
  for (uint c = 0; c < m_MovingImages.GetSize(); c++)
  {
/*
    typedef itk::ImageDuplicator<ImageType> DuperType;
    typename DuperType::Pointer dupef = DuperType::New();
    dupef->SetInputImage(m_MovingImages[c]);
    dupef->Update();
    m_OutputImages.Append(dupef->GetOutput());
*/
    m_OutputImages.Append(this->DownsampleImage(m_MovingImages[c]));
  }

  // Downsample input images
  m_DownFixedImages.Clear();
  for (uint c = 0; c < m_FixedImages.GetSize(); c++)
    m_DownFixedImages.Append(this->DownsampleImage(m_FixedImages[c]));
  m_DownMovingImages.Clear();
  for (uint c = 0; c < m_MovingImages.GetSize(); c++)
    m_DownMovingImages.Append(this->DownsampleImage(m_MovingImages[c]));

  // Initialize phi(x) = x
  m_DeformationField = DeformationFieldType::New();
  //m_DeformationField->CopyInformation(m_DownFixedImages[0]);
  m_DeformationField->SetDirection(m_DownFixedImages[0]->GetDirection());
  m_DeformationField->SetOrigin(m_DownFixedImages[0]->GetOrigin());
  m_DeformationField->SetSpacing(m_DownFixedImages[0]->GetSpacing());
  m_DeformationField->SetRegions(m_DownFixedImages[0]->GetLargestPossibleRegion());
  m_DeformationField->Allocate();

  DisplacementType zerov;
  zerov.Fill(0.0);
  m_DeformationField->FillBuffer(zerov);

  typedef itk::ImageRegionIteratorWithIndex<DeformationFieldType> IteratorType;
  IteratorType it(m_DeformationField, m_DeformationField->GetLargestPossibleRegion());

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    ImagePointType p;
    m_DeformationField->TransformIndexToPhysicalPoint(it.GetIndex(), p);

    DisplacementType v;
    for (uint i = 0; i < Dimension; i++)
      v[i] = p[i];
    it.Set(v);
  }

  // Store displacement field as well
  m_DisplacementField = DeformationFieldType::New();
  //m_DisplacementField->CopyInformation(m_DownFixedImages[0]);
  m_DisplacementField->SetDirection(m_DownFixedImages[0]->GetDirection());
  m_DisplacementField->SetOrigin(m_DownFixedImages[0]->GetOrigin());
  m_DisplacementField->SetSpacing(m_DownFixedImages[0]->GetSpacing());
  m_DisplacementField->SetRegions(m_DownFixedImages[0]->GetLargestPossibleRegion());
  m_DisplacementField->Allocate();
  m_DisplacementField->FillBuffer(zerov);

  // Initialize using user-specified deformation if available
  if (!m_InitialDisplacementField.IsNull())
  {
    m_DisplacementField =
      this->DownsampleDeformation(m_InitialDisplacementField);

    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      ImageIndexType ind = it.GetIndex();

      ImagePointType p;
      m_DeformationField->TransformIndexToPhysicalPoint(ind, p);

      DisplacementType v = m_DisplacementField->GetPixel(ind);

      DisplacementType h;
      for (uint i = 0; i < Dimension; i++)
        h[i] = p[i] + v[i];
      m_DeformationField->SetPixel(ind, h);
    }

    m_OutputImages.Clear();
    for (uint ichan = 0; ichan < m_MovingImages.GetSize(); ichan++)
    {
      typedef itk::WarpImageFilter<
        ImageType, ImageType, DeformationFieldType>
        WarperType;
      typename WarperType::Pointer warpf = WarperType::New();
      warpf->SetInput(m_DownMovingImages[ichan]);
      warpf->SetDeformationField(m_DisplacementField);
      warpf->SetOutputDirection(m_DeformationField->GetDirection());
      warpf->SetOutputOrigin(m_DeformationField->GetOrigin());
      warpf->SetOutputSpacing(m_DeformationField->GetSpacing());
      warpf->Update();
      m_OutputImages.Append(warpf->GetOutput());
    }
  }

  // Initialize delta
  m_Delta = 0.0;

  // Greedy optimization
  for (uint iter = 1; iter <= m_Iterations; iter++)
  {
    bool converge = this->Step();
    if (converge)
      break;
  }

  // Upsample deformations
  m_DeformationField = this->UpsampleDeformation(m_DeformationField);
  m_DisplacementField = this->UpsampleDeformation(m_DisplacementField);

  // Warp images
  m_OutputImages.Clear();
  for (uint ichan = 0; ichan < m_MovingImages.GetSize(); ichan++)
  {
    typedef itk::WarpImageFilter<
      ImageType, ImageType, DeformationFieldType>
      WarperType;
    typename WarperType::Pointer warpf = WarperType::New();
    warpf->SetInput(m_MovingImages[ichan]);
    warpf->SetDeformationField(m_DisplacementField);
    warpf->SetOutputDirection(m_DeformationField->GetDirection());
    warpf->SetOutputOrigin(m_DeformationField->GetOrigin());
    warpf->SetOutputSpacing(m_DeformationField->GetSpacing());
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
  uint numChannels = m_DownFixedImages.GetSize();

  // Find scale adjustment
  ImageSpacingType spacing = m_DownFixedImages[0]->GetSpacing();
  double minSpacing = spacing[0];
  for (uint i = 1; i < Dimension; i++)
    if (spacing[i] < minSpacing)
      minSpacing = spacing[i];

  DisplacementType zerov;
  zerov.Fill(0.0);

  // Compute velocity field
  // v = sum_c { (fixed_c - moving_c) * grad(moving_c) }
  DeformationFieldPointer velocF = DeformationFieldType::New();
  //velocF->CopyInformation(m_DownFixedImages[0]);
  velocF->SetDirection(m_DownFixedImages[0]->GetDirection());
  velocF->SetOrigin(m_DownFixedImages[0]->GetOrigin());
  velocF->SetSpacing(m_DownFixedImages[0]->GetSpacing());
  velocF->SetRegions(m_DownFixedImages[0]->GetLargestPossibleRegion());
  velocF->Allocate();
  velocF->FillBuffer(zerov);

  typedef itk::ImageRegionIteratorWithIndex<DeformationFieldType> IteratorType;

  for (uint ichan = 0; ichan < numChannels; ichan++)
  {
    typedef itk::DerivativeImageFilter<ImageType, ImageType>
      DerivativeFilterType;

    for (uint dim = 0; dim < Dimension; dim++)
    {
      typename DerivativeFilterType::Pointer derivf = DerivativeFilterType::New();
      derivf->SetInput(m_OutputImages[ichan]);
      derivf->SetDirection(dim);
      derivf->SetUseImageSpacingOn();
      derivf->Update();

      ImagePointer grad_d = derivf->GetOutput();

      IteratorType it(velocF, velocF->GetLargestPossibleRegion());
      for (it.GoToBegin(); !it.IsAtEnd(); ++it)
      {
        DisplacementType v = it.Get();

        ImageIndexType ind = it.GetIndex();

        double w =
          m_DownFixedImages[ichan]->GetPixel(ind) -
          m_OutputImages[ichan]->GetPixel(ind);

        v[dim] += w * grad_d->GetPixel(ind);

        it.Set(v);
      }
    } // for dim
  } // for ichan

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
    m_Delta = m_MaxStep * minSpacing / maxVeloc;
  }

  // Test for convergence
  if ((maxVeloc*m_Delta) < (1e-10*minSpacing))
    return true;

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    it.Set(it.Get() * m_Delta);
  }

  // Compose velocity field
  // h(x) = h( g(x) ) where g(x) = x + v
  typedef itk::WarpVectorImageFilter<
    DeformationFieldType, DeformationFieldType, DeformationFieldType>
    ComposerType;
  typename ComposerType::Pointer compf = ComposerType::New();
  compf->SetInput(m_DeformationField);
  compf->SetDeformationField(velocF);
  compf->SetOutputDirection(m_DeformationField->GetDirection());
  compf->SetOutputOrigin(m_DeformationField->GetOrigin());
  compf->SetOutputSpacing(m_DeformationField->GetSpacing());
  compf->Update();

  m_DeformationField = compf->GetOutput();

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    ImageIndexType ind = it.GetIndex();

    ImagePointType p;
    m_DeformationField->TransformIndexToPhysicalPoint(ind, p);

    DisplacementType u;
    for (uint i = 0; i < Dimension; i++)
      u[i] = p[i];
    m_DisplacementField->SetPixel(ind,
      m_DeformationField->GetPixel(ind) - u);
  }

  // Warp images
  m_OutputImages.Clear();
  for (uint ichan = 0; ichan < numChannels; ichan++)
  {
    typedef itk::WarpImageFilter<
      ImageType, ImageType, DeformationFieldType>
      WarperType;
    typename WarperType::Pointer warpf = WarperType::New();
    warpf->SetInput(m_DownMovingImages[ichan]);
    warpf->SetDeformationField(m_DisplacementField);
    warpf->SetOutputDirection(m_DeformationField->GetDirection());
    warpf->SetOutputOrigin(m_DeformationField->GetOrigin());
    warpf->SetOutputSpacing(m_DeformationField->GetSpacing());
    warpf->Update();
    m_OutputImages.Append(warpf->GetOutput());
  }

  return false;
}

#endif
