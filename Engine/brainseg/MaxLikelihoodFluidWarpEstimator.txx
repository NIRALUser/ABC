
#ifndef _MaxLikelihoodFluidWarpEstimator_txx
#define _MaxLikelihoodFluidWarpEstimator_txx

#include "itkAddImageFilter.h"
#include "itkDivideImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkSquareImageFilter.h"

#include "itkApproximateLogImageFilter.h"
#include "itkStatisticsImageFilter.h"

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
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
::MaxLikelihoodFluidWarpEstimator()
{
  m_Iterations = 10;
  m_MaxStep = 0.5;
  m_Delta = 0.0;
  m_KernelWidth = 1.0;
  m_NumberOfScales = 3;
  m_Modified = false;
}

template <class TPixel, unsigned int Dimension>
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
::~MaxLikelihoodFluidWarpEstimator()
{
}

template <class TPixel, unsigned int Dimension>
void
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
::SetLikelihoodImages(const DynArray<ImagePointer>& images)
{
  m_LikelihoodImages = images;
  m_Modified = true;

  this->ScaleLikelihoodImages();
}

template <class TPixel, unsigned int Dimension>
void
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
::SetPriorImages(const DynArray<ImagePointer>& images)
{
  m_PriorImages = images;
  m_Modified = true;
}

template <class TPixel, unsigned int Dimension>
void
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
::SetMask(MaskPointer m)
{
  m_Mask = m;
  m_Modified = true;
}

template <class TPixel, unsigned int Dimension>
void
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
::ScaleLikelihoodImages()
{
#if 0
  ImagePointer sumImage = ImageType::New();
  sumImage->CopyInformation(m_LikelihoodImages[0]);
  sumImage->SetRegions(m_LikelihoodImages[0]->GetLargestPossibleRegion());
  sumImage->Allocate();
  sumImage->FillBuffer(1e-20);

  for (unsigned int c = 0; c < m_LikelihoodImages.GetSize(); c++)
  {
    typedef itk::AddImageFilter<ImageType, ImageType, ImageType> AddFilterType;
    typename AddFilterType::Pointer addf = AddFilterType::New();
    addf->SetInput1(sumImage);
    addf->SetInput2(m_LikelihoodImages[c]);
    addf->Update();

    sumImage = addf->GetOutput();
  }

  for (unsigned int c = 0; c < m_LikelihoodImages.GetSize(); c++)
  {
    typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DivideFilterType;
    typename DivideFilterType::Pointer divf = DivideFilterType::New();
    divf->SetInput1(m_LikelihoodImages[c]);
    divf->SetInput2(sumImage);
    divf->Update();

    m_LikelihoodImages[c] = divf->GetOutput();
  }
#else

  for (unsigned int c = 0; c < m_LikelihoodImages.GetSize(); c++)
  {
    typedef itk::StatisticsImageFilter<ImageType> StatFilterType;
    typename StatFilterType::Pointer statf = StatFilterType::New();
    statf->SetInput(m_LikelihoodImages[c]);
    statf->Update();

    typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DivideFilterType;
    typename DivideFilterType::Pointer divf = DivideFilterType::New();
    divf->SetInput1(m_LikelihoodImages[c]);
    divf->SetConstant2(statf->GetMaximum());
    divf->Update();

    m_LikelihoodImages[c] = divf->GetOutput();
  }

#endif
}

template <class TPixel, unsigned int Dimension>
typename MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>::ImagePointer
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
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
typename MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>::ImagePointer
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
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
typename MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>::DeformationFieldPointer
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
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
typename MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>::DeformationFieldPointer
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
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
typename MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>::DeformationFieldPointer
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
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
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
::Update()
{
  if (!m_Modified)
    return;

  if (m_Iterations < 5)
    itkExceptionMacro(<< "Must have at least 5 iterations");

std::cerr << "PP lik " << m_LikelihoodImages.GetSize() << std::endl;
std::cerr << "PP pr " << m_PriorImages.GetSize() << std::endl;

  if (m_LikelihoodImages.GetSize() != m_PriorImages.GetSize())
    itkExceptionMacro(<< "Number of classes must match");

  unsigned int numClasses = m_LikelihoodImages.GetSize();

  ImageSizeType size = m_LikelihoodImages[0]->GetLargestPossibleRegion().GetSize();
  ImageSpacingType spacing = m_LikelihoodImages[0]->GetSpacing();

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
  m_WarpedPriorImages.Clear();
  for (unsigned int c = 0; c < numClasses; c++)
  {
/*
    typedef itk::ImageDuplicator<ImageType> DuperType;
    typename DuperType::Pointer dupef = DuperType::New();
    dupef->SetInputImage(m_PriorImages[c]);
    dupef->Update();
    m_WarpedPriorImages.Append(dupef->GetOutput());
*/
    m_WarpedPriorImages.Append(
      this->DownsampleImage(m_PriorImages[c],
        m_MultiScaleSizes[m_NumberOfScales-1],
        m_MultiScaleSpacings[m_NumberOfScales-1]) );

  }

  // Initialize H-field and downsample
  m_DeformationField = DeformationFieldType::New();
  //m_DeformationField->CopyInformation(m_LikelihoodImages[0]);
  m_DeformationField->SetDirection(m_LikelihoodImages[0]->GetDirection());
  m_DeformationField->SetOrigin(m_LikelihoodImages[0]->GetOrigin());
  m_DeformationField->SetSpacing(m_LikelihoodImages[0]->GetSpacing());
  m_DeformationField->SetRegions(m_LikelihoodImages[0]->GetLargestPossibleRegion());
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
  //m_DisplacementField->CopyInformation(m_LikelihoodImages[0]);
  m_DisplacementField->SetDirection(m_LikelihoodImages[0]->GetDirection());
  m_DisplacementField->SetOrigin(m_LikelihoodImages[0]->GetOrigin());
  m_DisplacementField->SetSpacing(m_LikelihoodImages[0]->GetSpacing());
  m_DisplacementField->SetRegions(m_LikelihoodImages[0]->GetLargestPossibleRegion());
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

    m_WarpedPriorImages.Clear();
    for (unsigned int c = 0; c < numClasses; c++)
    {
      typedef itk::WarpImageFilter<
        ImageType, ImageType, DeformationFieldType>
        WarperType;
      typename WarperType::Pointer warpf = WarperType::New();
      warpf->SetInput(m_PriorImages[c]);
      warpf->SetEdgePaddingValue(0.0);
      warpf->SetDeformationField(m_DisplacementField);
      warpf->SetOutputDirection(m_DeformationField->GetDirection());
      warpf->SetOutputOrigin(m_DeformationField->GetOrigin());
      warpf->SetOutputSpacing(m_DeformationField->GetSpacing());
      warpf->Update();
      m_WarpedPriorImages.Append(
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
      m_DownLikelihoodImages = m_LikelihoodImages;
      m_DownPriorImages = m_PriorImages;
    }
    else
    {
      m_DownLikelihoodImages.Clear();
      m_DownPriorImages.Clear();
      for (unsigned int c = 0; c < numClasses; c++)
      {
        m_DownLikelihoodImages.Append(
          this->DownsampleImage(m_LikelihoodImages[c], currsize, currspacing) );
        m_DownPriorImages.Append(
          this->DownsampleImage(m_PriorImages[c], currsize, currspacing) );
      }
    }

    if (s < (m_NumberOfScales-1))
    {
      m_DeformationField = this->UpsampleDeformation(
        m_DeformationField, currsize, currspacing);
      m_DisplacementField = this->UpsampleDisplacement(
        m_DisplacementField, currsize, currspacing);

      m_WarpedPriorImages.Clear();
      for (unsigned int c = 0; c < numClasses; c++)
      {
        typedef itk::WarpImageFilter<
          ImageType, ImageType, DeformationFieldType>
          WarperType;
        typename WarperType::Pointer warpf = WarperType::New();
        warpf->SetInput(m_DownPriorImages[c]);
        if (c < (numClasses-1))
          warpf->SetEdgePaddingValue(0.0);
        else
          warpf->SetEdgePaddingValue(1.0);
        warpf->SetDeformationField(m_DisplacementField);
        warpf->SetOutputDirection(m_DeformationField->GetDirection());
        warpf->SetOutputOrigin(m_DeformationField->GetOrigin());
        warpf->SetOutputSpacing(m_DeformationField->GetSpacing());
        warpf->Update();
        m_WarpedPriorImages.Append(warpf->GetOutput());
      }
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
  m_WarpedPriorImages.Clear();
  for (unsigned int c = 0; c < numClasses; c++)
  {
    typedef itk::WarpImageFilter<
      ImageType, ImageType, DeformationFieldType>
      WarperType;
    typename WarperType::Pointer warpf = WarperType::New();
    warpf->SetInput(m_PriorImages[c]);
    if (c < (numClasses-1))
      warpf->SetEdgePaddingValue(0.0);
    else
      warpf->SetEdgePaddingValue(1.0);
    warpf->SetDeformationField(m_DisplacementField);
    warpf->SetOutputDirection(m_DeformationField->GetDirection());
    warpf->SetOutputOrigin(m_DeformationField->GetOrigin());
    warpf->SetOutputSpacing(m_DeformationField->GetSpacing());
    warpf->Update();
    m_WarpedPriorImages.Append(warpf->GetOutput());
  }

  m_Modified = false;
}

template <class TPixel, unsigned int Dimension>
bool
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
::Step()
{
  typedef itk::AddImageFilter<ImageType, ImageType, ImageType> AddFilterType;
  typedef itk::MultiplyImageFilter<ImageType, ImageType, ImageType> MultiplyFilterType;
  typedef itk::SubtractImageFilter<ImageType, ImageType, ImageType> SubtractFilterType;

  unsigned int numClasses = m_DownLikelihoodImages.GetSize();

  ImageSpacingType spacing = m_DownLikelihoodImages[0]->GetSpacing();

  ImageSizeType size = m_DownLikelihoodImages[0]->GetLargestPossibleRegion().GetSize();

  DisplacementType edgev;
  //edgev.Fill(vnl_huge_val(0.0f));
  edgev.Fill(sqrt(-1.0f));
  DisplacementType zerov;
  zerov.Fill(0.0);

  // Denominator = sum of warped prior * likelihood over class c
std::cerr << "PP DEBUG denom" << std::endl;
  ImagePointer sumProdImage = ImageType::New();
  sumProdImage->CopyInformation(m_WarpedPriorImages[0]);
  sumProdImage->SetRegions(m_WarpedPriorImages[0]->GetLargestPossibleRegion());
  sumProdImage->Allocate();
  sumProdImage->FillBuffer(0);

  for (unsigned int c = 0; c < m_PriorImages.GetSize(); c++)
  {
    typename MultiplyFilterType::Pointer mulf = MultiplyFilterType::New();
    mulf->SetInput1(m_WarpedPriorImages[c]);
    mulf->SetInput2(m_DownLikelihoodImages[c]);
    mulf->Update();

    typename AddFilterType::Pointer addf = AddFilterType::New();
    addf->SetInput1(sumProdImage);
    addf->SetInput2(mulf->GetOutput());
    addf->Update();
  }

  // Compute objective
/*
// if (m_DisplayObjective)
// {
  typedef itk::ApproximateLogImageFilter<ImageType, ImageType> LogFilterType;
  typename LogFilterType::Pointer logf = LogFilterType::New();
  logf->SetInput(sumProdImage);
  logf->Update();

  typedef itk::StatisticsImageFilter<ImageType> StatFilterType;
  typename StatFilterType::Pointer statf = StatFilterType::New();
  statf->SetInput(logf->GetOutput());
  statf->Update();

  std::cout << "  log likelihood = " << statf->GetSum() << std::endl;
  //}
*/

  // Compute derivative of objective
  ImagePointer derivLogImage;
  {
    // Taylor order 2 derivative
    typename SubtractFilterType::Pointer subf = SubtractFilterType::New();
    subf->SetConstant1(2.0);
    subf->SetInput2(sumProdImage);
    subf->Update();

    derivLogImage = subf->GetOutput();

/*
    // Taylor order 3 derivative
    typedef itk::SquareImageFilter<ImageType, ImageType> SquareFilterType;
    typename SquareFilterType::Pointer sqf = SquareFilterType::New();
    sqf->SetInput(sumProdImage);
    sqf->Update();

    typename MultiplyFilterType::Pointer mulf = MultiplyFilterType::New();
    mulf->SetInput1(sumProdImage);
    mulf->SetConstant2(3.0);
    mulf->Update();

    typename SubtractFilterType::Pointer subf = SubtractFilterType::New();
    subf->SetInput1(sqf->GetOutput());
    subf->SetInput2(mulf->GetOutput());
    subf->Update();

    typename AddFilterType::Pointer addf = AddFilterType::New();
    addf->SetInput1(subf->GetOutput());
    addf->SetConstant2(3.0);
    addf->Update();

    derivLogImage = addf->GetOutput();
*/
  }

  // Velocity field
  // v = sum_c { (fixed_c - moving_c) * grad(moving_c) }

  DynArray<ImagePointer> velocImages;
  for (unsigned int dim = 0; dim < Dimension; dim++)
  {
    ImagePointer tmp = ImageType::New();
    //tmp->CopyInformation(m_DownLikelihoodImages[0]);
    tmp->SetDirection(m_DownLikelihoodImages[0]->GetDirection());
    tmp->SetOrigin(m_DownLikelihoodImages[0]->GetOrigin());
    tmp->SetSpacing(m_DownLikelihoodImages[0]->GetSpacing());
    tmp->SetRegions(m_DownLikelihoodImages[0]->GetLargestPossibleRegion());
    tmp->Allocate();
    tmp->FillBuffer(0);
    velocImages.Append(tmp);
  }

std::cerr << "PP DEBUG veloc" << std::endl;
  for (unsigned int c = 0; c < numClasses; c++)
  {
    for (unsigned int dim = 0; dim < Dimension; dim++)
    {
      typedef itk::DerivativeImageFilter<ImageType, ImageType>
        DerivativeFilterType;
      typename DerivativeFilterType::Pointer gradf = DerivativeFilterType::New();
      gradf->SetInput(m_WarpedPriorImages[c]);
      gradf->SetDirection(dim);
      gradf->SetUseImageSpacingOn();
      gradf->Update();

      typename MultiplyFilterType::Pointer mulf = MultiplyFilterType::New();
      mulf->SetInput1(m_DownLikelihoodImages[c]);
      mulf->SetInput2(gradf->GetOutput());
      mulf->Update();

      typename MultiplyFilterType::Pointer mulf2 = MultiplyFilterType::New();
      mulf2->SetInput1(mulf->GetOutput());
      mulf2->SetInput2(derivLogImage);
      mulf2->Update();

      typename AddFilterType::Pointer addf = AddFilterType::New();
      addf->SetInput1(velocImages[dim]);
      addf->SetInput2(mulf2->GetOutput());
      addf->Update();

      velocImages[dim] = addf->GetOutput();
    } // for dim
  } // for c

  // Put together accumulated forces into a single vector image
  typedef itk::ComposeImageFilter<ImageType, DeformationFieldType> VectorComposerType;

  typename VectorComposerType::Pointer vecf = VectorComposerType::New();
  for (unsigned int dim = 0; dim < Dimension; dim++)
    vecf->SetInput(dim, velocImages[dim]);
  vecf->Update();
  
  DeformationFieldPointer velocF = vecf->GetOutput();

  // Apply Green's kernel to velocity field
  typedef VectorBlurImageFilter<DeformationFieldType, DeformationFieldType>
    DeformationSmootherType;
  typename DeformationSmootherType::Pointer defsmoother = DeformationSmootherType::New();
  //defsmoother->SetKernelWidth(adjustedWidth);
  defsmoother->SetKernelWidth(m_KernelWidth);
  defsmoother->SetInput(velocF);
  defsmoother->Update();

  velocF = defsmoother->GetOutput();

  typedef itk::ImageRegionIteratorWithIndex<DeformationFieldType> IteratorType;
  IteratorType it(velocF, velocF->GetLargestPossibleRegion());

  // Compute max velocity magnitude
  double maxVeloc = 1e-20;
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    DisplacementType v = it.Get();
    double d = v.GetNorm();
    if (d > maxVeloc)
      maxVeloc = d;
  }

std::cout << "PP DEBUG maxVeloc = " << maxVeloc << std::endl;

  // Update delta at initial step
  if (m_Delta == 0.0)
    m_Delta = m_MaxStep / maxVeloc;

  // Test for convergence
  //if ((maxVeloc*m_Delta) < 1e-5)
  //  return true;

  double adaptDelta = m_Delta;
  if ((maxVeloc*m_Delta) > m_MaxStep)
    adaptDelta = m_MaxStep / maxVeloc;

  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    it.Set(it.Get() * adaptDelta);

  // Compose velocity field
  // new h(x) <= h( g(x) ) where g(x) = x + v
  typedef itk::WarpVectorImageFilter<
    DeformationFieldType, DeformationFieldType, DeformationFieldType>
    ComposerType;
  typename ComposerType::Pointer compf = ComposerType::New();
  compf->SetInput(m_DeformationField);
  compf->SetDeformationField(velocF);
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
  m_WarpedPriorImages.Clear();
  for (unsigned int c = 0; c < numClasses; c++)
  {
    typedef itk::WarpImageFilter<
      ImageType, ImageType, DeformationFieldType>
      WarperType;
    typename WarperType::Pointer warpf = WarperType::New();
    warpf->SetInput(m_DownPriorImages[c]);
    if (c < (numClasses-1))
      warpf->SetEdgePaddingValue(0.0);
    else
      warpf->SetEdgePaddingValue(1.0);
    warpf->SetDeformationField(m_DisplacementField);
    warpf->SetOutputDirection(m_DeformationField->GetDirection());
    warpf->SetOutputOrigin(m_DeformationField->GetOrigin());
    warpf->SetOutputSpacing(m_DeformationField->GetSpacing());
    warpf->Update();
    m_WarpedPriorImages.Append(warpf->GetOutput());
  }

  return false;
}

#endif
