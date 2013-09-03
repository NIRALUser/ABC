
#ifndef _MaxLikelihoodFluidWarpEstimator_txx
#define _MaxLikelihoodFluidWarpEstimator_txx

#include "itkAddImageFilter.h"
#include "itkDivideImageFilter.h"
#include "itkMultiplyImageFilter.h"
#include "itkSubtractImageFilter.h"
#include "itkSquareImageFilter.h"

#include "itkTsallisLogImageFilter.h"
#include "itkTsallisLogDerivativeImageFilter.h"

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
  m_RegularityWeight = 1.0;
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

  //this->ScaleLikelihoodImages();
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

  typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DivideFilterType;

  double maxMaxL = 1e-20;

  for (unsigned int c = 0; c < m_LikelihoodImages.GetSize(); c++)
  {
    double maxL = 1e-20;

    typedef itk::ImageRegionIteratorWithIndex<ImageType> IteratorType;
    IteratorType it(m_LikelihoodImages[c], m_LikelihoodImages[c]->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
      if (it.Get() > maxL)
        maxL = it.Get();

    if (maxL > maxMaxL)
      maxMaxL = maxL;

    typename DivideFilterType::Pointer divf = DivideFilterType::New();
    divf->SetInput1(m_LikelihoodImages[c]);
    divf->SetConstant2(maxL);
    divf->Update();

    m_LikelihoodImages[c] = divf->GetOutput();
  }

/*
  for (unsigned int c = 0; c < m_LikelihoodImages.GetSize(); c++)
  {
    typename DivideFilterType::Pointer divf = DivideFilterType::New();
    divf->SetInput1(m_LikelihoodImages[c]);
    divf->SetConstant2(maxMaxL);
    divf->Update();

    m_LikelihoodImages[c] = divf->GetOutput();
  }
*/

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
typename MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>::VectorFieldPointer
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
::DownsampleDisplacement(VectorFieldPointer img, ImageSizeType downsize, ImageSpacingType downspacing)
{
  ImageRegionType region = img->GetLargestPossibleRegion();
  ImageSizeType size = region.GetSize();

  ImageSpacingType spacing = img->GetSpacing();

  typedef itk::VectorResampleImageFilter<VectorFieldType, VectorFieldType>
    ResamplerType;
  typename ResamplerType::Pointer resf = ResamplerType::New();
  resf->SetInput(img);
  VectorType zerov;
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
typename MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>::VectorFieldPointer
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
::ApplyKernel(VectorFieldPointer img)
{
  typedef VectorBlurImageFilter<VectorFieldType, VectorFieldType>
    VectorSmootherType;
  typename VectorSmootherType::Pointer vsmoothf = VectorSmootherType::New();
  vsmoothf->SetKernelWidth(m_CurrentKernelWidth);
  vsmoothf->SetInput(img);
  vsmoothf->Update();

  return vsmoothf->GetOutput();
}

template <class TPixel, unsigned int Dimension>
typename MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>::VectorFieldPointer
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
::ApplyKernelFlip(VectorFieldPointer img)
{
  typedef VectorBlurImageFilter<VectorFieldType, VectorFieldType>
    VectorSmootherType;
  typename VectorSmootherType::Pointer vsmoothf = VectorSmootherType::New();
  vsmoothf->SetKernelWidth(m_CurrentKernelWidth);
  vsmoothf->SetInput(img);
  vsmoothf->Update();

  typedef itk::MultiplyImageFilter<VectorFieldType, ImageType, VectorFieldType> VectorMulFilterType;
  typename VectorMulFilterType::Pointer vflipf = VectorMulFilterType::New();
  vflipf->SetInput1(vsmoothf->GetOutput());
  vflipf->SetConstant2(-1.0);
  vflipf->Update();

  return vflipf->GetOutput();
}

template <class TPixel, unsigned int Dimension>
typename MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>::VectorFieldPointer
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
::UpsampleDisplacement(VectorFieldPointer img, ImageSizeType upsize, ImageSpacingType upspacing)
{
  typedef itk::VectorResampleImageFilter<VectorFieldType, VectorFieldType>
    ResamplerType;
  typename ResamplerType::Pointer resf = ResamplerType::New();
  resf->SetInput(img);
  VectorType zerov;
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

  if (m_Iterations < 1)
    itkExceptionMacro(<< "Must have at least 1 iteration");

  if (m_LikelihoodImages.GetSize() != m_PriorImages.GetSize())
    itkExceptionMacro(<< "Number of classes must match");

  m_CurrentKernelWidth = m_KernelWidth;

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

  // Initialize momenta and velocity
  m_Momenta = VectorFieldType::New();
  //m_Momenta->CopyInformation(m_LikelihoodImages[0]);
  m_Momenta->SetDirection(m_LikelihoodImages[0]->GetDirection());
  m_Momenta->SetOrigin(m_LikelihoodImages[0]->GetOrigin());
  m_Momenta->SetSpacing(m_LikelihoodImages[0]->GetSpacing());
  m_Momenta->SetRegions(m_LikelihoodImages[0]->GetLargestPossibleRegion());
  m_Momenta->Allocate();

  m_Velocity = VectorFieldType::New();
  //m_Velocity->CopyInformation(m_LikelihoodImages[0]);
  m_Velocity->SetDirection(m_LikelihoodImages[0]->GetDirection());
  m_Velocity->SetOrigin(m_LikelihoodImages[0]->GetOrigin());
  m_Velocity->SetSpacing(m_LikelihoodImages[0]->GetSpacing());
  m_Velocity->SetRegions(m_LikelihoodImages[0]->GetLargestPossibleRegion());
  m_Velocity->Allocate();

  VectorType zerov;
  zerov.Fill(0.0);

  m_Momenta->FillBuffer(zerov);
  m_Velocity->FillBuffer(zerov);

  if (m_NumberOfScales > 1)
  {
    m_Momenta =
      this->DownsampleDisplacement(
        m_Momenta,
        m_MultiScaleSizes[m_NumberOfScales-1],
        m_MultiScaleSpacings[m_NumberOfScales-1]);
    m_Velocity =
      this->DownsampleDisplacement(
        m_Velocity,
        m_MultiScaleSizes[m_NumberOfScales-1],
        m_MultiScaleSpacings[m_NumberOfScales-1]);
  }

  typedef itk::ImageRegionIteratorWithIndex<VectorFieldType> IteratorType;

  // Initialize using user-specified velocity if available
  if (!m_InitialMomenta.IsNull())
  {
    m_Velocity = this->ApplyKernelFlip(m_InitialMomenta);

    m_WarpedPriorImages.Clear();
    for (unsigned int c = 0; c < numClasses; c++)
    {
      typedef itk::WarpImageFilter<
        ImageType, ImageType, VectorFieldType>
        WarperType;
      typename WarperType::Pointer warpf = WarperType::New();
      warpf->SetInput(m_PriorImages[c]);
      warpf->SetEdgePaddingValue(0.0);
      warpf->SetDisplacementField(m_Velocity);
      warpf->SetOutputDirection(m_Velocity->GetDirection());
      warpf->SetOutputOrigin(m_Velocity->GetOrigin());
      warpf->SetOutputSpacing(m_Velocity->GetSpacing());
      warpf->Update();
      m_WarpedPriorImages.Append(
        this->DownsampleImage(
          warpf->GetOutput(),
          m_MultiScaleSizes[m_NumberOfScales-1],
          m_MultiScaleSpacings[m_NumberOfScales-1]) );
    }

    m_Momenta =
      this->DownsampleDisplacement(
        m_InitialMomenta,
        m_MultiScaleSizes[m_NumberOfScales-1],
        m_MultiScaleSpacings[m_NumberOfScales-1]);

    m_Velocity =
      this->DownsampleDisplacement(
        m_Velocity,
        m_MultiScaleSizes[m_NumberOfScales-1],
        m_MultiScaleSpacings[m_NumberOfScales-1]);
  } // if v init exist

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
      m_Momenta = this->UpsampleDisplacement(m_Momenta, currsize, currspacing);
      m_Velocity = this->ApplyKernelFlip(m_Momenta);

      m_WarpedPriorImages.Clear();
      for (unsigned int c = 0; c < numClasses; c++)
      {
        typedef itk::WarpImageFilter<
          ImageType, ImageType, VectorFieldType>
          WarperType;
        typename WarperType::Pointer warpf = WarperType::New();
        warpf->SetInput(m_DownPriorImages[c]);
        if (c < (numClasses-1))
          warpf->SetEdgePaddingValue(0.0);
        else
          warpf->SetEdgePaddingValue(1.0);
        warpf->SetDisplacementField(m_Velocity);
        warpf->SetOutputDirection(m_Velocity->GetDirection());
        warpf->SetOutputOrigin(m_Velocity->GetOrigin());
        warpf->SetOutputSpacing(m_Velocity->GetSpacing());
        warpf->Update();
        m_WarpedPriorImages.Append(warpf->GetOutput());
      }
    }

std::cout << "  fluid optimization with kernel width = " << m_CurrentKernelWidth << std::endl;

    // Gradient descent
    double objf = this->ComputeObjective(m_Momenta);

    double ref_objf = objf;

    m_Delta = 0.0;

    for (unsigned int iter = 1; iter <= m_Iterations; iter++)
    {
      double prev_objf = objf;

      this->ComputeGradient();

      double objf_test = objf;

      // Line search
      unsigned int iline;
      for (iline = 1; iline <= 50; iline++)
      {
        typedef itk::MultiplyImageFilter<VectorFieldType, ImageType, VectorFieldType> VectorMulFilterType;
        typename VectorMulFilterType::Pointer vscalf = VectorMulFilterType::New();
        vscalf->SetInput1(m_GradientMomenta);
        vscalf->SetConstant2(m_Delta);
        vscalf->Update();

        typedef itk::SubtractImageFilter<VectorFieldType, VectorFieldType, VectorFieldType> VectorSubFilterType;
        typename VectorSubFilterType::Pointer vsubf = VectorSubFilterType::New();
        vsubf->SetInput1(m_Momenta);
        vsubf->SetInput2(vscalf->GetOutput());
        vsubf->Update();

        VectorFieldPointer Atest = vsubf->GetOutput();

        objf_test = this->ComputeObjective(Atest);

        if (objf_test < objf)
        {
          m_Momenta = Atest;
          m_Velocity = this->ApplyKernelFlip(Atest);
          objf = objf_test;
          m_Delta *= 1.2;
          break;
        }

        m_Delta *= 0.5;
      }
      
      if (iline > 50)
        break;

      if (fabs(prev_objf - objf) < 1e-4*fabs(ref_objf - objf))
        break;
    }

    m_CurrentKernelWidth *= 0.5;
  }

  // Warp images using final deformation
  m_WarpedPriorImages.Clear();
  for (unsigned int c = 0; c < numClasses; c++)
  {
    typedef itk::WarpImageFilter<
      ImageType, ImageType, VectorFieldType>
      WarperType;
    typename WarperType::Pointer warpf = WarperType::New();
    warpf->SetInput(m_PriorImages[c]);
    if (c < (numClasses-1))
      warpf->SetEdgePaddingValue(0.0);
    else
      warpf->SetEdgePaddingValue(1.0);
    warpf->SetDisplacementField(m_Velocity);
    warpf->SetOutputDirection(m_Velocity->GetDirection());
    warpf->SetOutputOrigin(m_Velocity->GetOrigin());
    warpf->SetOutputSpacing(m_Velocity->GetSpacing());
    warpf->Update();
    m_WarpedPriorImages.Append(warpf->GetOutput());
  }

  m_Modified = false;
}

template <class TPixel, unsigned int Dimension>
double
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
::ComputeObjective(VectorFieldPointer A)
{
  typedef itk::AddImageFilter<ImageType, ImageType, ImageType> AddFilterType;
  typedef itk::MultiplyImageFilter<ImageType, ImageType, ImageType> MultiplyFilterType;

  unsigned int numClasses = m_DownLikelihoodImages.GetSize();

  // Apply momenta
  VectorFieldPointer V = this->ApplyKernelFlip(A);

  DynArray<ImagePointer> warpedPriorImages;
  for (unsigned int c = 0; c < numClasses; c++)
  {
    typedef itk::WarpImageFilter<ImageType, ImageType, VectorFieldType> WarperType;
    typename WarperType::Pointer warpf = WarperType::New();
    warpf->SetInput(m_DownPriorImages[c]);
    if (c < (numClasses-1))
      warpf->SetEdgePaddingValue(0.0);
    else
      warpf->SetEdgePaddingValue(1.0);
    warpf->SetDisplacementField(V);
    warpf->SetOutputDirection(V->GetDirection());
    warpf->SetOutputOrigin(V->GetOrigin());
    warpf->SetOutputSpacing(V->GetSpacing());
    warpf->Update();
    warpedPriorImages.Append(warpf->GetOutput());
  }

  // Denominator = sum of warped prior * likelihood over class c
  ImagePointer sumProdImage = ImageType::New();
  sumProdImage->CopyInformation(m_DownPriorImages[0]);
  sumProdImage->SetRegions(m_DownPriorImages[0]->GetLargestPossibleRegion());
  sumProdImage->Allocate();
  sumProdImage->FillBuffer(0);

  //for (unsigned int c = 0; c < m_PriorImages.GetSize(); c++)
  for (unsigned int c = 0; c < (m_PriorImages.GetSize()-1); c++)
  {
    typename MultiplyFilterType::Pointer mulf = MultiplyFilterType::New();
    mulf->SetInput1(warpedPriorImages[c]);
    mulf->SetInput2(m_DownLikelihoodImages[c]);
    mulf->Update();

    typename AddFilterType::Pointer addf = AddFilterType::New();
    addf->SetInput1(sumProdImage);
    addf->SetInput2(mulf->GetOutput());
    addf->Update();

    sumProdImage = addf->GetOutput();
  }

  typedef itk::TsallisLogImageFilter<ImageType, ImageType> LogFilterType;
  typename LogFilterType::Pointer logf = LogFilterType::New();
  logf->SetInput(sumProdImage);
  logf->Update();

  double objf = 0;

  typedef itk::ImageRegionIteratorWithIndex<ImageType> IteratorType;
  IteratorType it(logf->GetOutput(), sumProdImage->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    objf -= it.Get();

  if (m_RegularityWeight > 0.0)
  {
    double VdotA = 0; // Sobolev
    typedef itk::ImageRegionIteratorWithIndex<VectorFieldType> VectorIteratorType;
    VectorIteratorType vIt(V, V->GetLargestPossibleRegion());
    for (vIt.GoToBegin(); !vIt.IsAtEnd(); ++vIt)
    {
      VectorType vvec = vIt.Get();
      VectorType avec = A->GetPixel(vIt.GetIndex());
      for (unsigned int d = 0; d < Dimension; d++)
        VdotA += vvec[d] * avec[d];
    }

    objf += m_RegularityWeight * VdotA;
  }

  return objf;
}

template <class TPixel, unsigned int Dimension>
void
MaxLikelihoodFluidWarpEstimator<TPixel, Dimension>
::ComputeGradient()
{
  typedef itk::AddImageFilter<ImageType, ImageType, ImageType> AddFilterType;
  typedef itk::DivideImageFilter<ImageType, ImageType, ImageType> DivideFilterType;
  typedef itk::MultiplyImageFilter<ImageType, ImageType, ImageType> MultiplyFilterType;
  typedef itk::SubtractImageFilter<ImageType, ImageType, ImageType> SubtractFilterType;

  unsigned int numClasses = m_DownLikelihoodImages.GetSize();

  ImageSpacingType spacing = m_DownLikelihoodImages[0]->GetSpacing();

  ImageSizeType size = m_DownLikelihoodImages[0]->GetLargestPossibleRegion().GetSize();

  m_Velocity = this->ApplyKernelFlip(m_Momenta);

  VectorType edgev;
  //edgev.Fill(vnl_huge_val(0.0f));
  edgev.Fill(sqrt(-1.0f));
  VectorType zerov;
  zerov.Fill(0.0);

  m_WarpedPriorImages.Clear();
  for (unsigned int c = 0; c < numClasses; c++)
  {
    typedef itk::WarpImageFilter<
      ImageType, ImageType, VectorFieldType>
      WarperType;
    typename WarperType::Pointer warpf = WarperType::New();
    warpf->SetInput(m_DownPriorImages[c]);
    if (c < (numClasses-1))
      warpf->SetEdgePaddingValue(0.0);
    else
      warpf->SetEdgePaddingValue(1.0);
    warpf->SetDisplacementField(m_Velocity);
    warpf->SetOutputDirection(m_Velocity->GetDirection());
    warpf->SetOutputOrigin(m_Velocity->GetOrigin());
    warpf->SetOutputSpacing(m_Velocity->GetSpacing());
    warpf->Update();
    m_WarpedPriorImages.Append(warpf->GetOutput());
  }

  // Denominator = sum of warped prior * likelihood over class c
  ImagePointer sumProdImage = ImageType::New();
  sumProdImage->CopyInformation(m_WarpedPriorImages[0]);
  sumProdImage->SetRegions(m_WarpedPriorImages[0]->GetLargestPossibleRegion());
  sumProdImage->Allocate();
  //sumProdImage->FillBuffer(1e-20);
  sumProdImage->FillBuffer(0);

  //for (unsigned int c = 0; c < m_PriorImages.GetSize(); c++)
  for (unsigned int c = 0; c < (m_PriorImages.GetSize()-1); c++)
  {
    typename MultiplyFilterType::Pointer mulf = MultiplyFilterType::New();
    mulf->SetInput1(m_WarpedPriorImages[c]);
    mulf->SetInput2(m_DownLikelihoodImages[c]);
    mulf->Update();

    typename AddFilterType::Pointer addf = AddFilterType::New();
    addf->SetInput1(sumProdImage);
    addf->SetInput2(mulf->GetOutput());
    addf->Update();

    sumProdImage = addf->GetOutput();
  }

  // log derivative
  typedef itk::TsallisLogDerivativeImageFilter<ImageType, ImageType> LogDerivType;
  typename LogDerivType::Pointer dlogf = LogDerivType::New();
  dlogf->SetInput(sumProdImage);
  dlogf->Update();
  ImagePointer derivLogImage = dlogf->GetOutput();

  // Functional gradient of velocity 
  DynArray<ImagePointer> gradImages;
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
    gradImages.Append(tmp);
  }

  //for (unsigned int c = 0; c < numClasses; c++)
  for (unsigned int c = 0; c < (numClasses-1); c++)
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

/*
      typename SubtractFilterType::Pointer subf = SubtractFilterType::New();
      subf->SetInput1(gradImages[dim]);
      subf->SetInput2(mulf2->GetOutput());
      subf->Update();

      gradImages[dim] = subf->GetOutput();
*/
      typename AddFilterType::Pointer addf = AddFilterType::New();
      addf->SetInput1(gradImages[dim]);
      addf->SetInput2(mulf2->GetOutput());
      addf->Update();

      gradImages[dim] = addf->GetOutput();
    } // for dim
  } // for c

  // Put together accumulated forces into a single vector image
  typedef itk::ComposeImageFilter<ImageType, VectorFieldType> VectorComposerType;

  typename VectorComposerType::Pointer vecf = VectorComposerType::New();
  for (unsigned int dim = 0; dim < Dimension; dim++)
    vecf->SetInput(dim, gradImages[dim]);
  vecf->Update();

  typedef itk::MultiplyImageFilter<VectorFieldType, ImageType, VectorFieldType> VectorMulFilterType;
  typename VectorMulFilterType::Pointer vscalf = VectorMulFilterType::New();
  vscalf->SetInput1(m_Momenta);
  vscalf->SetConstant2(m_RegularityWeight * 2.0);
  vscalf->Update();

  typedef itk::AddImageFilter<VectorFieldType, VectorFieldType, VectorFieldType> VectorAddFilterType;
  typename VectorAddFilterType::Pointer vaddf = VectorAddFilterType::New();
  vaddf->SetInput1(vecf->GetOutput());
  vaddf->SetInput2(vscalf->GetOutput());
  vaddf->Update();

  m_GradientMomenta = this->ApplyKernel(vaddf->GetOutput());

  // Update delta at initial step
  if (m_Delta == 0.0)
  {
    // Compute max gradient magnitude
    typedef itk::ImageRegionIteratorWithIndex<VectorFieldType> IteratorType;
    IteratorType it(m_GradientMomenta, m_GradientMomenta->GetLargestPossibleRegion());

    double maxGrad = 1e-20;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      VectorType v = it.Get();
      double d = v.GetNorm();
      if (d > maxGrad)
        maxGrad = d;
    }

    m_Delta = m_MaxStep / maxGrad;
  }

}

#endif
