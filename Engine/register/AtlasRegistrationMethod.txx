
#ifndef _AtlasRegistrationMethod_txx
#define _AtlasRegistrationMethod_txx

#include "itkAffineTransform.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkImageFileReader.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

// MI registration module
#include "AtlasRegistrationMethod.h"
#include "PairRegistrationMethod.h"
#include "RegistrationParameters.h"

#include "itkCurvatureFlowImageFilter.h"
#include "itkGradientAnisotropicDiffusionImageFilter.h"

#include "vnl/vnl_math.h"

#include "LLSBiasCorrector.h"

#include "Log.h"
#include "muFile.h"

#include "ImageDirectionStandardizer.h"

#include <fstream>

template <class TOutputPixel, class TProbabilityPixel>
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::AtlasRegistrationMethod()
{

  m_Suffix = "";

  m_OutputDirectory = "";

  m_TemplateFileName = "";

  m_ProbabilityFileNames.Clear();
  m_ImageFileNames.Clear();

  m_AtlasOrientation = "";
  m_ImageOrientations.Clear();

  m_Images.Clear();
  m_Probabilities.Clear();

  m_TemplateAffineTransform = AffineTransformType::New();
  m_AffineTransforms.Clear();

  m_AffineTransformReadFlags = FlagArrayType(1);
  m_AffineTransformReadFlags[0] = 0;

  m_UseNonLinearInterpolation = true;

  m_OutsideFOVCode = vnl_huge_val(1.0f);

  m_FOVMask = 0;

  m_DoneRegistration = false;
  m_DoneResample = false;

  m_Modified = false;

  m_PrefilteringMethod = "";
  m_PrefilteringIterations = 10;
  m_PrefilteringTimeStep = 0.1;

  m_AtlasLinearTransformChoice = AFFINE_TRANSFORM;
  m_ImageLinearTransformChoice = AFFINE_TRANSFORM;

}

template <class TOutputPixel, class TProbabilityPixel>
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::~AtlasRegistrationMethod()
{

  m_ImageFileNames.Clear();
  m_ProbabilityFileNames.Clear();

  m_Probabilities.Clear();
  m_Images.Clear();
  m_AffineTransforms.Clear();

}

template <class TOutputPixel, class TProbabilityPixel>
void
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::VerifyInitialization()
{

  if (m_ImageFileNames.GetSize() < 1)
    itkExceptionMacro(<< "No data images specified");

  if (m_ImageOrientations.GetSize() != m_ImageFileNames.GetSize())
    itkExceptionMacro(<< "Image - orientation info mismatch");

  /*
  // No atlas checks:
  // It's OK if we have no associated atlas files, then only do image-image
  // registrations
  if (m_TemplateFileName.length() == 0)
    itkExceptionMacro(<< "Template file name not specified");
  if (m_ProbabilityFileNames.GetSize() < 1)
    itkExceptionMacro(<< "No probability images specified");
  */

}

template <class TOutputPixel, class TProbabilityPixel>
void
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::WriteParameters()
{

  itkDebugMacro(<< "Write parameters");

  if (!m_DoneRegistration)
    this->RegisterImages();

  std::string firststr =
    m_OutputDirectory + mu::get_name(m_ImageFileNames[0].c_str()) + std::string("_to_");

  std::string suffixstr;
  if (m_Suffix.length() != 0)
    suffixstr = std::string("_") + m_Suffix;
  else
    suffixstr = std::string("");

  if (m_TemplateFileName.length() != 0)
  {
    std::string name = mu::get_name(m_TemplateFileName.c_str());

    // Write recently computed affine transform
    if (m_AffineTransformReadFlags[0] == 0)
    {
      std::string affinefn =
        firststr + name + suffixstr + std::string(".affine");

      muLogMacro(<< "Writing " << affinefn << "...\n");

      PairRegistrationMethod<InternalImagePixelType>::
        WriteAffineTransform(affinefn.c_str(), m_TemplateAffineTransform);
    }

  } // if template defined


  for (unsigned int i = 1; i < m_ImageFileNames.GetSize(); i++)
  {
    if (m_AffineTransformReadFlags[i] != 0)
      continue;

    std::string name = mu::get_name(m_ImageFileNames[i].c_str());

    std::string fn =
      firststr + name + suffixstr + std::string(".affine");

    muLogMacro(<< "Writing " << fn << "...\n");

    PairRegistrationMethod<InternalImagePixelType>::
      WriteAffineTransform(fn.c_str(), m_AffineTransforms[i]);
  }

}

template <class TOutputPixel, class TProbabilityPixel>
void
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::ReadParameters()
{

  itkDebugMacro(<< "Read parameters");

  m_DoneRegistration = false;

  std::string firststr =
    m_OutputDirectory + mu::get_name(m_ImageFileNames[0].c_str()) +
    std::string("_to_");

  std::string suffixstr;
  if (m_Suffix.length() != 0)
    suffixstr = std::string("_") + m_Suffix;
  else
    suffixstr = std::string("");

  // Read template to image transforms
  if (m_TemplateFileName.length() != 0)
  {

    std::string name = mu::get_name(m_TemplateFileName.c_str());

    std::string fn =
      firststr + name + suffixstr + std::string(".affine");

    muLogMacro(<< "Reading " << fn << "...\n");

    try
    {
      m_TemplateAffineTransform =
        PairRegistrationMethod<InternalImagePixelType>::
          ReadAffineTransform(fn.c_str());
      m_AffineTransformReadFlags[0] = 1;
    }
    catch (...)
    {
      m_AffineTransformReadFlags[0] = 0;
    }

  }

  // Read image to image transforms
  for (unsigned int i = 1; i < m_ImageFileNames.GetSize(); i++)
  {
    std::string name = mu::get_name(m_ImageFileNames[i].c_str());

    std::string fn =
      firststr + name + suffixstr + std::string(".affine");

    muLogMacro(<< "Reading " << fn << "...\n");

    try
    {
      m_AffineTransforms[i]  =
        PairRegistrationMethod<InternalImagePixelType>::
          ReadAffineTransform(fn.c_str());
      m_AffineTransformReadFlags[i] = 1;
    }
    catch (...)
    {
      m_AffineTransformReadFlags[i] = 0;
    }
  }

  bool allReadOK = true;
  for (unsigned i = 0; i < m_ImageFileNames.GetSize(); i++)
    if (m_AffineTransformReadFlags[i] == 0)
       allReadOK = false;

  // Can assume that registration has been done?
  if (allReadOK)
  {
    m_DoneRegistration = true;
  }

}

template <class TOutputPixel, class TProbabilityPixel>
void
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::SetSuffix(std::string suffix)
{

  m_Suffix = suffix;

  m_Modified = true;

}

template <class TOutputPixel, class TProbabilityPixel>
void
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::SetTemplateFileName(std::string filename)
{

  m_TemplateFileName = filename;

  m_TemplateAffineTransform = AffineTransformType::New();

  m_AffineTransformReadFlags[0] = 0;

  m_DoneRegistration = false;
  m_DoneResample = false;

  m_Modified = true;

}


template <class TOutputPixel, class TProbabilityPixel>
void
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::SetImageFileNames(StringList names)
{

  itkDebugMacro(<< "SetImageFileNames");

  unsigned int numImages = names.GetSize();

  if (numImages == 0)
    itkExceptionMacro(<< "No images specified");
  
  m_ImageFileNames = names;

  m_TemplateAffineTransform = AffineTransformType::New();

  // Clear previous transforms
  m_AffineTransforms.Clear();
  m_AffineTransforms.Allocate(numImages);
  for (unsigned int i = 0; i < numImages; i++)
  {
    // Also append identity matrix for each image
    AffineTransformPointer transform = AffineTransformType::New();
    m_AffineTransforms.Append(transform);
  }

  m_AffineTransformReadFlags = FlagArrayType(numImages);
  for (unsigned int i = 0; i < numImages; i++)
    m_AffineTransformReadFlags[i] = 0;

  m_DoneRegistration = false;
  m_DoneResample = false;

  m_Modified = true;

}

template <class TOutputPixel, class TProbabilityPixel>
void
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::SetProbabilityFileNames(StringList names)
{

  unsigned int numProbabilities = names.GetSize();
  if (numProbabilities == 0)
  {
    itkExceptionMacro(<< "No probability images");
  }

  m_ProbabilityFileNames = names;

  m_DoneResample = false;

  m_Modified = true;

}

template <class TOutputPixel, class TProbabilityPixel>
void
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::SetAtlasOrientation(std::string orient)
{

  m_AtlasOrientation = orient;

  m_DoneRegistration = false;
  m_DoneResample = false;

  m_Modified = true;

}

template <class TOutputPixel, class TProbabilityPixel>
void
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::SetImageOrientations(StringList orientations)
{

  m_ImageOrientations = orientations;

  m_DoneRegistration = false;
  m_DoneResample = false;

  m_Modified = true;
}


template <class TOutputPixel, class TProbabilityPixel>
typename AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::ProbabilityImageList
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::GetProbabilities()
{

  itkDebugMacro(<< "GetProbabilities");

  this->Update();

  return m_Probabilities;

}

template <class TOutputPixel, class TProbabilityPixel>
typename AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::OutputImageList
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::GetImages()
{

  itkDebugMacro(<< "GetImages");

  this->Update();

  return m_Images;

}

template <class TOutputPixel, class TProbabilityPixel>
typename AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::OutputImagePointer
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::GetAffineTemplate()
{
  itkDebugMacro(<< "GetAffineTemplate");

  this->Update();

  return m_AffineTemplate;
}

template <class TOutputPixel, class TProbabilityPixel>
void
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::Update()
{
  itkDebugMacro(<< "Update");

  if (m_Modified)
  {
    this->ReadImages();
    m_DoneRegistration = false;
    m_DoneResample = false;
  }

  if (m_Modified || !m_DoneRegistration)
    this->RegisterImages();

  if (m_Modified || !m_DoneResample)
    this->ResampleImages();

  m_Modified = false;

}

template <class TOutputPixel, class TProbabilityPixel>
typename AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::InternalImagePointer
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::PrefilterImage(
  typename AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
  ::InternalImagePointer& img)
{
  if (m_PrefilteringIterations == 0)
    return img;

  if (m_PrefilteringMethod.compare("Grad aniso diffusion") == 0)
  {
    typedef itk::GradientAnisotropicDiffusionImageFilter<
      InternalImageType, InternalImageType> AnisoFilterType;
    AnisoFilterType::Pointer anisofilt = AnisoFilterType::New();

    anisofilt->SetInput(img);
    anisofilt->SetNumberOfIterations(m_PrefilteringIterations);
    anisofilt->SetTimeStep(m_PrefilteringTimeStep);
    anisofilt->Update();

    return anisofilt->GetOutput();
  }
  else if (m_PrefilteringMethod.compare("Curvature flow") == 0)
  {
    typedef itk::CurvatureFlowImageFilter<
      InternalImageType, InternalImageType> CurvatureFilterType;

    CurvatureFilterType::Pointer cfilt = CurvatureFilterType::New();
    cfilt->SetInput(img);
    cfilt->SetNumberOfIterations(m_PrefilteringIterations);
    cfilt->SetTimeStep(m_PrefilteringTimeStep);
    cfilt->Update();

    return cfilt->GetOutput();
  }
  else
  {
    // No filtering
    return img;
  }

}

template <class TOutputPixel, class TProbabilityPixel>
void
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::ReadImages()
{

  itkDebugMacro(<< "ReadImages");

  this->VerifyInitialization();

  typedef itk::ImageFileReader<InternalImageType> ReaderType;
  typedef typename ReaderType::Pointer ReaderPointer;

  typedef ImageDirectionStandardizer<InternalImageType> DirectionFixerType;
  typedef typename DirectionFixerType::Pointer DirectionFixerPointer;

  typedef PairRegistrationMethod<InternalImagePixelType> PairRegType;

  m_InputImages.Clear();
  m_InputImages.Initialize(m_ImageFileNames.GetSize(), 0);
  for (unsigned int i = 0; i < m_ImageFileNames.GetSize(); i++)
  {
    muLogMacro(
      << "Reading image " << i+1 << ": " << m_ImageFileNames[i] << "...\n");

    ReaderPointer imgreader = ReaderType::New();
    imgreader->SetFileName(m_ImageFileNames[i].c_str());

    imgreader->Update();

    InternalImagePointer img_i = imgreader->GetOutput();

    m_InputImages[i] = img_i;

    DirectionFixerPointer dirstandf = DirectionFixerType::New();
    dirstandf->SetTargetDirectionFromString(
      m_InputImages[0], m_ImageOrientations[0]);
    img_i = dirstandf->Standardize(img_i, m_ImageOrientations[i]);

//TODO PP DEBUG
    //img_i = this->PrefilterImage(img_i);

    m_InputImages[i] = img_i;
  }

}

template <class TOutputPixel, class TProbabilityPixel>
void
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::RegisterImages()
{

  itkDebugMacro(<< "RegisterImages");

  typedef itk::ImageFileReader<InternalImageType> ReaderType;
  typedef typename ReaderType::Pointer ReaderPointer;

  typedef ImageDirectionStandardizer<InternalImageType> DirectionFixerType;
  typedef typename DirectionFixerType::Pointer DirectionFixerPointer;

  typedef PairRegistrationMethod<InternalImagePixelType> PairRegType;

  // Get the first image (for reference)
  InternalImagePointer first = m_InputImages[0];

  // Register template to first image
  if ((m_TemplateFileName.length() != 0)
       &&
      (m_AffineTransformReadFlags[0] == 0))
  {
    if (m_AtlasLinearTransformChoice == ID_TRANSFORM)
    {
      m_TemplateAffineTransform = AffineTransformType::New();
      m_TemplateAffineTransform->SetIdentity();
    }

    itkDebugMacro(<< "Registering template " << m_TemplateFileName << "...");
    ReaderPointer reader = ReaderType::New();
    reader->SetFileName(m_TemplateFileName.c_str());
    reader->Update();

    InternalImagePointer templateImg = reader->GetOutput();

    if (m_AtlasOrientation.length() != 0)
    {
      DirectionFixerPointer dirstandf = DirectionFixerType::New();
      dirstandf->SetTargetDirectionFromString(
        m_InputImages[0], m_ImageOrientations[0]);
      templateImg = dirstandf->Standardize(templateImg, m_AtlasOrientation);
    }

    muLogMacro(<< "Registering template to first image...\n");

    if (m_AtlasLinearTransformChoice == AFFINE_TRANSFORM)
      m_TemplateAffineTransform =
        PairRegistrationMethod<InternalImagePixelType>::
          RegisterAffine(first, templateImg, PairRegType::QuantizeNone);

    if (m_AtlasLinearTransformChoice == RIGID_TRANSFORM)
      m_TemplateAffineTransform =
        PairRegistrationMethod<InternalImagePixelType>::
          RegisterRigid(first, templateImg, PairRegType::QuantizeNone);

  }

  // Register each image to first image
  for (unsigned int i = 1; i < m_ImageFileNames.GetSize(); i++)
  {
    if (m_AffineTransformReadFlags[i] != 0)
      continue;

    if (m_ImageLinearTransformChoice == ID_TRANSFORM)
    {
      m_AffineTransforms[i] = AffineTransformType::New();
      m_AffineTransforms[i]->SetIdentity();
      continue;
    }

    muLogMacro(<< "Registering image " << i+1 << " to first image...\n");

    InternalImagePointer img_i = m_InputImages[i];

    if (m_ImageLinearTransformChoice == AFFINE_TRANSFORM)
      m_AffineTransforms[i] = 
        PairRegistrationMethod<InternalImagePixelType>::
          RegisterAffine(first, img_i, PairRegType::QuantizeNone);

    if (m_ImageLinearTransformChoice == RIGID_TRANSFORM)
      m_AffineTransforms[i] = 
        PairRegistrationMethod<InternalImagePixelType>::
          RegisterRigid(first, img_i, PairRegType::QuantizeNone);

  }

  m_DoneRegistration = true;

}

template <class TOutputPixel, class TProbabilityPixel>
void
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::ResampleImages()
{

  itkDebugMacro(<< "ResampleImages");

  if (!m_DoneRegistration)
    return;

  // Define the internal reader type
  typedef itk::ImageFileReader<InternalImageType> ReaderType;
  typedef typename ReaderType::Pointer ReaderPointer;

  // Orientation standardizer
  typedef ImageDirectionStandardizer<InternalImageType> DirectionFixerType;
  typedef typename DirectionFixerType::Pointer DirectionFixerPointer;

  // Get the first image (for reference)
  InternalImagePointer first = m_InputImages[0];

  typedef itk::ResampleImageFilter<InternalImageType, InternalImageType>
    ResampleType;
  typedef typename ResampleType::Pointer ResamplePointer;

  typedef itk::LinearInterpolateImageFunction<InternalImageType, double>
    LinearInterpolatorType;
  typedef itk::BSplineInterpolateImageFunction<InternalImageType, double, double>
    SplineInterpolatorType;

  typename LinearInterpolatorType::Pointer linearInt =
    LinearInterpolatorType::New();

  // Spline interpolation, only available for input images, not atlas
  typename SplineInterpolatorType::Pointer splineInt =
    SplineInterpolatorType::New();
  splineInt->SetSplineOrder(5);

  // Resample the template
  if (m_TemplateFileName.length() != 0)
  {
    muLogMacro(<< "Resampling template...\n");

    ReaderPointer reader = ReaderType::New();
    reader->SetFileName(m_TemplateFileName.c_str());
    reader->Update();

    InternalImagePointer templateImg = reader->GetOutput();

    if (m_AtlasOrientation.length() != 0)
    {
      DirectionFixerPointer dirstandf = DirectionFixerType::New();
      dirstandf->SetTargetDirectionFromString(
        m_InputImages[0], m_ImageOrientations[0]);
      templateImg = dirstandf->Standardize(templateImg, m_AtlasOrientation);
    }

    ResamplePointer resampler = ResampleType::New();

    resampler->SetInput(templateImg);
    resampler->SetTransform(m_TemplateAffineTransform);

    resampler->SetInterpolator(linearInt);
    resampler->SetOutputParametersFromImage(first);
    resampler->SetDefaultPixelValue(0);

    resampler->Update();

    m_AffineTemplate = CopyOutputImage(resampler->GetOutput());

  }

//TODO
// HACK
  // Resample the "other" template
  if (m_OtherTemplateFileName.length() != 0)
  {
    muLogMacro(<< "Resampling other template...\n");

    ReaderPointer reader = ReaderType::New();
    reader->SetFileName(m_OtherTemplateFileName.c_str());
    reader->Update();

    InternalImagePointer templateImg = reader->GetOutput();

    if (m_AtlasOrientation.length() != 0)
    {
      DirectionFixerPointer dirstandf = DirectionFixerType::New();
      dirstandf->SetTargetDirectionFromString(
        m_InputImages[0], m_ImageOrientations[0]);
      templateImg = dirstandf->Standardize(templateImg, m_AtlasOrientation);
    }

    ResamplePointer resampler = ResampleType::New();

    resampler->SetInput(templateImg);
    resampler->SetTransform(m_TemplateAffineTransform);

    resampler->SetInterpolator(linearInt);
    resampler->SetOutputParametersFromImage(first);
    resampler->SetDefaultPixelValue(0);

    resampler->Update();

    m_OtherAffineTemplate = CopyOutputImage(resampler->GetOutput());
  }

  // Resample the probabilities
  for (unsigned int i = 0; i < m_Probabilities.GetSize(); i++)
    m_Probabilities[i] = 0;
  m_Probabilities.Clear();
  for (unsigned int i = 0; i < m_ProbabilityFileNames.GetSize(); i++)
  {
    muLogMacro(<< "Resampling atlas prior " << i+1 << "...\n");

    ReaderPointer reader = ReaderType::New();
    reader->SetFileName(m_ProbabilityFileNames[i].c_str());
    reader->Update();

    InternalImagePointer prob_i = reader->GetOutput();

    if (m_AtlasOrientation.length() != 0)
    {
      DirectionFixerPointer dirstandf = DirectionFixerType::New();
      dirstandf->SetTargetDirectionFromString(
        m_InputImages[0], m_ImageOrientations[0]);
      prob_i = dirstandf->Standardize(prob_i, m_AtlasOrientation);
    }

    ResamplePointer resampler = ResampleType::New();

    resampler->SetInput(prob_i);
    resampler->SetTransform(m_TemplateAffineTransform);

    resampler->SetInterpolator(linearInt);
    resampler->SetOutputParametersFromImage(first);
    resampler->SetDefaultPixelValue(0);

    resampler->Update();
  
    m_Probabilities.Append(CopyProbabilityImage(resampler->GetOutput()));
  }

  if (m_Probabilities.GetSize() > 0)
  {
    // Normalize probabilities
    typedef itk::ImageRegionIteratorWithIndex<ProbabilityImageType>
      ProbabilityIteratorType;
    ProbabilityIteratorType it(
      m_Probabilities[0], m_Probabilities[0]->GetLargestPossibleRegion());
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      ProbabilityImageIndexType ind = it.GetIndex();

      double sump = 1e-20;
      for (unsigned int k = 0; k < m_Probabilities.GetSize(); k++)
        sump += m_Probabilities[k]->GetPixel(ind);

      for (unsigned int k = 0; k < m_Probabilities.GetSize(); k++)
        m_Probabilities[k]->SetPixel(ind,
          m_Probabilities[k]->GetPixel(ind) / sump);
    }
  }

  // Clear image list
  for (unsigned int i = 0; i < m_Images.GetSize(); i++)
    m_Images[i] = 0;
  m_Images.Clear();

  // Do nothing for first image
  m_Images.Append(CopyOutputImage(first));

  // The FOV mask, regions where intensities in all channels do not
  // match FOV code
  m_FOVMask = ByteImageType::New();
  m_FOVMask->CopyInformation(m_Images[0]);
  m_FOVMask->SetRegions(m_Images[0]->GetLargestPossibleRegion());
  m_FOVMask->Allocate();

  typedef itk::ImageRegionIterator<ByteImageType> MaskIteratorType;
  MaskIteratorType maskIt(m_FOVMask, m_FOVMask->GetLargestPossibleRegion());

  maskIt.GoToBegin();
  while (!maskIt.IsAtEnd())
  {
    maskIt.Set(1);
    ++maskIt;
  }

  // Resample the other images
  for (unsigned int i = 1; i < m_ImageFileNames.GetSize(); i++)
  {
    muLogMacro(<< "Resampling input image " << i+1 << "...\n");

    ResamplePointer resampler = ResampleType::New();

    resampler->SetInput(m_InputImages[i]);
    resampler->SetTransform(m_AffineTransforms[i]);

    if (m_UseNonLinearInterpolation)
      resampler->SetInterpolator(splineInt);
    else
      resampler->SetInterpolator(linearInt);

    resampler->SetDefaultPixelValue(m_OutsideFOVCode);
    resampler->SetOutputParametersFromImage(first);

    resampler->Update();

    InternalImagePointer tmp = resampler->GetOutput();

    // Zero the mask region outside FOV and also the intensities with outside
    // FOV code
    typedef itk::ImageRegionIterator<InternalImageType> InternalIteratorType;

    InternalIteratorType tmpIt(tmp, first->GetLargestPossibleRegion());

    maskIt.GoToBegin();
    tmpIt.GoToBegin();
    while (!maskIt.IsAtEnd())
    {
      if (tmpIt.Get() == m_OutsideFOVCode)
      {
        maskIt.Set(0);
        tmpIt.Set(0);
      }
      ++maskIt;
      ++tmpIt;
    }

    // Add the image
    m_Images.Append(CopyOutputImage(tmp));
  }

  m_DoneResample = true;

}

template <class TOutputPixel, class TProbabilityPixel>
typename AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::OutputImagePointer
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::CopyOutputImage(
  typename AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
  ::InternalImagePointer img)
{

  itkDebugMacro(<< "CopyOutputImage");

  OutputImagePointer outimg = OutputImageType::New();
  outimg->CopyInformation(img);
  outimg->SetRegions(img->GetLargestPossibleRegion());
  outimg->Allocate();

  typedef itk::ImageRegionIterator<InternalImageType> InternalIteratorType;
  InternalIteratorType inputIter(img, img->GetLargestPossibleRegion());

  typedef itk::ImageRegionIterator<OutputImageType> OutputIteratorType;
  OutputIteratorType outputIter(outimg, outimg->GetLargestPossibleRegion());

  inputIter.GoToBegin();
  outputIter.GoToBegin();

  while (!inputIter.IsAtEnd())
  {
    outputIter.Set(static_cast<OutputImagePixelType>(inputIter.Get()));
    ++inputIter;
    ++outputIter;
  }

  return outimg;

}

template <class TOutputPixel, class TProbabilityPixel>
typename AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::ProbabilityImagePointer
AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
::CopyProbabilityImage(
  typename AtlasRegistrationMethod<TOutputPixel, TProbabilityPixel>
  ::InternalImagePointer img)
{

  itkDebugMacro(<< "CopyProbabilityImage");

  ProbabilityImagePointer outimg = ProbabilityImageType::New();
  outimg->CopyInformation(img);
  outimg->SetRegions(img->GetLargestPossibleRegion());
  outimg->Allocate();

  typedef itk::ImageRegionIterator<InternalImageType> InternalIteratorType;
  InternalIteratorType inputIter(img, img->GetLargestPossibleRegion());

  typedef itk::ImageRegionIterator<ProbabilityImageType>
    ProbabilityIteratorType;
  ProbabilityIteratorType outputIter(outimg,
    outimg->GetLargestPossibleRegion());

  inputIter.GoToBegin();
  outputIter.GoToBegin();

  while (!inputIter.IsAtEnd())
  {
    double p = inputIter.Get();
    if (p < 0.0)
      p = 0.0;
    outputIter.Set(static_cast<ProbabilityImagePixelType>(p));
    ++inputIter;
    ++outputIter;
  }

  return outimg;

}


#endif
