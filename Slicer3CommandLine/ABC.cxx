

// Atlas based segmentation
// ABC outdir --inputImage img1 ... --inputImage imgn atlasdir --atlasOrient RAI -b 4 --atlasFluidIters 5

#include "ABCCLP.h"

#include "mu.h"
#include "muFile.h"

#include "itkOutputWindow.h"
#include "itkTextOutput.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNumericTraits.h"
#include "itkResampleImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"

#include "DynArray.h"

// Use manually instantiated classes for the big program chunks
#define MU_MANUAL_INSTANTIATION
#include "AtlasRegistrationMethod.h"
#include "EMSegmentationFilter.h"
#include "PairRegistrationMethod.h"
#undef MU_MANUAL_INSTANTIATION

#include <exception>
#include <iostream>

#include <string>


typedef std::vector<std::string> StringList;

int run_ABC(int argc, char** argv)
{

  PARSE_ARGS;

  typedef itk::Image<unsigned char, 3> ByteImageType;
  typedef itk::Image<float, 3> FloatImageType;
  typedef itk::Image<short, 3> ShortImageType;

  typedef itk::ImageFileReader<FloatImageType> ReaderType;

  // Make sure last character in atlas directory string is a separator
  if (atlasDir[atlasDir.size()-1] != MU_DIR_SEPARATOR)
    atlasDir += "/";

  // Set up suffix string for images
  std::string suffstr = "_seg.mha";

  std::cout << "Reading input images: " << std::endl;

  DynArray<std::string> inputFiles;
  if (inputImage1.size() != 0)
    inputFiles.Append(inputImage1);
  if (inputImage2.size() != 0)
    inputFiles.Append(inputImage2);
  if (inputImage3.size() != 0)
    inputFiles.Append(inputImage3);
  if (inputImage4.size() != 0)
    inputFiles.Append(inputImage4);
  if (inputImage5.size() != 0)
    inputFiles.Append(inputImage5);

  DynArray<std::string> inputOrients;
/*
  if (inputOrient1.size() != 0)
    inputOrients.Append(inputOrient1);
  if (inputOrient2.size() != 0)
    inputOrients.Append(inputOrient2);
  if (inputOrient3.size() != 0)
    inputOrients.Append(inputOrient3);
  if (inputOrient4.size() != 0)
    inputOrients.Append(inputOrient4);
*/
  for (int i = 0; i < inputFiles.GetSize(); i++)
    inputOrients.Append(std::string("file"));

  muLogMacro(<< "Registering images using affine transform...\n");

  ByteImageType::Pointer fovmask;

  FloatImageType::Pointer templateImage;

  DynArray<FloatImageType::Pointer> images;
  DynArray<FloatImageType::Pointer> priors;
  {
    typedef AtlasRegistrationMethod<float, float> AtlasRegType;
    AtlasRegType::Pointer atlasreg = AtlasRegType::New();

    atlasreg->SetPrefilteringMethod("Curvature flow");
    atlasreg->SetPrefilteringIterations(1);
    atlasreg->SetPrefilteringTimeStep(0.01);

    //TODO: allow parameter to set output names?
    atlasreg->SetSuffix("");

    std::string templatefn = atlasDir + std::string("template.mha");
    atlasreg->SetTemplateFileName(templatefn);

    atlasreg->SetAtlasOrientation(atlasOrient);

    atlasreg->SetImageFileNames(inputFiles);
    atlasreg->SetImageOrientations(inputOrients);
    // NOTE: trafo write should be disabled for Slicer
    atlasreg->SetOutputDirectory(std::string(""));

    if (atlasMapType.compare("identity") == 0)
      atlasreg->SetAtlasLinearTransformChoice(AtlasRegType::ID_TRANSFORM);
    if (atlasMapType.compare("rigid") == 0)
      atlasreg->SetAtlasLinearTransformChoice(AtlasRegType::RIGID_TRANSFORM);

    if (coregMapType.compare("identity") == 0)
      atlasreg->SetImageLinearTransformChoice(AtlasRegType::ID_TRANSFORM);
    if (coregMapType.compare("rigid") == 0)
      atlasreg->SetImageLinearTransformChoice(AtlasRegType::RIGID_TRANSFORM);

    // Location of the priors (1.mha, 2.mha, ... 99.mha, ... etc)
    atlasreg->SetAtlasDirectory(atlasDir);

    // NOTE: always start from scratch in Slicer
    //muLogMacro(<< "Attempting to read previous registration results..."
    //  << std::endl);
    //atlasreg->ReadParameters();

    muLogMacro(<< "Registering and resampling images..." << std::endl);
    atlasreg->Update();

    // NOTE: Disable write, unless you have outdir?
    //atlasreg->WriteParameters();

    fovmask = atlasreg->GetFOVMask();

    images = atlasreg->GetImages();
    priors = atlasreg->GetProbabilities();

    templateImage = atlasreg->GetAffineTemplate();
  } // end atlas reg block


  std:: cout << "Start segmentation..." << std::endl;
  typedef EMSegmentationFilter<FloatImageType, FloatImageType> SegFilterType;
  SegFilterType::Pointer segfilter = SegFilterType::New();

  segfilter->SetTemplateImage(templateImage);

  segfilter->SetInputImages(images);
  segfilter->SetPriors(priors);

  SegFilterType::VectorType priorweights(priorAdjustVec.size());
  for (unsigned int i = 0; i < priorAdjustVec.size(); i++)
    priorweights[i] = priorAdjustVec[i];
  segfilter->SetPriorWeights(priorweights);

  segfilter->SetMaxBiasDegree(biasDegree);

  bool dowarp = (atlasFluidIters != 0);

  if (dowarp)
    segfilter->WarpingOn();
  else
    segfilter->WarpingOff();

  segfilter->SetWarpFluidIterations(atlasFluidIters);

  segfilter->Update();

  std::cout << "Writing segmentation images..." << std::endl;

  {
    typedef itk::ImageFileWriter<ByteImageType> ByteWriterType;
    ByteWriterType::Pointer writer = ByteWriterType::New();

    writer->SetFileName(labelImage.c_str());
    writer->SetInput(segfilter->GetOutput());
    writer->Update();
  }

  // Write registered - bias corrected images

  DynArray<std::string> outputFiles;
  outputFiles.Append(outputImage1);
  outputFiles.Append(outputImage2);
  outputFiles.Append(outputImage3);
  outputFiles.Append(outputImage4);

  DynArray<FloatImageType::Pointer> corrImages = segfilter->GetCorrected();

  for (unsigned int i = 0; i < corrImages.GetSize(); i++)
  {
    if (outputFiles[i].size() == 0)
      continue;

    typedef itk::RescaleIntensityImageFilter<FloatImageType, ShortImageType>
      RescalerType;
    RescalerType::Pointer resf = RescalerType::New();
    resf->SetInput(corrImages[i]);
    resf->SetOutputMinimum(0);
    resf->SetOutputMaximum(32000);
    resf->Update();

    typedef itk::ImageFileWriter<ShortImageType> ShortWriterType;
    ShortWriterType::Pointer writer = ShortWriterType::New();

    writer->SetInput(resf->GetOutput());
    writer->SetFileName(outputFiles[i].c_str());
    writer->Update();
  }
  

/*
  // Write brain class posteriors
  DynArray<ShortImageType::Pointer> posteriors =
    segfilter->GetShortPosteriors();

  for (unsigned int i = 0; i < (posteriors.GetSize()-3); i++)
  {
    std::ostringstream oss;
    oss << outputDir << mu::get_name(inputFiles[0].c_str())
      << "_posterior" << i << suffstr << std::ends;

    typedef itk::ImageFileWriter<ShortImageType> ShortWriterType;
    ShortWriterType::Pointer writer = ShortWriterType::New();

    writer->SetInput(posteriors[i]);
    writer->SetFileName(oss.str().c_str());
    writer->Update();
  }
*/

  return 0;

}

int
main(int argc, char** argv)
{

  PARSE_ARGS;

  itk::OutputWindow::SetInstance(itk::TextOutput::New());

  try
  {
    run_ABC(argc, argv);
  }
  catch (itk::ExceptionObject& e)
  {
    std::cerr << e << std::endl;
    return EXIT_FAILURE;
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }
  catch (std::string& s)
  {
    std::cerr << "Exception: " << s << std::endl;
    return EXIT_FAILURE;
  }
  catch (...)
  {
    std::cerr << "Unknown exception" << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;

}
