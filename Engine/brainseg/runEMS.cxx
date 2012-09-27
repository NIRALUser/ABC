
#include "itkCastImageFilter.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkNumericTraits.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkVersion.h"

#include "itksys/SystemTools.hxx"

#include "EMSParameters.h"
#include "EMSParametersXMLFile.h"

#include "DynArray.h"
#include "Log.h"
#include "MersenneTwisterRNG.h"
#include "Timer.h"

#include "muFile.h"

// Use manually instantiated classes for the big program chunks
#define MU_MANUAL_INSTANTIATION
#include "EMSegmentationFilter.h"
#include "AtlasRegistrationMethod.h"
#include "PairRegistrationMethod.h"
#undef MU_MANUAL_INSTANTIATION

#include "runEMS.h"

#include <iostream>
#include <string>
#include <sstream>

#include <stdlib.h>

typedef itk::Image<float, 3> FloatImageType;
typedef itk::Image<unsigned char, 3> ByteImageType;
typedef itk::Image<short, 3> ShortImageType;

typedef FloatImageType::Pointer FloatImagePointer;
typedef ByteImageType::Pointer ByteImagePointer;
typedef ShortImageType::Pointer ShortImagePointer;

void
runEMS(EMSParameters* emsp, bool debugflag, bool writemoreflag)
{

  if (!emsp->CheckValues())
    throw std::string("Invalid segmentation parameter values");
  
  // Create and start a new timer (for the whole process)
  Timer* timer = new Timer();

  // Initialize random number generators
  srand(542948474);
  MersenneTwisterRNG* rng = MersenneTwisterRNG::GetGlobalInstance();
  rng->Initialize(87584359);

  // Directory separator string
  std::string separator = std::string("/");
  separator[0] = MU_DIR_SEPARATOR;

  // Get output directory
  std::string outdir = emsp->GetOutputDirectory();
  // Make sure last character in output directory string is a separator
  if (outdir[outdir.size()-1] != MU_DIR_SEPARATOR)
    outdir += separator;

  // Create the output directory, stop if it does not exist
  if(!mu::create_dir(outdir.c_str()))
    return;

  // Set up the logger
  {
    std::string logfn = outdir + emsp->GetSuffix() + ".log";
    (mu::Log::GetInstance())->EchoOn();
    (mu::Log::GetInstance())->SetOutputFileName(logfn.c_str());
  }

  // Write out the parameters in XML
  {
    std::string xmlfn = outdir + emsp->GetSuffix() + ".xml";
    writeEMSParametersXML(xmlfn.c_str(), emsp);
  }

  // Set up suffix string for images
  std::string fmt = emsp->GetOutputFormat();
  std::string outext = ".mha";
  if (itksys::SystemTools::Strucmp(fmt.c_str(), "Analyze") == 0)
    outext = ".hdr";
  else if (itksys::SystemTools::Strucmp(fmt.c_str(), "GIPL") == 0)
    outext = ".gipl";
  else if (itksys::SystemTools::Strucmp(fmt.c_str(), "Nrrd") == 0)
    outext = ".nrrd";
  else if (itksys::SystemTools::Strucmp(fmt.c_str(), "Meta") == 0)
    outext = ".mha";
  else if (itksys::SystemTools::Strucmp(fmt.c_str(), "Nifti") == 0)
    outext = ".nii";
  else
  {
    muLogMacro(<< "WARNING: output format unrecognized, using Meta format\n");
  }
  std::string suffstr =
    std::string("_") + std::string(emsp->GetSuffix()) + outext;
  std::string metasuffstr =
    std::string("_") + std::string(emsp->GetSuffix()) + std::string(".mha");

  std::vector<double> prWeights = emsp->GetPriorWeights();

  muLogMacro(<< "ABC: Atlas Based Classification\n");
  muLogMacro(<< "========================================\n");
  muLogMacro(<< "Program compiled on: " << __DATE__ << "\n");
  muLogMacro(<< "\n");

  muLogMacro(<< "Marcel Prastawa - prastawa@sci.utah.edu\n");
  muLogMacro(<< "This software is provided for research purposes only\n");
  muLogMacro(<< "\n");

  muLogMacro(<< "Using ITK version "
    << itk::Version::GetITKMajorVersion() << "."
    << itk::Version::GetITKMinorVersion() << "."
    << itk::Version::GetITKBuildVersion() <<  "\n");
  muLogMacro(<< "\n");

  // Write input parameters
  muLogMacro(<< "=== Parameters ===\n");
  muLogMacro(<< "Suffix: " << emsp->GetSuffix() << "\n");
  muLogMacro(<< "Atlas Directory: " << emsp->GetAtlasDirectory() << "\n");
  muLogMacro(<< "Atlas Orientation: " << emsp->GetAtlasOrientation() << "\n");
  muLogMacro(<< "Output Directory: " << emsp->GetOutputDirectory() << "\n");
  muLogMacro(<< "Output Format: " << emsp->GetOutputFormat() << "\n");
  muLogMacro(<< "Input images: \n");
  for (unsigned int i = 0; i < emsp->GetImages().GetSize(); i++)
    muLogMacro(<< "  " << "[" << (emsp->GetImageOrientations())[i] << "] " <<
      (emsp->GetImages())[i] << "\n");
  muLogMacro(
    << "Non-linear filtering, method: " << emsp->GetFilterMethod() << ", "
    << emsp->GetFilterIterations()
    << " iterations, dt = " << emsp->GetFilterTimeStep() << "\n");
  muLogMacro(
    << "Prior weight scales: " );
  for (unsigned int i = 0; i < prWeights.size(); i++)
    muLogMacro(<< prWeights[i] << " ");
  muLogMacro(<< "\n");
  muLogMacro(
    << "Max bias polynomial degree: " << emsp->GetMaxBiasDegree() << "\n");
  muLogMacro(<< "Initial Distribution Estimator: " << emsp->GetInitialDistributionEstimator() << "\n");
  muLogMacro(<< "Atlas warping: " << emsp->GetDoAtlasWarp() << "\n");
  muLogMacro(
    << "Atlas warp fluid iterations: " << emsp->GetAtlasWarpFluidIterations()
    << "\n");
  muLogMacro(<< "Atlas warp fluid max step: " << emsp->GetAtlasWarpFluidMaxStep() << "\n");
  muLogMacro(<< "\n");

  muLogMacro(<< "=== Start ===\n");

  //Get atlas directory
  std::string atlasdir = emsp->GetAtlasDirectory();
  // Make sure last character is a separator
  if (atlasdir[atlasdir.size()-1] != MU_DIR_SEPARATOR)
    atlasdir += separator;

  muLogMacro(<< "Registering images using affine transform...\n");

  ByteImagePointer fovmask;

  FloatImagePointer templateImg;

  DynArray<FloatImagePointer> images;
  DynArray<FloatImagePointer> priors;
  {
    typedef AtlasRegistrationMethod<float, float> AtlasRegType;
    AtlasRegType::Pointer atlasreg = AtlasRegType::New();

    if (debugflag)
      atlasreg->DebugOn();

    atlasreg->SetPrefilteringMethod(emsp->GetFilterMethod().c_str());
    atlasreg->SetPrefilteringIterations(emsp->GetFilterIterations());
    atlasreg->SetPrefilteringTimeStep(emsp->GetFilterTimeStep());

    atlasreg->SetSuffix(emsp->GetSuffix());

    atlasreg->SetAtlasOrientation(emsp->GetAtlasOrientation());

    atlasreg->SetImageFileNames(emsp->GetImages());
    atlasreg->SetImageOrientations(emsp->GetImageOrientations());
    atlasreg->SetOutputDirectory(outdir);

    std::string atlasmapstr = emsp->GetAtlasLinearMapType();
    if (atlasmapstr.compare("id") == 0)
      atlasreg->SetAtlasLinearTransformChoice(AtlasRegType::ID_TRANSFORM);
    if (atlasmapstr.compare("rigid") == 0)
      atlasreg->SetAtlasLinearTransformChoice(AtlasRegType::RIGID_TRANSFORM);

    std::string imagemapstr = emsp->GetImageLinearMapType();
    if (imagemapstr.compare("id") == 0)
      atlasreg->SetImageLinearTransformChoice(AtlasRegType::ID_TRANSFORM);
    if (imagemapstr.compare("rigid") == 0)
      atlasreg->SetImageLinearTransformChoice(AtlasRegType::RIGID_TRANSFORM);

    // Directory with the template and priors (template.mha, 1.mha, ... 99.mha)
    atlasreg->SetAtlasDirectory(atlasdir);

    muLogMacro(<< "Attempting to read previous registration results..."
      << std::endl);
    atlasreg->ReadParameters();

    muLogMacro(<< "Registering and resampling images..." << std::endl);
    Timer* regtimer = new Timer();
    atlasreg->Update();

    atlasreg->WriteParameters();
    regtimer->Stop();

    muLogMacro(<< "Registration took " << regtimer->GetElapsedHours() << " hours, ");
    muLogMacro(<< regtimer->GetElapsedMinutes() << " minutes, ");
    muLogMacro(<< regtimer->GetElapsedSeconds() << " seconds\n");
    delete regtimer;

    fovmask = atlasreg->GetFOVMask();

    images = atlasreg->GetImages();
    priors = atlasreg->GetProbabilities();

    templateImg = atlasreg->GetAffineTemplate();

    // Write the registered template and images
    if (writemoreflag)
    {
      muLogMacro(<< "Writing affine-registered template...\n");

      typedef itk::RescaleIntensityImageFilter<FloatImageType, ByteImageType>
        ByteRescaleType;

      ByteRescaleType::Pointer rescaler = ByteRescaleType::New();
      rescaler->SetOutputMinimum(0);
      rescaler->SetOutputMaximum(255);
      rescaler->SetInput(atlasreg->GetAffineTemplate());
      rescaler->Update();

      typedef itk::ImageFileWriter<ByteImageType> ByteWriterType;
      ByteWriterType::Pointer writer = ByteWriterType::New();

      std::string fn =
        outdir + mu::get_name((emsp->GetImages()[0]).c_str()) +
        std::string("_template_affine") + suffstr;

      writer->SetInput(rescaler->GetOutput());
      writer->SetFileName(fn.c_str());
      writer->UseCompressionOn();
      writer->Update();

      for (unsigned int i = 0; i < images.GetSize(); i++)
      {
        typedef itk::RescaleIntensityImageFilter<FloatImageType, ShortImageType>
          ShortRescaleType;

        ShortRescaleType::Pointer rescaler = ShortRescaleType::New();
        rescaler->SetOutputMinimum(0);
        rescaler->SetOutputMaximum(itk::NumericTraits<short>::max());
        rescaler->SetInput(images[i]);
        rescaler->Update();

        std::string fn =
          outdir + mu::get_name((emsp->GetImages()[i]).c_str()) +
          std::string("_registered") + suffstr;

        typedef itk::ImageFileWriter<ShortImageType> ShortWriterType;
        ShortWriterType::Pointer writer = ShortWriterType::New();

        writer->SetInput(rescaler->GetOutput());
        writer->SetFileName(fn.c_str());
        writer->UseCompressionOn();
        writer->Update();
      }
    }
  } // end atlas reg block


  muLogMacro(<< "Rescale intensity of filtered images...\n");
  {
    typedef itk::RescaleIntensityImageFilter<FloatImageType, FloatImageType>
      RescaleType;
    RescaleType::Pointer rescaler = RescaleType::New();
    rescaler->SetOutputMinimum(1);
    rescaler->SetOutputMaximum(4096);

    FloatImageType::SizeType size =
      images[0]->GetLargestPossibleRegion().GetSize();
    FloatImageType::IndexType ind;

    for (unsigned int i = 0; i < images.GetSize(); i++)
    {
      FloatImagePointer tmp = images[i];

      rescaler->SetInput(tmp);
      rescaler->Update();

      FloatImagePointer rImg = rescaler->GetOutput();
      for (ind[2] = 0; ind[2] < (long)size[2]; ind[2]++)
        for (ind[1] = 0; ind[1] < (long)size[1]; ind[1]++)
          for (ind[0] = 0; ind[0] < (long)size[0]; ind[0]++)
          {
            tmp->SetPixel(ind, rImg->GetPixel(ind));
          }
    }
  }

  muLogMacro(<< "Start segmentation...\n");
  typedef EMSegmentationFilter<FloatImageType, FloatImageType> SegFilterType;
  SegFilterType::Pointer segfilter = SegFilterType::New();

  if (debugflag)
    segfilter->DebugOn();

  segfilter->SetTemplateImage(templateImg);

  segfilter->SetInputImages(images);
  segfilter->SetPriors(priors);

  segfilter->SetFOVMask(fovmask);

  SegFilterType::VectorType prWeightsVec(prWeights.size());
  for (unsigned int i = 0; i < prWeights.size(); i++)
    prWeightsVec[i] = prWeights[i];
  segfilter->SetPriorWeights(prWeightsVec);

  segfilter->SetMaxBiasDegree(emsp->GetMaxBiasDegree());

  segfilter->SetInitialDistributionEstimator(emsp->GetInitialDistributionEstimator());

  if(emsp->GetDoAtlasWarp())
    segfilter->WarpingOn();
  else
    segfilter->WarpingOff();
  segfilter->SetWarpFluidIterations(emsp->GetAtlasWarpFluidIterations());
  segfilter->SetWarpFluidMaxStep(emsp->GetAtlasWarpFluidMaxStep());
  segfilter->Update();

  DynArray<std::string> names = emsp->GetImages();

  // Write the labels
  muLogMacro(<< "Writing labels...\n");
  {
    typedef itk::ImageFileWriter<ByteImageType> ByteWriterType;
    ByteWriterType::Pointer writer = ByteWriterType::New();

    writer->SetInput(segfilter->GetOutput());

    std::string fn = outdir + mu::get_name(names[0].c_str()) + std::string("_labels") + suffstr;
    writer->SetFileName(fn.c_str());
    writer->UseCompressionOn();
    writer->Update();
  }

  // Write the secondary outputs
  if (writemoreflag)
  {
    muLogMacro(<< "Writing filtered and bias corrected images...\n");
    DynArray<FloatImagePointer> imgset = segfilter->GetCorrected();
    for (unsigned i = 0; i < imgset.GetSize(); i++)
    {
      typedef itk::CastImageFilter<FloatImageType, ShortImageType> CasterType;
      CasterType::Pointer caster = CasterType::New();

      caster->SetInput(imgset[i]);
      caster->Update();
      std::string fn =
        outdir + mu::get_name(names[i].c_str()) + std::string("_corrected")
        + suffstr;

      typedef itk::ImageFileWriter<ShortImageType> ShortWriterType;
      ShortWriterType::Pointer writer = ShortWriterType::New();

      writer->SetInput(caster->GetOutput());
      writer->SetFileName(fn.c_str());
      writer->UseCompressionOn();
      writer->Update();
    }

    // Short posteriors
    muLogMacro(<< "Writing posterior images...\n");
    DynArray<ShortImagePointer> probset = segfilter->GetShortPosteriors();
    for (unsigned int i = 0; i < (probset.GetSize()-3); i++)
    {
      typedef itk::ImageFileWriter<ShortImageType> ShortWriterType;
      ShortWriterType::Pointer writer = ShortWriterType::New();

      std::string first = outdir + mu::get_name(names[0].c_str());

      std::ostringstream oss;
      oss << first << "_posterior" << i << suffstr << std::ends;

      writer->SetInput(probset[i]);
      writer->SetFileName(oss.str().c_str());
      writer->UseCompressionOn();
      writer->Update();
    }

/*
    // Byte posteriors
    muLogMacro(<< "Writing posterior images...\n");
    DynArray<ByteImagePointer> probset = segfilter->GetBytePosteriors();
    for (unsigned i = 0; i < (probset.GetSize()-3); i++)
    {
      typedef itk::ImageFileWriter<ByteImageType> ByteWriterType;
      ByteWriterType::Pointer writer = ByteWriterType::New();

      std::string first = outdir + mu::get_name(names[0].c_str());

      std::ostringstream oss;
      oss << first << "_posterior" << i << suffstr << std::ends;

      writer->SetInput(probset[i]);
      writer->SetFileName(oss.str().c_str());
      writer->UseCompressionOn();
      writer->Update();
    }

    // Floating point posteriors
    muLogMacro(<< "Writing posterior images...\n");
    DynArray<FloatImagePointer> probset = segfilter->GetPosteriors();
    for (unsigned i = 0; i < (probset.GetSize()-3); i++)
    {
      typedef itk::ImageFileWriter<FloatImageType> FloatWriterType;
      FloatWriterPointer writer = FloatWriterType::New();

      std::string first = outdir + mu::get_name(names[0].c_str());

      std::ostringstream oss;
      oss << first << "_posterior" << i << metasuffstr << std::ends;

      writer->SetInput(probset[i]);
      writer->SetFileName(oss.str().c_str());
      writer->UseCompressionOn();
      writer->Update();
    }
*/

  }

  // Write warped template and bspline trafo
  if(emsp->GetDoAtlasWarp())
  {
    typedef itk::RescaleIntensityImageFilter<FloatImageType, ByteImageType>
      ByteRescaleType;

    ByteRescaleType::Pointer rescaler = ByteRescaleType::New();
    rescaler->SetOutputMinimum(0);
    rescaler->SetOutputMaximum(255);
    rescaler->SetInput(segfilter->GetWarpedTemplateImage());
    rescaler->Update();

    typedef itk::ImageFileWriter<ByteImageType> ByteWriterType;
    ByteWriterType::Pointer writer = ByteWriterType::New();

    std::string fn =
      outdir + mu::get_name((emsp->GetImages()[0]).c_str()) +
      std::string("_template_warped") + suffstr;

    writer->SetInput(rescaler->GetOutput());
    writer->SetFileName(fn.c_str());
    writer->UseCompressionOn();
    writer->Update();

/*
// Deprecated
    fn =
      outdir + mu::get_name((emsp->GetImages()[0]).c_str()) +
      std::string("_to_template") + 
      std::string("_") + std::string(emsp->GetSuffix()) + ".bspline";

    PairRegistrationMethod<float>::WriteBSplineTransform(
      fn.c_str(), segfilter->GetTemplateBSplineTransform());

*/
    fn =
      outdir + mu::get_name((emsp->GetImages()[0]).c_str()) +
      std::string("_to_template") + 
      std::string("_dispF_") + std::string(emsp->GetSuffix()) + ".mha";

    typedef itk::ImageFileWriter<SegFilterType::DeformationFieldType>
      DeformationWriterType;
    DeformationWriterType::Pointer defwriter = DeformationWriterType::New();
    defwriter->SetInput(segfilter->GetTemplateFluidDeformation());
    defwriter->SetFileName(fn.c_str());
    defwriter->Update();
  }

  timer->Stop();

  muLogMacro(<< "All segmentation processes took " << timer->GetElapsedHours() << " hours, ");
  muLogMacro(<< timer->GetElapsedMinutes() << " minutes, ");
  muLogMacro(<< timer->GetElapsedSeconds() << " seconds\n");

  delete timer;

}
