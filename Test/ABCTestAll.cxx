
#include "mu.h"

#include "itkOutputWindow.h"
#include "itkTextOutput.h"

#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "EMSParameters.h"
#include "runEMS.h"

#include <exception>
#include <iostream>


void
printUsage(char* progname)
{
  std::cerr << "Usage: " << progname << " <atlasdir> <datadir> <outdir>" << std::endl;
  std::cerr << std::endl;
}

int
main(int argc, char** argv)
{

  if (argc != 4)
  {
    printUsage(argv[0]);
    return -1;
  }

  itk::OutputWindow::SetInstance(itk::TextOutput::New());

  std::string atlasdir = argv[1];
  std::string datadir = argv[2];
  std::string outdir = argv[3];

std::cout << "Testing ABC with atlas at " << atlasdir << " and data at " << datadir << " writing results to " << outdir << std::endl;

  try
  {
    EMSParameters::Pointer emsp = EMSParameters::New();
    emsp->SetSuffix("seg");
    emsp->SetAtlasDirectory(atlasdir);
    emsp->SetOutputDirectory(outdir);
    emsp->SetDoAtlasWarp(true);
    emsp->SetAtlasWarpFluidIterations(20);
    emsp->SetFilterIterations(50);
    emsp->SetMaxBiasDegree(2);

    emsp->AddImage(datadir+std::string("/testimage_1.mha"), std::string("RAI"));
    emsp->AddImage(datadir+std::string("/testimage_2.mha"), std::string("RAI"));

    emsp->AppendPriorWeight(1.0);
    emsp->AppendPriorWeight(1.0);
    emsp->AppendPriorWeight(1.0);

    runEMS(emsp, false, true);
  }
  catch (itk::ExceptionObject& e)
  {
    std::cerr << e << std::endl;
    return -1;
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception: " << e.what() << std::endl;
    return -1;
  }
  catch (std::string& s)
  {
    std::cerr << "Exception: " << s << std::endl;
    return -1;
  }
  catch (...)
  {
    std::cerr << "Unknown exception" << std::endl;
    return -1;
  }

  // Read output probabilities, compare against truth in datadir
  for (uint c = 0; c < 2; c++)
  {
    typedef itk::Image<float, 3> ImageType;
    typedef itk::ImageFileReader<ImageType> ReaderType;

    std::ostringstream oss;

    ReaderType::Pointer r1 = ReaderType::New();
    oss << outdir << "/testimage_1_posterior" << c << "_seg.mha" << std::ends;
    r1->SetFileName(oss.str().c_str());
    r1->Update();
    ImageType::Pointer seg = r1->GetOutput();

    ReaderType::Pointer r2 = ReaderType::New();
    oss.str("");
    oss << datadir << "/testtruth_p" << c+1 << ".mha" << std::ends;
    r2->SetFileName(oss.str().c_str());
    r2->Update();
    ImageType::Pointer truth = r2->GetOutput();

    itk::ImageRegionIteratorWithIndex<ImageType> it(seg, seg->GetLargestPossibleRegion());

    double sumDiff = 0;
    double thresDiff = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
      ImageType::IndexType ind = it.GetIndex();
      double d = (seg->GetPixel(ind)/32767.0) - (truth->GetPixel(ind)/255.0);
      sumDiff += d*d;
      thresDiff += 1.0;
    }
std::cout << "Max diff^2 = " << thresDiff << std::endl;
    thresDiff *= 0.05;

    std::cout << "Sum diff^2 for class " << c << " = " << sumDiff << std::endl;
    if (sumDiff > thresDiff)
      return -1;
  }

  return 0;

}
