
#include "SimpleGreedyFluidRegistration.h"

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkTextOutput.h"

#include <sstream>
#include <vector>

int main(int argc, char** argv)
{
  // Use text output
  itk::TextOutput::Pointer textout = itk::TextOutput::New();
  itk::OutputWindow::SetInstance(textout);

  srand(39280694);

  typedef itk::Image<float, 3> FloatImageType;
  typedef itk::Image<short, 3> ShortImageType;

  typedef itk::ImageFileReader<FloatImageType> ReaderType;

  int numChannels = (argc - 1) / 2;
std::cout << numChannels << " channels" << std::endl;

  DynArray<FloatImageType::Pointer> fixedImages;
  for (int i = 0; i < numChannels; i++)
  {
    ReaderType::Pointer r = ReaderType::New();
std::cout << "Fixed: " << argv[1+i] << std::endl;
    r->SetFileName(argv[1+i]);
    r->Update();
    fixedImages.Append(r->GetOutput());
  }

  DynArray<FloatImageType::Pointer> movingImages;
  for (int i = 0; i < numChannels; i++)
  {
    ReaderType::Pointer r = ReaderType::New();
std::cout << "Moving: " << argv[1+numChannels+i] << std::endl;
    r->SetFileName(argv[1+numChannels+i]);
    r->Update();
    movingImages.Append(r->GetOutput());
  }

std::cout << "Start fluid" << std::endl;
  typedef SimpleGreedyFluidRegistration<float, 3> FluidWarperType;
  FluidWarperType::Pointer fluid = FluidWarperType::New();
try
{
  fluid->SetFixedImages(fixedImages);
  fluid->SetMovingImages(movingImages);
  fluid->SetIterations(20);
  fluid->SetMaxStep(0.5);
  fluid->Update();
}
catch (...)
{
  std::cerr << "Exception in fluid" << std::endl;
}

  DynArray<FloatImageType::Pointer> outImages = fluid->GetOutputImages();
std::cout << "out: " << outImages.GetSize() << " images" << std::endl;

  typedef itk::ImageFileWriter<ShortImageType> WriterType;

  for (int i = 0; i < outImages.GetSize(); i++)
  {
    std::ostringstream oss;
    oss << "out" << i+1 << ".mha" << std::ends;

    typedef itk::RescaleIntensityImageFilter<FloatImageType, ShortImageType>
      ResType;
    ResType::Pointer res = ResType::New();
    res->SetInput(outImages[i]);
    res->SetOutputMinimum(0);
    res->SetOutputMaximum(4095);
    res->Update();

    WriterType::Pointer w = WriterType::New();
    w->SetFileName(oss.str().c_str());
    w->SetInput(res->GetOutput());
    w->Update();
  }

  return 0;
}
