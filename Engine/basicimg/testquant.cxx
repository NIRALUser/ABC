
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

#include "itkOutputWindow.h"
#include "itkTextOutput.h"

#include "KMeansQuantizeImageFilter.h"

#include <iostream>

int
main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << " <image>" << std::endl;
    return -1;
  }

  itk::OutputWindow::SetInstance(itk::TextOutput::New());

  typedef itk::Image<short, 3> ShortImageType;
  typedef itk::Image<unsigned char, 3> ByteImageType;

  typedef itk::ImageFileReader<ShortImageType> ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(argv[1]);
  reader->Update();

  typedef KMeansQuantizeImageFilter<ShortImageType, ByteImageType>
    QuantizerType;
  QuantizerType::Pointer qfilter = QuantizerType::New();
  qfilter->SetInput(reader->GetOutput());
  qfilter->SetNumberOfBins(64);
  qfilter->SetTrimFraction(0.01);
  qfilter->TrimAboveOff();
  qfilter->TrimBelowOff();
  qfilter->SetTrimAboveValue(65);
  qfilter->SetTrimBelowValue(0);
  qfilter->Update();

  typedef itk::ImageFileWriter<ByteImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();

  writer->SetFileName("testquant.mha");
  writer->SetInput(qfilter->GetOutput());
  writer->Update();

  return 0;

}
