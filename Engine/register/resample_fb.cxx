
#include "itkAffineTransform.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"

#include "itkOutputWindow.h"
#include "itkTextOutput.h"

#include <iostream>
#include <fstream>

#include "PairRegistrationMethod.h"

int
main(int argc, char **argv)
{

  if (argc != 6)
  {
    std::cerr << "Usage: " << argv[0];
    std::cerr << " <affine1> <affine2> <target> <input> <output>" << std::endl;
    return 1;
  }

  itk::OutputWindow::SetInstance(itk::TextOutput::New());

  std::cout << "Transform 1: " << argv[1] << std::endl;
  std::cout << "Transform 2: " << argv[2] << std::endl;
  std::cout << "Target image: " << argv[3] << std::endl;
  std::cout << "Input image: " << argv[4] << std::endl;
  std::cout << "Output image: " << argv[5] << std::endl;

  const unsigned int Dimension = 3;
  typedef short PixelType;

  typedef itk::Image<PixelType, Dimension>  ImageType;

  typedef PairRegistrationMethod<PixelType>::AffineTransformType TransformType;

  typedef itk::ImageFileReader<ImageType> ImageReaderType;

  // Read the affine transform
  TransformType::Pointer transform1 =
    PairRegistrationMethod<PixelType>::ReadAffineTransform(argv[1]);
  TransformType::Pointer transform2 =
    PairRegistrationMethod<PixelType>::ReadAffineTransform(argv[2]);

  // Read the images
  ImageReaderType::Pointer  reader1 = ImageReaderType::New();
  ImageReaderType::Pointer  reader2 = ImageReaderType::New();

  ImageType::Pointer targetImg;
  ImageType::Pointer inputImg;

  std::cout << "Reading images..." << std::endl;
  try
  {
    reader1->SetFileName(argv[3]);
    reader1->Update();

    targetImg = reader1->GetOutput();

    reader2->SetFileName(argv[4]);
    reader2->Update();

    inputImg = reader2->GetOutput();
  }
  catch (itk::ExceptionObject& exc)
  {
    std::cerr << "Exception caught!" << std::endl;
    std::cerr << exc << std::endl;
  }

  //
  // Compute chained affine inv(H2)*H1
  //

  // Inverse transform
//TODO: no inverse just chain them directly
  TransformType::Pointer invTrafo2 = transform2->GetInverse();

  // Get raw matrices
  TransformType::MatrixType m1 = transform1->GetMatrix();
  TransformType::MatrixType m2 = invTrafo2->GetMatrix();
  TransformType::OffsetType o1 = transform1->GetOffset();
  TransformType::OffsetType o2 = invTrafo2->GetOffset();
  
  typedef itk::AffineTransform<double, 3> ITKAffineType;

  ITKAffineType::Pointer finalT = ITKAffineType::New();
  finalT->SetMatrix(m2*m1);
  finalT->SetOffset(m2*o1 + o2);

  // Resample images
  std::cout << "Resampling..." << std::endl;

  typedef itk::ResampleImageFilter<ImageType, ImageType>
    ResampleFilterType;
  ResampleFilterType::Pointer resampler = ResampleFilterType::New();

  resampler->SetTransform(finalT);
  resampler->SetInput(inputImg);

  typedef itk::BSplineInterpolateImageFunction<ImageType, double>
    InterpolatorType;

  InterpolatorType::Pointer interp = InterpolatorType::New();
  interp->SetSplineOrder(3);

  resampler->SetInterpolator(interp);
  resampler->SetOutputParametersFromImage(targetImg);
  resampler->SetDefaultPixelValue(0);

  typedef itk::ImageFileWriter<ImageType>  WriterType;
  WriterType::Pointer writer = WriterType::New();

  writer->SetFileName(argv[5]);
  writer->SetInput(resampler->GetOutput() );
  writer->UseCompressionOn();
  writer->Update();

  return 0;

}

