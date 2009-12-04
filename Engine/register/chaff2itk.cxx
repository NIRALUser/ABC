
#include "itkOutputWindow.h"
#include "itkTextOutput.h"

#include <iostream>
#include <fstream>

#include "itkTransformFileWriter.h"

#include "ChainedAffineTransform3D.h"
#include "PairRegistrationMethod.h"

int
_real_main(int argc, char** argv)
{
  ChainedAffineTransform3D::Pointer trafo = 
    PairRegistrationMethod<float>::ReadAffineTransform(argv[1]);

  ChainedAffineTransform3D::MatrixType M = trafo->GetMatrix();
  ChainedAffineTransform3D::OutputVectorType trans = trafo->GetOffset();

  // Convert trafo to floating point affine
  typedef itk::AffineTransform<float, 3> OutputTransformType;

  OutputTransformType::Pointer outT = OutputTransformType::New();
  OutputTransformType::MatrixType o_M;
  for (unsigned int r = 0; r < 3; r++)
    for (unsigned int c = 0; c < 3; c++)
      o_M[r][c] = (float)M[r][c];
  OutputTransformType::OutputVectorType o_trans;
  for (unsigned int i = 0; i < 3; i++)
    o_trans[i] = (float)trans[i];

  outT->SetMatrix(o_M);
  outT->SetOffset(o_trans);

  itk::TransformFileWriter::Pointer writer = itk::TransformFileWriter::New();
  writer->SetInput(outT);
  writer->SetFileName(argv[2]);
  writer->Update();

  return 0;
}

int
main(int argc, char **argv)
{
  if (argc != 3)
  {
    std::cerr << "Usage: " << argv[0] << " <chained_affine> <itkoutput>" << std::endl;
    return -1;
  }

  itk::OutputWindow::SetInstance(itk::TextOutput::New());

  try
  {
    return _real_main(argc, argv);
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

  return 0;
}

