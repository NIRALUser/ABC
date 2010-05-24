
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "itkResampleImageFilter.h"
#include "itkSmoothingRecursiveGaussianImageFilter.h"

#include "ChainedAffineTransform3D.h"

#include <cstdlib>
#include <iostream>

typedef itk::Image<float, 3> ImageType;

inline double square(double x)
{ return x*x; }

inline double urand()
{ return 2.0 * rand() / (double)RAND_MAX - 1.0; }

void writeImage(const char* fn, const ImageType* img)
{
  typedef itk::ImageFileWriter<ImageType> WriterType;
  WriterType::Pointer w = WriterType::New();
  w->SetInput(img);
  w->SetFileName(fn);
  w->UseCompressionOn();
  w->Update();
}

ImageType::Pointer blurImage(const ImageType* img)
{
  typedef itk::SmoothingRecursiveGaussianImageFilter<ImageType, ImageType>
    BlurType;
  BlurType::Pointer b = BlurType::New();
  b->SetInput(img);
  b->SetSigma(4.0);
  b->Update();
  return b->GetOutput();
}

void biasImage(ImageType* img, uint maxdegree)
{
  uint numcoeffs = (maxdegree+1)*(maxdegree+2)*(maxdegree+3)/6;

  std::vector<double> coeffs(numcoeffs, 0);
  for (uint i = 0; i < numcoeffs; i++)
    coeffs[i] = urand() * 0.1;

  double xc, yc, zc;
  double xmid = 64;
  double ymid = 64;
  double zmid = 64;

  itk::ImageRegionIteratorWithIndex<ImageType> it(img, img->GetLargestPossibleRegion());
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    ImageType::IndexType index = it.GetIndex();
    double logbias = 0;

    unsigned int c = 0;

    for (int order = 0; order <= maxdegree; order++) {
      for (int xorder = 0; xorder <= order; xorder++) {
        for (int yorder = 0; yorder <= (order-xorder); yorder++) {

          int zorder = order - xorder - yorder;

          xc = (index[0] - xmid) / xmid;
          yc = (index[1] - ymid) / ymid;
          zc = (index[2] - zmid) / zmid;

          double poly =
            (double)(pow(xc,xorder) * pow(yc,yorder) * pow(zc,zorder));

          logbias += coeffs[c] * poly;
        }
      }
    }

    double bias = exp(logbias);

    img->SetPixel(index, img->GetPixel(index)*bias);

  }

}

int main()
{

  ImageType::SizeType size;
  size.Fill(128);

  ImageType::RegionType region;
  region.SetSize(size);

  // Create test bull's eye data
  ImageType::Pointer img1 = ImageType::New();
  ImageType::Pointer img2 = ImageType::New();
  ImageType::Pointer p1 = ImageType::New();
  ImageType::Pointer p2 = ImageType::New();
  ImageType::Pointer p3 = ImageType::New();

  img1->SetRegions(region); img1->Allocate();
  img2->SetRegions(region); img2->Allocate();
  p1->SetRegions(region); p1->Allocate();
  p2->SetRegions(region); p2->Allocate();
  p3->SetRegions(region); p3->Allocate();

  img1->FillBuffer(0);
  img2->FillBuffer(0);
  p1->FillBuffer(0);
  p2->FillBuffer(0);
  p3->FillBuffer(255.0);

  itk::ImageRegionIteratorWithIndex<ImageType> it(img1, region);
  for (it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    ImageType::IndexType ind = it.GetIndex();

    double e1 = 0;
    e1 += square((ind[0]-64.0) / 42);
    e1 += square((ind[1]-64.0) / 50);
    e1 += square((ind[2]-64.0) / 42);

    double e2 = 0;
    e2 += square((ind[0]-64.0) / 20);
    e2 += square((ind[1]-64.0) / 28);
    e2 += square((ind[2]-64.0) / 20);

    if (e1 <= 1.0 && e2 > 1.0)
    {
      img1->SetPixel(ind, 4000);
      img2->SetPixel(ind, 2200);
      p1->SetPixel(ind, 255.0);
      p3->SetPixel(ind, 0.0);
    }
    if (e2 <= 1.0)
    {
      img1->SetPixel(ind, 2000);
      img2->SetPixel(ind, 5000);
      p2->SetPixel(ind, 255.0);
      p3->SetPixel(ind, 0.0);
    }

    img1->SetPixel(ind, img1->GetPixel(ind)+urand()*500);
    img2->SetPixel(ind, img2->GetPixel(ind)+urand()*500);
  }

  biasImage(img1, 2);
  biasImage(img2, 2);

  writeImage("testimage_1.mha", img1);
  //writeImage("testimage_2.mha", img2);
  writeImage("testtruth_p1.mha", p1);
  writeImage("testtruth_p2.mha", p2);
  writeImage("testtruth_p3.mha", p3);

  // Transform images for registration test
  // also generate atlas by blurring

  ChainedAffineTransform3D::Pointer A = ChainedAffineTransform3D::New();
  A->SetIdentity();

  ChainedAffineTransform3D::ParametersType affP = A->GetParameters();
  affP[0] = 2.0;
  affP[1] = -3.0;
  affP[2] = 0.0;
  affP[3] = 0.01;
  affP[4] = -0.01;
  affP[5] = 0.05;
  affP[6] = 1.2;
  A->SetParameters(affP);

  typedef itk::ResampleImageFilter<ImageType, ImageType> ResamplerType;
  ResamplerType::Pointer res = ResamplerType::New();
  res->SetSize(size);

  res->SetInput(img1);
  res->SetTransform(A);
  res->Update();
  img1 = res->GetOutput();
  img1 = blurImage(img1);
  writeImage("template.mha", img1);

  res->SetInput(p1);
  res->SetTransform(A);
  res->Update();
  p1 = res->GetOutput();
  p1 = blurImage(p1);
  writeImage("1.mha", p1);

  res->SetInput(p2);
  res->SetTransform(A);
  res->Update();
  p2 = res->GetOutput();
  p2 = blurImage(p2);
  writeImage("2.mha", p2);

  res->SetInput(p3);
  res->SetTransform(A);
  res->SetDefaultPixelValue(255.0);
  res->Update();
  p3 = res->GetOutput();
  p3 = blurImage(p3);
  writeImage("3.mha", p3);

  affP[0] = -1.0;
  affP[1] = 4.0;
  affP[2] = -1.75;
  affP[3] = 0.003;
  affP[4] = 0.05;
  affP[5] = -0.01;
  affP[6] = 0.9;
  affP[7] = 1.1;
  A->SetParameters(affP);

  res->SetInput(img2);
  res->SetTransform(A);
  res->SetDefaultPixelValue(0.0);
  res->Update();
  img2 = res->GetOutput();
  writeImage("testimage_2.mha", img2);

  return 0;
}
