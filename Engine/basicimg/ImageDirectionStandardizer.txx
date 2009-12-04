
#ifndef _ImageDirectionStandardizer_txx
#define _ImageDirectionStandardizer_txx

#include "ImageDirectionStandardizer.h"

#include "itkExceptionObject.h"
#include "itkImageDuplicator.h"
#include "itkOrientImageFilter.h"
#include "itkResampleImageFilter.h"

#include "itksys/SystemTools.hxx"

#include <ctype.h>

template <class TImage>
ImageDirectionStandardizer<TImage>
::ImageDirectionStandardizer()
{
  m_TargetOrientationCode = "RAI";
  m_TargetImageOrientation.SetIdentity();
}

template <class TImage>
ImageDirectionStandardizer<TImage>
::~ImageDirectionStandardizer()
{

}

template <class TImage>
typename ImageDirectionStandardizer<TImage>::CoordinateOrientationCode
ImageDirectionStandardizer<TImage>
::_ParseOrientationString(std::string& s)
{
  if (s.length() != 3)
    itkExceptionMacro(<< "Invalid direction code: " << s);

  if (itksys::SystemTools::Strucmp(s.c_str(), "RAI") != 0)
    return itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAI;
  if (itksys::SystemTools::Strucmp(s.c_str(), "RAS") != 0)
    return itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RAS;
  if (itksys::SystemTools::Strucmp(s.c_str(), "RPI") != 0)
    return itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPI;
  if (itksys::SystemTools::Strucmp(s.c_str(), "RPS") != 0)
    return itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_RPS;

  if (itksys::SystemTools::Strucmp(s.c_str(), "LAI") != 0)
    return itk::SpatialOrientation::ITK_COORDINATE_ORIENTATION_LAI;

  itkExceptionMacro(<< "Invalid direction code: " << s);
}

template <class TImage>
typename ImageDirectionStandardizer<TImage>::ImageDirectionType
ImageDirectionStandardizer<TImage>
::_GetDirectionFromString(std::string& s)
{
  if (s.length() != 3)
    itkExceptionMacro(<< "Invalid direction code: " << s);

  // Case insensitive
  for (unsigned int i = 0; i < 3; i++)
    s[i] = toupper(s[i]);

  // Verify that there are no duplicates in orientation encoding
  for (unsigned int i = 0; i < 3; i++)
    for (unsigned int j = i+1; j < 3; j++)
      if (s[i] == s[j])
        itkExceptionMacro(<< "Invalid direction code: " << s); 

  ImageDirectionType dir;
  dir.SetIdentity();

  for (unsigned int i = 0; i < 3; i++)
  {
    char ch = s[i];

    switch (ch)
    {
      case 'R':
        dir[0][i] = 1.0;
        dir[1][i] = 0.0;
        dir[2][i] = 0.0;
        break;
      case 'L':
        dir[0][i] = -1.0;
        dir[1][i] = 0.0;
        dir[2][i] = 0.0;
        break;

      case 'A':
        dir[0][i] = 0.0;
        dir[1][i] = 1.0;
        dir[2][i] = 0.0;
        break;
      case 'P':
        dir[0][i] = 0.0;
        dir[1][i] = -1.0;
        dir[2][i] = 0.0;
        break;

      case 'I':
        dir[0][i] = 0.0;
        dir[1][i] = 0.0;
        dir[2][i] = 1.0;
        break;
      case 'S':
        dir[0][i] = 0.0;
        dir[1][i] = 0.0;
        dir[2][i] = -1.0;
        break;

      default:
       itkExceptionMacro(<< "Invalid orientation: " << s);
    }
  }

  return dir;
}

template <class TImage>
void
ImageDirectionStandardizer<TImage>
::SetTargetDirectionFromString(ImageType* img, std::string& s)
{
  m_TargetOrientationCode = s;
  m_TargetImageOrientation = img->GetDirection();
}

template <class TImage>
typename ImageDirectionStandardizer<TImage>::ImagePointer
ImageDirectionStandardizer<TImage>
::Standardize(ImageType* img, std::string& dirstring)
{
  if (img->GetImageDimension() != 3)
    itkExceptionMacro(<< "Only supports 3D image");

  typedef itk::OrientImageFilter<ImageType, ImageType> OrienterType;

  typename OrienterType::Pointer orient = OrienterType::New();

  if (itksys::SystemTools::Strucmp(dirstring.c_str(), "file") == 0)
  {
    orient->UseImageDirectionOn();
  }
  else
  {
    orient->UseImageDirectionOff();
    ImageDirectionType cosm = this->_GetDirectionFromString(dirstring);
    orient->SetGivenCoordinateDirection(cosm);
  }

  if (itksys::SystemTools::Strucmp(m_TargetOrientationCode.c_str(), "file")
      == 0)
  {
    orient->SetDesiredCoordinateDirection(m_TargetImageOrientation);
  }
  else
  {
    orient->SetDesiredCoordinateDirection(
      this->_GetDirectionFromString(m_TargetOrientationCode));
  }

  orient->SetInput(img);
  orient->Update();

  return orient->GetOutput();

}

#endif
