/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

// Approximation of log using Taylor expansion around 1, better behaved for probability values in [0,1]
// prastawa@sci.utah.edu 7/2013

#ifndef __itkApproximateLogImageFilter_h
#define __itkApproximateLogImageFilter_h

#include "itkUnaryFunctorImageFilter.h"
#include "vnl/vnl_math.h"

namespace itk
{
namespace Functor
{
/**
 * \class ApproximateLog
 * \brief
 * \ingroup ITKImageIntensity
 */
template< class TInput, class TOutput >
class ApproximateLog
{
public:
  ApproximateLog() {}
  ~ApproximateLog() {}
  bool operator!=(const ApproximateLog &) const
  {
    return false;
  }

  bool operator==(const ApproximateLog & other) const
  {
    return !( *this != other );
  }

  inline TOutput operator()(const TInput & A) const
  {
    double A1 = static_cast< double >( A ) - 1.0;
    double A1s = A1*A1;
    double A1c = A1s*A1;
    double A1q = A1s*A1s;
    return static_cast< TOutput >( A1 - A1s / 2 + A1c / 3 - A1q / 4 );
  }
};
}
/** \class ApproximateLogImageFilter
 * \brief Computes the log() of each pixel.
 *
 * \ingroup IntensityImageFilters
 * \ingroup MultiThreaded
 * \ingroup ITKImageIntensity
 */
template< class TInputImage, class TOutputImage >
class ITK_EXPORT ApproximateLogImageFilter:
  public
  UnaryFunctorImageFilter< TInputImage, TOutputImage,
                           Functor::ApproximateLog< typename TInputImage::PixelType,
                                         typename TOutputImage::PixelType >   >
{
public:
  /** Standard class typedefs. */
  typedef ApproximateLogImageFilter Self;
  typedef UnaryFunctorImageFilter<
    TInputImage, TOutputImage,
    Functor::ApproximateLog< typename TInputImage::PixelType,
                  typename TOutputImage::PixelType > > Superclass;

  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ApproximateLogImageFilter,
               UnaryFunctorImageFilter);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro( InputConvertibleToDoubleCheck,
                   ( Concept::Convertible< typename TInputImage::PixelType, double > ) );
  itkConceptMacro( DoubleConvertibleToOutputCheck,
                   ( Concept::Convertible< double, typename TOutputImage::PixelType > ) );
  /** End concept checking */
#endif

protected:
  ApproximateLogImageFilter() {}
  virtual ~ApproximateLogImageFilter() {}

private:
  ApproximateLogImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &); //purposely not implemented
};
} // end namespace itk

#endif
