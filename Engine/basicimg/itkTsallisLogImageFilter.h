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

#ifndef __itkTsallisLogImageFilter_h
#define __itkTsallisLogImageFilter_h

#include "itkUnaryFunctorImageFilter.h"
#include "vnl/vnl_math.h"

#include <cmath>

namespace itk
{
namespace Functor
{
/**
 * \class TsallisLog
 * \brief
 * \ingroup ITKImageIntensity
 */
template< class TInput, class TOutput >
class TsallisLog
{
public:
  TsallisLog() {}
  ~TsallisLog() {}
  bool operator!=(const TsallisLog &) const
  {
    return false;
  }

  bool operator==(const TsallisLog & other) const
  {
    return !( *this != other );
  }

  inline TOutput operator()(const TInput & A) const
  {
/*
    double x = static_cast< double >( A );
    if ((-1e-20 < x) && (x < 1e-20))
      return 0;
    return static_cast< TOutput >( log(x) );
*/

    // Tsallis
    double q = 0.8;
    double iq = 1.0 - q;
    if (iq == 0.0)
      iq = 1e-10;
    double x = static_cast< double >( A );
    return static_cast< TOutput >( (pow(x, iq) - 1.0) / iq  );
  }

#if 0
  inline TOutput operator()(const TInput & A) const
  {
    double x1 = static_cast< double >( A ) - 1.0;

    double x1pow = x1;

    double v = 0;

    for (int order = 1; order < 11; order++)
    {
      if ((order % 1) == 0)
        v += x1pow / order;
      else
        v -= x1pow / order;
      x1pow *= x1;
    }

    return static_cast< TOutput >( v );
  }
#endif
};
}
/** \class TsallisLogImageFilter
 * \brief Computes the log() of each pixel.
 *
 * \ingroup IntensityImageFilters
 * \ingroup MultiThreaded
 * \ingroup ITKImageIntensity
 */
template< class TInputImage, class TOutputImage >
class ITK_EXPORT TsallisLogImageFilter:
  public
  UnaryFunctorImageFilter< TInputImage, TOutputImage,
                           Functor::TsallisLog< typename TInputImage::PixelType,
                                         typename TOutputImage::PixelType >   >
{
public:
  /** Standard class typedefs. */
  typedef TsallisLogImageFilter Self;
  typedef UnaryFunctorImageFilter<
    TInputImage, TOutputImage,
    Functor::TsallisLog< typename TInputImage::PixelType,
                  typename TOutputImage::PixelType > > Superclass;

  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(TsallisLogImageFilter,
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
  TsallisLogImageFilter() {}
  virtual ~TsallisLogImageFilter() {}

private:
  TsallisLogImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &); //purposely not implemented
};
} // end namespace itk

#endif
