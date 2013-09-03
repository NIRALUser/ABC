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

// prastawa@sci.utah.edu 7/2013

#ifndef __itkTsallisLogDerivativeImageFilter_h
#define __itkTsallisLogDerivativeImageFilter_h

#include "itkUnaryFunctorImageFilter.h"
#include "vnl/vnl_math.h"

#include <cmath>

namespace itk
{
namespace Functor
{
/**
 * \class TsallisLogDerivative
 * \brief
 * \ingroup ITKImageIntensity
 */
template< class TInput, class TOutput >
class TsallisLogDerivative
{
public:
  TsallisLogDerivative() {}
  ~TsallisLogDerivative() {}
  bool operator!=(const TsallisLogDerivative &) const
  {
    return false;
  }

  bool operator==(const TsallisLogDerivative & other) const
  {
    return !( *this != other );
  }

  inline TOutput operator()(const TInput & A) const
  {
/*
    double x = static_cast< double >( A );
    if ((-1e-20 < x) && (x < 1e-20))
      return 0;
    return static_cast< TOutput >( 1.0 / x );
*/
    double q = 0.8;
    double x = static_cast< double >( A );

    return static_cast< TOutput >( pow(x + 0.001, -q) );

/*
    double x1 = x - 1.0;
    //return static_cast< TOutput >( 1.0 - q*x1 + q*(q+1)*x1*x1 * (0.5 - (q+2)*x1/6.0) );
    return static_cast< TOutput >( 1.0 - q*x1 );
*/

/*
    double v = 1.0;

    double x1pow = x1;
    double qpow = q;
    double nfac = 1;
    for (int order = 1; order <= 5; order++)
    {
      if (order % 2 == 1)
        v -= x1pow * qpow / nfac;
      else
        v += x1pow * qpow / nfac;
      qpow *= (q + order);
      x1pow *= x1;
      nfac *= (order + 1);
    }
    return static_cast< TOutput >( v );
*/

  }
};
}
/** \class TsallisLogDerivativeImageFilter
 * \brief Computes the log() of each pixel.
 *
 * \ingroup IntensityImageFilters
 * \ingroup MultiThreaded
 * \ingroup ITKImageIntensity
 */
template< class TInputImage, class TOutputImage >
class ITK_EXPORT TsallisLogDerivativeImageFilter:
  public
  UnaryFunctorImageFilter< TInputImage, TOutputImage,
                           Functor::TsallisLogDerivative< typename TInputImage::PixelType,
                                         typename TOutputImage::PixelType >   >
{
public:
  /** Standard class typedefs. */
  typedef TsallisLogDerivativeImageFilter Self;
  typedef UnaryFunctorImageFilter<
    TInputImage, TOutputImage,
    Functor::TsallisLogDerivative< typename TInputImage::PixelType,
                  typename TOutputImage::PixelType > > Superclass;

  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(TsallisLogDerivativeImageFilter,
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
  TsallisLogDerivativeImageFilter() {}
  virtual ~TsallisLogDerivativeImageFilter() {}

private:
  TsallisLogDerivativeImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &); //purposely not implemented
};
} // end namespace itk

#endif
