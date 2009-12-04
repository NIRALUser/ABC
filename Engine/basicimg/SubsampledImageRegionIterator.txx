
#ifndef _SubsampledImageRegionIterator_txx
#define _SubsampledImageRegionIterator_txx

#include "SubsampledImageRegionIterator.h"

#include <cmath>

template<typename TImage>
SubsampledImageRegionIterator<TImage>
::SubsampledImageRegionIterator(TImage *ptr, const RegionType& region, double sampleSpacing)
  : itk::ImageRegionConstIteratorWithIndex<TImage>(ptr, region) 
{
  typename TImage::SpacingType spacing = ptr->GetSpacing();

  for (unsigned int dim = 0; dim < TImage::ImageDimension; dim++)
  {
    long skip_dim =  (long)floor(sampleSpacing / spacing[dim]);
    if (skip_dim < 1)
      skip_dim = 1;
    m_Skips[dim] = skip_dim;
  }

  for (unsigned int dim = 0; dim < TImage::ImageDimension; dim++)
  {
    long c = (this->m_EndIndex[dim]-1) / m_Skips[dim];
    this->m_SnappedEndIndex[dim] = c * m_Skips[dim];
  }
}

template<class TImage>
void
SubsampledImageRegionIterator<TImage> 
::GoToBegin()
{
  Superclass::GoToBegin();

  // Can just use super class method
}

template<class TImage>
void
SubsampledImageRegionIterator<TImage> 
::GoToReverseBegin()
{
  // Call super class method
  Superclass::GoToReverseBegin();

  // Make adjustment here
  for(unsigned int in = 0; in < TImage::ImageDimension; in++ )
  {
    this->m_PositionIndex[in] = m_SnappedEndIndex[in];
  }

  const InternalPixelType * buffer = this->m_Image->GetBufferPointer();
  const unsigned long offset =
    this->m_Image->ComputeOffset(this->m_PositionIndex);
  this->m_Position = buffer + offset;
}

template<class TImage>
SubsampledImageRegionIterator<TImage> &
SubsampledImageRegionIterator<TImage>
::operator++()
{   
  this->m_Remaining = false;
  for(unsigned int in = 0; in < TImage::ImageDimension; in++ )
  {
    // Go to next location in the subsampled grid
    long old_in = this->m_PositionIndex[in];
    this->m_PositionIndex[in] = old_in + m_Skips[in];

    if( this->m_PositionIndex[in] < this->m_EndIndex[in] )
    {
      long diff_in = this->m_PositionIndex[in] - old_in;
      this->m_Position += diff_in*this->m_OffsetTable[in];
      this->m_Remaining = true;
      break;
    }
    else
    {
      // Go back to first position for this dimension
      this->m_Position -= this->m_OffsetTable[in] * old_in;
      this->m_PositionIndex[in] = this->m_BeginIndex[in];
    }
  }

  if( !this->m_Remaining ) // It will not advance here otherwise
  {
    this->m_Position = this->m_End;
  }

  return *this;
}

template<class TImage>
SubsampledImageRegionIterator<TImage> &
SubsampledImageRegionIterator<TImage> 
::operator--()
{
  this->m_Remaining = false;
  for( unsigned int in=0; in<TImage::ImageDimension; in++ )
  {
    long old_in = this->m_PositionIndex[in];

    if (this->m_PositionIndex[in] > this->m_BeginIndex[in])
    { 
      this->m_PositionIndex[in] = old_in - m_Skips[in];
      if (this->m_PositionIndex[in] < 0)
        this->m_PositionIndex[in] = 0;
      long diff_in = old_in - this->m_PositionIndex[in];
      this->m_Position -= diff_in*this->m_OffsetTable[in];
      this->m_Remaining = true;
      break;
    }
    else
    {
      // At the beginning
      this->m_PositionIndex[in] = this->m_SnappedEndIndex[in];
      const InternalPixelType * buffer = this->m_Image->GetBufferPointer();
      const unsigned long offset =
        this->m_Image->ComputeOffset(this->m_PositionIndex);
      this->m_Position = buffer + offset;
    }
  }

  if( !this->m_Remaining ) // It will not advance here otherwise
  {
    this->m_Position = this->m_End;
  }

  return *this;
}

template<class TImage>
void
SubsampledImageRegionIterator<TImage> 
::SnapToGrid()
{
  for(unsigned int in = 0; in < TImage::ImageDimension; in++ )
  {
    long c = static_cast<long>(
      floor(this->m_PositionIndex[in] / m_Skips[in] + 0.5));
    this->m_PositionIndex[in] = c * m_Skips[in];
  }

  const InternalPixelType * buffer = this->m_Image->GetBufferPointer();
  const unsigned long offset =
    this->m_Image->ComputeOffset(this->m_PositionIndex);
  this->m_Position = buffer + offset;
}

#endif
