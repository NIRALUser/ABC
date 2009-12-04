
#ifndef _UniqueDynArray_txx
#define _UniqueDynArray_txx

#include "UniqueDynArray.h"

#include "muException.h"

template <class T>
T*
_UniqueDynArray_safeAlloc(unsigned int n)
{ 
  T* array = new T[n];
  if (array == NULL)
    muExceptionMacro(
      << "UniqueDynArray: Failed allocating " << n << " elements");
  return array;
}

template <class T>
UniqueDynArray<T>
::UniqueDynArray()
{
  m_Size = 0;
  m_MaxSize = 5;

  //m_Array = new T[m_MaxSize];
  m_Array = _UniqueDynArray_safeAlloc<T>(m_MaxSize);
}

template <class T>
UniqueDynArray<T>
::UniqueDynArray(const UniqueDynArray<T>& l)
{
  m_Size = l.m_Size;
  m_MaxSize = l.m_MaxSize;

  //m_Array = new T[m_MaxSize];
  m_Array = _UniqueDynArray_safeAlloc<T>(m_MaxSize);
  for (unsigned int i = 0; i < m_Size; i++)
    m_Array[i] = l.m_Array[i];
}

template <class T>
UniqueDynArray<T>
::~UniqueDynArray()
{
  delete [] m_Array;
}

template <class T>
T&
UniqueDynArray<T>
::operator[](unsigned int i) const
{
  if (i >= m_Size)
  {
    muExceptionMacro(
      << "UniqueDynArray[i] access out of bounds, index = " << i
      << ", size = " << m_Size);
  }

  return m_Array[i];
}

template <class T>
UniqueDynArray<T>&
UniqueDynArray<T>
::operator=(const UniqueDynArray& l)
{
  delete [] m_Array;

  m_MaxSize = l.m_MaxSize;
  m_Size = l.m_Size;

  //m_Array = new T[m_MaxSize];
  m_Array = _UniqueDynArray_safeAlloc<T>(m_MaxSize);
  for (unsigned int i = 0; i < m_Size; i++)
    m_Array[i] = l.m_Array[i];

  return *this;
}

template <class T>
void
UniqueDynArray<T>
::Allocate(unsigned int size)
{
  if (size == 0 || size <= m_Size)
    return;

  m_MaxSize = size + 1;

  //T* newArray = new T[m_MaxSize];
  T* newArray = _UniqueDynArray_safeAlloc<T>(m_MaxSize);

  for (unsigned int i = 0; i < m_Size; i++)
    newArray[i] = m_Array[i];

  delete [] m_Array;

  m_Array = newArray;
}

template <class T>
void
UniqueDynArray<T>
::Clear()
{

  m_MaxSize = 5;
  m_Size = 0;

  delete [] m_Array;

  //m_Array = new T[m_MaxSize];
  m_Array = _UniqueDynArray_safeAlloc<T>(m_MaxSize);

}

template <class T>
unsigned int
UniqueDynArray<T>
::Append(const T& e)
{
  unsigned int pos = this->Find(e);
  if (pos < m_Size)
    return pos;

  if (m_Size == m_MaxSize)
  {
    m_MaxSize = 2*m_MaxSize + 1;
    //T* newArray = new T[m_MaxSize];
    T* newArray = _UniqueDynArray_safeAlloc<T>(m_MaxSize);
    for (unsigned int i = 0; i < m_Size; i++)
      newArray[i] = m_Array[i];
    delete [] m_Array;
    m_Array = newArray;
  }

  m_Array[m_Size] = e;
  m_Size++;

  return m_Size-1;
}

template <class T>
void
UniqueDynArray<T>
::Remove(unsigned int i)
{
  if (i >= m_Size)
  {
    muExceptionMacro(
      << "UniqueDynArray::Remove access out of bounds, index = " << i
      << ", size = " << m_Size);
  }

  for (unsigned int j = i; j < (m_Size-1); j++)
    m_Array[j] = m_Array[j+1];
  m_Size--;

/* Skip this part for speed
  // Reclaim memory when wasted space gets too large
  if (m_Size < (m_MaxSize/8))
  {
    m_MaxSize /= 8;
    //T* newArray = new T[m_MaxSize];
    T* newArray = _UniqueDynArray_safeAlloc<T>(m_MaxSize);
    for (unsigned int i = 0; i < m_Size; i++)
      newArray[i] = m_Array[i];
    delete [] m_Array;
    m_Array = newArray;
  }
*/
}

template <class T>
bool
UniqueDynArray<T>
::Contains(const T& e)
{
  unsigned int pos = this->Find(e);
  if (pos >= m_Size)
    return false;
  else
    return true;
}

template <class T>
unsigned int
UniqueDynArray<T>
::Find(const T& e)
{
  unsigned int loc = m_Size + 1;
  for (unsigned int i = 0; i < m_Size; i++)
  {
    if (m_Array[i] == e)
    {
      loc = i;
      break;
    }
  }
  return loc;
}

template <class T>
void
UniqueDynArray<T>
::Pack()
{
  this->Allocate(m_Size+1);
}

#endif
