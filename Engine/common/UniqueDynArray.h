
////////////////////////////////////////////////////////////////////////////////
//
// Dynamically allocated array, with unique members
// Template argument must support == operator
//
////////////////////////////////////////////////////////////////////////////////

// prastawa@cs.unc.edu 5/2007

#ifndef _UniqueDynArray_h
#define _UniqueDynArray_h

template <class T>
class UniqueDynArray
{

public:
  UniqueDynArray();
  UniqueDynArray(const UniqueDynArray& l);
  ~UniqueDynArray();

  // Reference with bounds checking
  T& operator[](unsigned int i) const;

  UniqueDynArray<T>& operator=(const UniqueDynArray<T>& l);

  void Allocate(unsigned int n);
  void Clear();

  // Add new element e, returning position in array
  unsigned int Append(const T& e);

  void Remove(unsigned int i);

  bool Contains(const T& e);
  unsigned int Find(const T& e);

  void Pack();

  inline unsigned int GetSize() const { return m_Size; }

  inline T* GetRawArray() const { return m_Array; }

  // Pointers for iterators and STL sort()
  inline T* GetFirst() const { return m_Array; }
  inline T* GetLast() const { return (m_Array+m_Size); }

  // Cast operator
  //inline operator T*() const { return m_Array; }

private:

  T* m_Array;

  unsigned int m_MaxSize;
  unsigned int m_Size;

};

#ifndef MU_MANUAL_INSTANTIATION
#include "UniqueDynArray.txx"
#endif

#endif
