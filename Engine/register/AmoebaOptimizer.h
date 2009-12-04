
//
// Nelder-Mead downhill simplex minimization
// Adapted from Numerical Recipes
//

#ifndef _AmoebaOptimizer_h
#define _AmoebaOptimizer_h

#include "itkSingleValuedNonLinearOptimizer.h"

#include <vector>

class AmoebaOptimizer:
  public itk::SingleValuedNonLinearOptimizer
{
public:

  /** Standard class typedefs. */
  typedef AmoebaOptimizer Self;
  typedef itk::SingleValuedNonLinearOptimizer    Superclass;
  typedef itk::SmartPointer<Self>                Pointer;
  typedef itk::SmartPointer<const Self>          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(AmoebaOptimizer, itk::SingleValuedNonLinearOptimizer );

  typedef enum {
    Converged,
    MaximumNumberOfIterations,
    MetricError
  } StopConditionType;

  void StartOptimization();

  void ResumeOptimization();

  void StopOptimization();

  itkSetMacro(MaxIterations, unsigned int);
  itkGetConstMacro(MaxIterations, unsigned int);

  itkSetMacro(FunctionTolerance, double);
  itkGetConstMacro(FunctionTolerance, double);

  itkSetMacro(ParameterTolerance, double);
  itkGetConstMacro(ParameterTolerance, double);

  itkGetConstMacro(Value, double);

  itkGetConstMacro(CurrentIteration, unsigned int);

  // Print simplex info every few iterations?
  itkSetMacro(Verbose, bool);
  itkGetConstMacro(Verbose, bool);
  itkBooleanMacro(Verbose);

  itkSetMacro(InitialSimplexDeltas, ParametersType);

protected:

  AmoebaOptimizer();
  virtual ~AmoebaOptimizer();

  // Test a point which is along a line formed by a point in the simplex
  // and the simplex centroid
  double TestPoint(unsigned int which, double factor);

  double SimplexParameterDistance(unsigned int min_idx);
  double SimplexValueDistance(unsigned int min_idx);

  void InitializeSimplex();

private:

  ParametersType m_InitialSimplexDeltas;

  std::vector<ParametersType> m_Simplex;
  ParametersType m_SimplexValues;

  // Convergence parameters
  unsigned int m_MaxIterations;
  double m_FunctionTolerance;
  double m_ParameterTolerance;

  unsigned int m_CurrentIteration;

  bool m_Verbose;

  double m_Value;

  StopConditionType m_StopCondition;

};

#endif
