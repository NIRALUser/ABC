
//

#ifndef _SimulatedAnnealingOptimizer_h
#define _SimulatedAnnealingOptimizer_h

#include "itkSingleValuedNonLinearOptimizer.h"

#include <vector>

class SimulatedAnnealingOptimizer:
  public itk::SingleValuedNonLinearOptimizer
{
public:

  /** Standard class typedefs. */
  typedef SimulatedAnnealingOptimizer Self;
  typedef itk::SingleValuedNonLinearOptimizer    Superclass;
  typedef itk::SmartPointer<Self>                Pointer;
  typedef itk::SmartPointer<const Self>          ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(SimulatedAnnealingOptimizer, itk::SingleValuedNonLinearOptimizer );

  typedef enum {
    Converged,
    MaximumNumberOfIterations,
    MetricError
  } StopConditionType;

  void StartOptimization();

  void ResumeOptimization();

  void StopOptimization();

  itkSetMacro(BurnInIterations, unsigned int);
  itkGetConstMacro(BurnInIterations, unsigned int);

  itkSetMacro(MaxIterations, unsigned int);
  itkGetConstMacro(MaxIterations, unsigned int);

  itkGetConstMacro(Value, double);

  itkGetConstMacro(CurrentIteration, unsigned int);

  // Print simplex info every few iterations?
  itkSetMacro(Verbose, bool);
  itkGetConstMacro(Verbose, bool);
  itkBooleanMacro(Verbose);

  void SetTemperatureRange(double tmin, double tmax)
  { m_MinTemperature = tmin; m_MaxTemperature = tmax; }

  itkSetMacro(RandomWalkSteps, ParametersType);

protected:

  SimulatedAnnealingOptimizer();
  virtual ~SimulatedAnnealingOptimizer();

private:

  ParametersType m_RandomWalkSteps;

  unsigned int m_BurnInIterations;
  unsigned int m_MaxIterations;

  unsigned int m_CurrentIteration;

  bool m_Verbose;

  bool m_Temperature;
  bool m_MinTemperature;
  bool m_MaxTemperature;

  double m_Value;

  StopConditionType m_StopCondition;

};

#endif
