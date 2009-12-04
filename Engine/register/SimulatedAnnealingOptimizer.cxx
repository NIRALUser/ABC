
#include "itkCommand.h"
#include "itkEventObject.h"
#include "itkExceptionObject.h"

#include "MersenneTwisterRNG.h"

#include "SimulatedAnnealingOptimizer.h"

#include <iostream>
#include <cstring>
#include <cmath>


SimulatedAnnealingOptimizer
::SimulatedAnnealingOptimizer()
{
  m_BurnInIterations = 100;
  m_MaxIterations = 1000;

  m_Verbose = false;

  m_RandomWalkSteps = ParametersType(1);
  m_RandomWalkSteps[0] = 1.0;

  m_CurrentIteration = 0;

  m_Temperature = 10.0;
  m_MinTemperature = 0.1;
  m_MaxTemperature = 10.0;

  m_StopCondition = MaximumNumberOfIterations;
}

SimulatedAnnealingOptimizer
::~SimulatedAnnealingOptimizer()
{

}

void
SimulatedAnnealingOptimizer
::StartOptimization()
{
  m_CurrentIteration = 0;

  this->SetCurrentPosition(this->GetInitialPosition());

  this->ResumeOptimization();
}

void
SimulatedAnnealingOptimizer
::ResumeOptimization()
{
  InvokeEvent(itk::StartEvent());

  unsigned int n = m_CostFunction->GetNumberOfParameters();

  unsigned int adjustIters = m_MaxIterations / 10;

  double stepT = (m_MaxTemperature-m_MinTemperature) / 10.0;

  double T = m_MaxTemperature;
  double K = 1.0 / T;

  ParametersType p_curr = this->GetCurrentPosition();
  ParametersType p_min = p_curr;

  ParametersType stepVars(n);
  for (unsigned int i = 0; i < n; i++)
    stepVars[i] = m_RandomWalkSteps[i]*m_RandomWalkSteps[i];

  m_Value = m_CostFunction->GetValue(p_curr);

  double v_curr = m_Value;

  MersenneTwisterRNG* rng = MersenneTwisterRNG::GetGlobalInstance();

  m_CurrentIteration = 0;

  for (unsigned int iter = 0; iter < m_MaxIterations; iter++)
  {
    m_CurrentIteration++;

    ParametersType p_next = p_curr;

    // Random walk
    for (unsigned int i = 0; i < n; i++)
      p_next[i] += 
        (2.0*rng->GenerateUniformRealClosedInterval()-1.0) * stepVars[i];
      //p_next[i] += rng->GenerateNormal(0.0, stepVars[i]);

    double v_next = m_CostFunction->GetValue(p_next);

    double acc = exp(K * (v_next - v_curr));
    if (acc > 1.0)
      acc = 1.0;

    double q = rng->GenerateUniformRealOpenInterval();

    if (q <= acc)
    {
      p_curr = p_next;
      v_curr = v_next;
    }

    //if (iter < m_BurnInIterations)
    //  continue;

    if (v_curr < m_Value)
    {
      m_Value = v_curr;
      p_min = p_curr;
      this->SetCurrentPosition(p_min);
    }

    if ((iter % adjustIters) == 0)
    {
      T -= stepT;
      K = 1.0 / T;
    }

    this->InvokeEvent(itk::IterationEvent());

  }

}

void
SimulatedAnnealingOptimizer
::StopOptimization()
{
  InvokeEvent(itk::EndEvent());
}
