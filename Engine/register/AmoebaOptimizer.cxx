
#include "itkCommand.h"
#include "itkEventObject.h"
#include "itkExceptionObject.h"

#include "AmoebaOptimizer.h"

#include <iostream>
#include <cstring>
#include <cmath>

//
// Private helper functions
//

// Compute "radius" of simplex points w.r.t minima
double
AmoebaOptimizer::
SimplexParameterDistance(unsigned int min_idx)
{
  ParametersType simplex_min = m_Simplex[min_idx];

  double rad_p = 0;
  for (unsigned int i = 0; i < m_Simplex.size(); i++)
  {
    // Skip best simplex point
    if (i == min_idx)
      continue;

    // L-inf distance
    ParametersType simplex_i = m_Simplex[i];

    double max_i = 0;
    for (unsigned int j = 0; j < simplex_i.GetSize(); j++)
    {
      double d = fabs(simplex_i[j] - simplex_min[j]);
      if (d > max_i)
        max_i = d;
    }

    if (max_i > rad_p)
      rad_p = max_i;
  }

  return rad_p;
}

// Compute "radius" of simplex values w.r.t minimum value
double
AmoebaOptimizer::
SimplexValueDistance(unsigned int min_idx)
{
  double simplex_minval = m_SimplexValues[min_idx];

  double rad_v = 0;
  for (unsigned int i = 0; i < m_Simplex.size(); i++)
  {
    // Skip best simplex point
    if (i == min_idx)
      continue;

    // L-1 distance
    double d = fabs(m_SimplexValues[i] - simplex_minval);
    if (d > rad_v)
      rad_v = d;
  }

  return rad_v;
}

//
// Method definitions
//

AmoebaOptimizer
::AmoebaOptimizer()
{
  m_MaxIterations = 1000;

  m_FunctionTolerance = 1e-2;
  m_ParameterTolerance = 1e-3;

  m_Verbose = false;

  m_InitialSimplexDeltas = ParametersType(1);
  m_InitialSimplexDeltas[0] = 1.0;

  m_CurrentIteration = 0;

  m_StopCondition = MaximumNumberOfIterations;
}

AmoebaOptimizer
::~AmoebaOptimizer()
{

}

double
AmoebaOptimizer
::TestPoint(unsigned int which, double factor)
{
  unsigned int n = m_CostFunction->GetNumberOfParameters();

  ParametersType centroid(n);
  centroid.Fill(0.0);

  for (unsigned int i = 0; i < m_Simplex.size(); i++)
  {
    ParametersType simplex_i = m_Simplex[i];
    for (unsigned int j = 0; j < n; j++)
      centroid[j] += simplex_i[j];
  }

  for (unsigned int j = 0; j < n; j++)
    centroid[j] /= m_Simplex.size();

  ParametersType simplex_w = m_Simplex[which];

  // Test point = center + f * (test_simplex - center)
  ParametersType testPoint(n);
  for (unsigned int j = 0; j < n; j++)
    testPoint[j] =
      (1.0f - factor) * centroid[j] + factor * simplex_w[j];

  double test_value = m_CostFunction->GetValue(testPoint);

  // If test value is better, update simplex
  if (test_value < m_SimplexValues[which])
  {
    m_SimplexValues[which] = test_value;
    for (unsigned int j = 0; j < n; j++)
      simplex_w[j] = testPoint[j];
    m_Simplex[which] = simplex_w;
  }

  return test_value;
}

void
AmoebaOptimizer
::InitializeSimplex()
{

  ParametersType currentp = this->GetCurrentPosition();

  unsigned int m = m_InitialSimplexDeltas.Size();
  
  m_Simplex.clear();
  m_Simplex.push_back(currentp);
  for (unsigned int i = 0; i < m; i++)
  {
    ParametersType p_i = currentp;
    p_i[i] += m_InitialSimplexDeltas[i];
    m_Simplex.push_back(p_i);
  }

  // Evaluate values at the initial simplex locations
  m_SimplexValues = ParametersType(m_Simplex.size());
  for (unsigned int i = 0; i < m_Simplex.size(); i++)
    m_SimplexValues[i] = m_CostFunction->GetValue(m_Simplex[i]);

}

void
AmoebaOptimizer
::StartOptimization()
{
  m_CurrentIteration = 0;

  this->SetCurrentPosition(this->GetInitialPosition());

  this->InitializeSimplex();

  this->ResumeOptimization();
}

void
AmoebaOptimizer
::ResumeOptimization()
{
  InvokeEvent(itk::StartEvent());

  // Minimum value is minium simplex value
  m_Value = m_SimplexValues[0];
  for (unsigned int i = 1; i < m_Simplex.size(); i++)
  {
    double v = m_SimplexValues[i];
    if (v < m_Value)
      m_Value = v;
  }

  unsigned int n = m_CostFunction->GetNumberOfParameters();

  // Loop until convergence
  while (true)
  {
    m_CurrentIteration++;

    //double oldValue = m_Value;

    // Find the worst, next-worst, and best values
    unsigned int iworst = 0;
    unsigned int inextworst = 1;

    if (m_SimplexValues[iworst] < m_SimplexValues[inextworst])
    {
      unsigned int t = iworst;
      iworst = inextworst;
      inextworst = t;
    }

    unsigned int ibest = 0;

    for (unsigned int i = 0; i < m_Simplex.size(); i++)
    {
      double v = m_SimplexValues[i];
      if (v <= m_SimplexValues[ibest])
        ibest = i;

      if (v > m_SimplexValues[iworst])
      {
        inextworst = iworst;
        iworst = i;
      }
      else if (v > m_SimplexValues[inextworst] && i != iworst)
      {
        inextworst = i;
      }
    }

    if (m_Verbose && (m_CurrentIteration % n) == 1)
    {
      std::cout.setf(std::ios::fixed);
      std::cout.precision(3);
      std::cout << "Iter " << m_CurrentIteration
        << " simplex: best value = " << m_SimplexValues[ibest]
        << " worst value = " << m_SimplexValues[iworst] << std::endl;
      ParametersType simplex_best = m_Simplex[ibest];
      std::cout << "best = ";
      for (unsigned int j = 0; j < n; j++)
        std::cout << simplex_best[j] << " ";
      std::cout << std::endl;
    }

    double rad_p = this->SimplexParameterDistance(ibest);
    double rad_v = this->SimplexValueDistance(ibest);

    // Stop if simplex is too small or simplex values are too flat
    // or after N iterations
    if (m_CurrentIteration > m_MaxIterations
        ||
        rad_p < m_ParameterTolerance
        ||
        rad_v < m_FunctionTolerance)
    {
      if (m_CurrentIteration > m_MaxIterations)
        m_StopCondition = MaximumNumberOfIterations;
      else
        m_StopCondition = Converged;

      this->SetCurrentPosition(m_Simplex[ibest]);
      m_Value = m_SimplexValues[ibest];

      this->StopOptimization();
      break;
    }

    // Try reflection of worst point
    double test_value = this->TestPoint(iworst, -1.0f);

    if (test_value <= m_SimplexValues[ibest])
    {
      // Reflected point is better than current estimate, try going further
      test_value = this->TestPoint(iworst, 2.0f);
    }
    else if (test_value >= m_SimplexValues[inextworst])
    {
      // Reflected point worse than next-worst estimate, do contraction
      double oldWorstValue = m_SimplexValues[iworst];
      test_value = this->TestPoint(iworst, 0.5f);
      if (test_value >= oldWorstValue)
      {
        // Still failed, contract around best estimate
        ParametersType simplex_best = m_Simplex[ibest];
        for (unsigned int i = 0; i < m_Simplex.size(); i++)
        {
          if (i == ibest)
            continue;
          ParametersType simplex_i = m_Simplex[i];
          for (unsigned int j = 0; j < n; j++)
            simplex_i[j] = 0.5f * (simplex_i[j] + simplex_best[j]);
          m_Simplex[i] = simplex_i;
          m_SimplexValues[i] = m_CostFunction->GetValue(simplex_i);
        }
      }
    }

    this->SetCurrentPosition(m_Simplex[ibest]);
    m_Value = m_SimplexValues[ibest];

/*
    if (fabs(m_Value - oldValue) < m_FunctionTolerance)
    {
      m_StopCondition = Converged;
      break;
    }
*/

    this->InvokeEvent(itk::IterationEvent());

  } // main loop

}

void
AmoebaOptimizer
::StopOptimization()
{
  InvokeEvent(itk::EndEvent());
}
