
#include "Timer.h"

#include <cmath>

Timer
::Timer()
{
  this->Start();
}

void
Timer
::Start()
{
  m_StartClock = clock();
  m_StartTime = time(NULL);

  m_ElapsedTimeInSecondsOnly = 0;
  m_ElapsedCPUTimeInSecondsOnly = 0;

  m_ElapsedHours = 0;
  m_ElapsedMinutes = 0;
  m_ElapsedSeconds = 0;

  m_Started = true;
}

void
Timer
::Stop()
{

  if (!m_Started)
    return;

  m_ElapsedCPUTimeInSecondsOnly =
    (double)(clock()-m_StartClock) / CLOCKS_PER_SEC;

  double secs = difftime(time(NULL), m_StartTime);

  m_ElapsedTimeInSecondsOnly = secs;

  double hours = floor(secs / 3600.0);
  secs -= hours * 3600.0;

  double mins = floor(secs / 60.0);
  secs -= mins * 60.0;

  m_ElapsedHours = (unsigned int)hours;
  m_ElapsedMinutes = (unsigned int)mins;
  m_ElapsedSeconds = (unsigned int)secs;

  m_Started = false;

}
