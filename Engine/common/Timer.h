
////////////////////////////////////////////////////////////////////////////////
//
// Handles simple timings
//
////////////////////////////////////////////////////////////////////////////////

// prastawa@cs.unc.edu 10/2003

#ifndef _Timer_h
#define _Timer_h

#include <ctime>

class Timer
{

public:

  Timer();
  ~Timer() {}

  void Start();
  void Stop();

  inline double GetElapsedTimeInSecondsOnly()
  { this->Stop(); return m_ElapsedTimeInSecondsOnly; }

  inline double GetElapsedCPUTimeInSecondsOnly()
  { this->Stop(); return m_ElapsedCPUTimeInSecondsOnly; }

  inline unsigned int GetElapsedHours()
  { this->Stop(); return m_ElapsedHours; }
  inline unsigned int GetElapsedMinutes()
  { this->Stop(); return m_ElapsedMinutes; }
  inline unsigned int GetElapsedSeconds()
  { this->Stop(); return m_ElapsedSeconds; }

private:

  clock_t m_StartClock;
  time_t m_StartTime;

  unsigned int m_ElapsedHours;
  unsigned int m_ElapsedMinutes;
  unsigned int m_ElapsedSeconds;

  double m_ElapsedTimeInSecondsOnly;
  double m_ElapsedCPUTimeInSecondsOnly;

  bool m_Started;

};

#endif
