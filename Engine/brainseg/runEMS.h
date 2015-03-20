
// Wrapper function that puts everything together (filtering, registration,
// segmentation)

// prastawa@cs.unc.edu 11/2003

#ifndef _runEMS_h
#define _runEMS_h

#define ABC_VERSION "1.5.1"

#include "EMSParameters.h"

void runEMS(EMSParameters* params, bool debugflag, bool writemoreflag);

#endif
