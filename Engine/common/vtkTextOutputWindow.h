
//
// Explicit class for VTK text output
// For bypassing default Windows output
//

#ifndef __vtkTextOutputWindow_h
#define __vtkTextOutputWindow_h

#include "vtkOutputWindow.h"


class vtkTextOutputWindow : public vtkOutputWindow
{
public:
  vtkTypeRevisionMacro(vtkTextOutputWindow, vtkOutputWindow);

  static vtkTextOutputWindow* New();

  virtual void DisplayText(const char*);

 protected:
  vtkTextOutputWindow(); 
  virtual ~vtkTextOutputWindow(); 
private:
  vtkTextOutputWindow(const vtkTextOutputWindow&);  // Not implemented.
  void operator=(const vtkTextOutputWindow&);  // Not implemented.
};

#endif
