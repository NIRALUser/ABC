
#include "vtkObjectFactory.h"

#include "vtkTextOutputWindow.h"

#include <iostream>

vtkCxxRevisionMacro(vtkTextOutputWindow, "$Revision: 1.8 $");
vtkStandardNewMacro(vtkTextOutputWindow);

vtkTextOutputWindow::vtkTextOutputWindow()
{

}

vtkTextOutputWindow::~vtkTextOutputWindow()
{

}

void vtkTextOutputWindow::DisplayText(const char* text)
{
  if(!text)
  {
   return;
  }

  std::cout << text << std::endl;
}

