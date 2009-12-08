
#include <FL/Fl.h>
#include <FL/Fl_Window.h>

#include <string>

#include "itkOutputWindow.h"
#include "itkTextOutput.h"

#include "GUI.h"

int main(int argc, char** argv)
{

  itk::OutputWindow::SetInstance(itk::TextOutput::New());

  GUI* gui = new GUI();

  Fl_Window* win = gui->MakeWindow();

  gui->Initialize();

  win->show();

  return Fl::run();

}
