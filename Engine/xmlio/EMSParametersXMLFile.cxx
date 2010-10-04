
#include "EMSParametersXMLFile.h"

#include "muFile.h"

#include "itksys/SystemTools.hxx"

#include <fstream>
#include <sstream>
#include <string>

#include <stdlib.h>

EMSParametersXMLFileReader
::EMSParametersXMLFileReader()
{
  m_PObject = 0;
  m_LastFile = "";
  m_LastOrient = "file";
}

EMSParametersXMLFileReader
::~EMSParametersXMLFileReader()
{

}

int
EMSParametersXMLFileReader
::CanReadFile(const char* name)
{
  if(!itksys::SystemTools::FileExists(name) ||
     itksys::SystemTools::FileIsDirectory(name) ||
     itksys::SystemTools::FileLength(name) == 0)
    return 0;
  return 1;
}

void
EMSParametersXMLFileReader
::StartElement(const char* name, const char** atts)
{
  m_CurrentString = "";

  if(itksys::SystemTools::Strucmp(name,"SEGMENTATION-PARAMETERS") == 0)
  {
    m_PObject = EMSParameters::New();
  }
  else if(itksys::SystemTools::Strucmp(name,"IMAGE") == 0)
  {
    m_LastFile = "";
    m_LastOrient = "file";
  }
}

void
EMSParametersXMLFileReader
::EndElement(const char* name)
{
  if(itksys::SystemTools::Strucmp(name,"SEGMENTATION-PARAMETERS") == 0)
  {
    m_OutputObject = &(*m_PObject);
  }
  else if(itksys::SystemTools::Strucmp(name,"SUFFIX") == 0)
  {
    m_PObject->SetSuffix(m_CurrentString);
  }
  else if(itksys::SystemTools::Strucmp(name,"ATLAS-DIRECTORY") == 0)
  {
    m_PObject->SetAtlasDirectory(m_CurrentString);
  }
  else if(itksys::SystemTools::Strucmp(name,"ATLAS-ORIENTATION") == 0)
  {
    m_PObject->SetAtlasOrientation(m_CurrentString);
  }
  else if(itksys::SystemTools::Strucmp(name,"OUTPUT-DIRECTORY") == 0)
  {
    m_PObject->SetOutputDirectory(m_CurrentString);
  }
  else if(itksys::SystemTools::Strucmp(name,"OUTPUT-FORMAT") == 0)
  {
    m_PObject->SetOutputFormat(m_CurrentString);
  }
  else if(itksys::SystemTools::Strucmp(name,"FILE") == 0)
  {
    m_LastFile = m_CurrentString;
  }
  else if(itksys::SystemTools::Strucmp(name,"ORIENTATION") == 0)
  {
    m_LastOrient = m_CurrentString;
  }
  else if(itksys::SystemTools::Strucmp(name,"IMAGE") == 0)
  {
    m_PObject->AddImage(m_LastFile, m_LastOrient);
  }
  else if(itksys::SystemTools::Strucmp(name,"FILTER-ITERATIONS") == 0)
  {
    int iter = atoi(m_CurrentString.c_str());
    if (iter < 0)
      itkExceptionMacro(<< "Error: negative #iterations for filtering");
    m_PObject->SetFilterIterations(iter);
  }
  else if(itksys::SystemTools::Strucmp(name,"FILTER-TIME-STEP") == 0)
  {
    double dt = atof(m_CurrentString.c_str());
    m_PObject->SetFilterTimeStep(dt);
  }
  else if(itksys::SystemTools::Strucmp(name,"FILTER-METHOD") == 0)
  {
    m_PObject->SetFilterMethod(m_CurrentString);
  }
  else if(itksys::SystemTools::Strucmp(name,"MAX-BIAS-DEGREE") == 0)
  {
    int degree = atoi(m_CurrentString.c_str());
    if (degree < 0)
      itkExceptionMacro(<< "Error: negative bias degree");
    m_PObject->SetMaxBiasDegree(degree);
  }
  else if(itksys::SystemTools::Strucmp(name,"PRIOR") == 0)
  {
    double p = atof(m_CurrentString.c_str());
    m_PObject->AppendPriorWeight(p);
  }
  else if(itksys::SystemTools::Strucmp(name,"DO-ATLAS-WARP") == 0)
  {
    unsigned int i = atoi(m_CurrentString.c_str());
    m_PObject->SetDoAtlasWarp(i != 0);
  }
  else if(itksys::SystemTools::Strucmp(name,"ATLAS-WARP-FLUID-ITERATIONS") == 0)
  {
    unsigned int i = atoi(m_CurrentString.c_str());
    m_PObject->SetAtlasWarpFluidIterations(i);
  }
  else if(itksys::SystemTools::Strucmp(name,"ATLAS-WARP-FLUID-MAX-STEP") == 0)
  {
    double ms = atof(m_CurrentString.c_str());
    m_PObject->SetAtlasWarpFluidMaxStep(ms);
  }
  else if(itksys::SystemTools::Strucmp(name,"ATLAS-LINEAR-MAP-TYPE") == 0)
  {
    m_PObject->SetAtlasLinearMapType(m_CurrentString);
  }
  else if(itksys::SystemTools::Strucmp(name,"IMAGE-LINEAR-MAP-TYPE") == 0)
  {
    m_PObject->SetImageLinearMapType(m_CurrentString);
  }
}

void
EMSParametersXMLFileReader
::CharacterDataHandler(const char* inData, int inLength)
{
  for (int i = 0; i < inLength; i++)
    m_CurrentString += inData[i];
}

EMSParametersXMLFileWriter
::EMSParametersXMLFileWriter()
{

}

EMSParametersXMLFileWriter
::~EMSParametersXMLFileWriter()
{

}

int
EMSParametersXMLFileWriter
::CanWriteFile(const char* name)
{
  return true;
}

// Support function
template <typename T>
static void
WriteField(EMSParametersXMLFileWriter* writer, const char* attname,
  T value, std::ofstream& output)
{
  writer->WriteStartElement(attname, output);
  output << value;
  writer->WriteEndElement(attname, output);
  output << std::endl;
}

int
EMSParametersXMLFileWriter
::WriteFile()
{
  if (m_InputObject == 0)
    itkExceptionMacro(<< "No object to write");

  if (!m_InputObject->CheckValues())
    itkExceptionMacro(<< "Invalid values");

  if (m_Filename.length() == 0)
    itkExceptionMacro(<< "No file name specified");

  std::ofstream output(m_Filename.c_str());
  if (output.fail())
    itkExceptionMacro(<< "Can not open " << m_Filename);

  // Header
  WriteStartElement("?xml version=\"1.0\"?",output);
  output << std::endl;
  WriteStartElement("!DOCTYPE SEGMENTATION-PARAMETERS",output);
  output << std::endl;

  WriteStartElement("SEGMENTATION-PARAMETERS", output);
  output << std::endl;

  EMSParameters::Pointer p = m_InputObject;

  WriteField<std::string>(this, "SUFFIX", p->GetSuffix(), output);

  WriteField<std::string>(this, "ATLAS-DIRECTORY", p->GetAtlasDirectory(), output);

  WriteField<std::string>(this, "ATLAS-ORIENTATION", p->GetAtlasOrientation(), output);

  WriteField<std::string>(this, "OUTPUT-DIRECTORY", p->GetOutputDirectory(), output);

  WriteField<std::string>(this, "OUTPUT-FORMAT", p->GetOutputFormat(), output);

  // Write the list of images
  for (unsigned int k = 0; k < p->GetImages().GetSize(); k++)
  {
    this->WriteStartElement("IMAGE", output);

    output << std::endl;
    output << "  ";
    WriteField<std::string>(this, "FILE", p->GetImages()[k], output);
    output << "  ";
    WriteField<std::string>(this, "ORIENTATION", p->GetImageOrientations()[k],
      output);

    this->WriteEndElement("IMAGE", output);
    output << std::endl;
  }

  WriteField<unsigned int>(this, "FILTER-ITERATIONS", p->GetFilterIterations(), output);

  WriteField<float>(this, "FILTER-TIME-STEP", p->GetFilterTimeStep(), output);

  WriteField<std::string>(this, "FILTER-METHOD", p->GetFilterMethod(), output);

  WriteField<unsigned int>(this, "MAX-BIAS-DEGREE", p->GetMaxBiasDegree(), output);

  std::vector<double> prWeights = p->GetPriorWeights();
  for (unsigned int i = 0; i < prWeights.size(); i++)
    WriteField<float>(this, "PRIOR", prWeights[i], output);

  WriteField<bool>(this, "DO-ATLAS-WARP", p->GetDoAtlasWarp(), output);

  WriteField<unsigned int>(this, "ATLAS-WARP-FLUID-ITERATIONS", p->GetAtlasWarpFluidIterations(), output);
  
  WriteField<float>(this, "ATLAS-WARP-FLUID-MAX-STEP", p->GetAtlasWarpFluidMaxStep(), output);

  WriteField<std::string>(this, "ATLAS-LINEAR-MAP-TYPE", p->GetAtlasLinearMapType(), output);

  WriteField<std::string>(this, "IMAGE-LINEAR-MAP-TYPE", p->GetImageLinearMapType(), output);

  // Finish
  WriteEndElement("SEGMENTATION-PARAMETERS", output);
  output << std::endl;
  output.close();

  return 0;
}

// Definition of some convenience functions
EMSParameters::Pointer
readEMSParametersXML(const char* fn)
{
  EMSParametersXMLFileReader::Pointer reader =
    EMSParametersXMLFileReader::New();
  try
  {
    reader->SetFilename(fn);
    reader->GenerateOutputInformation();
  }
  catch (...)
  {
    return 0;
  }
  return reader->GetOutputObject();
}

bool
writeEMSParametersXML(const char* fn, EMSParameters* p)
{
  if (p == 0)
    return false;
  if (!p->CheckValues())
    return false;

  EMSParametersXMLFileWriter::Pointer writer =
    EMSParametersXMLFileWriter::New();

  // Enforce XML file extension
  std::string outfn = fn;
  std::string ext = mu::get_ext(fn);
  if (ext.compare("xml") != 0)
    outfn += std::string(".xml");

  writer->SetFilename(outfn.c_str());
  writer->SetObject(p);
  writer->WriteFile();

  return true;
}

