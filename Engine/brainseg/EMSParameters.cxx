
#include "EMSParameters.h"

#include "itksys/SystemTools.hxx"

EMSParameters
::EMSParameters()
{
  m_Suffix = "";

  m_AtlasDirectory = "";

  m_AtlasOrientation = "RAI";

  m_DoAtlasWarp = false;

  m_OutputDirectory = "";
  m_OutputFormat = "Meta";

  m_Images.Clear();
  m_ImageOrientations.Clear();

  m_FilterMethod = "Curvature flow";
  m_FilterIterations = 1;
  m_FilterTimeStep = 0.01;

  m_MaxBiasDegree = 4;

  m_AtlasWarpFluidIterations = 10;

  m_AtlasWarpFluidMaxStep = 0.5;

  //m_PriorWeights = std::vector<double>(4, 1.0);

  m_AtlasLinearMapType = "affine";
  m_ImageLinearMapType = "affine";

  m_InitialDistributionEstimator = "robust";
}

EMSParameters
::~EMSParameters()
{

}

void
EMSParameters
::AddImage(std::string s, std::string orient)
{
  m_Images.Append(s);
  m_ImageOrientations.Append(orient);
}

void
EMSParameters
::ClearImages()
{
  m_Images.Clear();
  m_ImageOrientations.Clear();
}

bool
EMSParameters
::CheckValues()
{
  if (m_Suffix.length() == 0)
    return false;

  if (m_AtlasDirectory.length() == 0)
    return false;

  if (m_OutputDirectory.length() == 0)
    return false;

  bool validFormat = false;
  if (itksys::SystemTools::Strucmp(m_OutputFormat.c_str(), "Analyze") == 0)
    validFormat = true;
  if (itksys::SystemTools::Strucmp(m_OutputFormat.c_str(), "GIPL") == 0)
    validFormat = true;
  if (itksys::SystemTools::Strucmp(m_OutputFormat.c_str(), "Nrrd") == 0)
    validFormat = true;
  if (itksys::SystemTools::Strucmp(m_OutputFormat.c_str(), "Meta") == 0)
    validFormat = true;

  if (!validFormat)
    return false;

  if (m_Images.GetSize() == 0)
    return false;

  return true;
}

void
EMSParameters
::PrintSelf(std::ostream& os)
{
  os << "Suffix = " << m_Suffix << std::endl;
  os << "Atlas directory = " << m_AtlasDirectory << std::endl;
  os << "Atlas orientation = " << m_AtlasOrientation << std::endl;
  os << "Output directory = " << m_OutputDirectory << std::endl;
  os << "Output format = " << m_OutputFormat << std::endl;
  os << "Images:" << std::endl;
  for (unsigned int k = 0; k < m_Images.GetSize(); k++)
    os << "  " << m_Images[k] << " --- " << m_ImageOrientations[k] << std::endl;
  os << "Filter iterations = " << m_FilterIterations << std::endl;
  os << "Filter time step = " << m_FilterTimeStep << std::endl;
  os << "Max bias degree = " << m_MaxBiasDegree << std::endl;
  for (unsigned int i = 0; i < m_PriorWeights.size(); i++)
    os << "Prior " << i+1 << " = " << m_PriorWeights[i] << std::endl;
  os << "Initial Distribution Estimator = " << m_InitialDistributionEstimator << std::endl;
  if (m_DoAtlasWarp)
  {
    os << "Fluid atlas warping, with:" << std::endl;
    os << " - iterations = " << m_AtlasWarpFluidIterations << std::endl;
    os << " - max step" << m_AtlasWarpFluidMaxStep << std::endl;
  }
  else
  {
    os << "No atlas warping..." << std::endl;
  }
}
