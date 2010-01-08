
#ifndef _SimpleGreedyFluidRegistration_txx
#define _SimpleGreedyFluidRegistration_txx

template <class TPixel, unsigned int Dimension>
SimpleGreedyFluidRegistration<TPixel, Dimension>
::SimpleGreedyFluidRegistration()
{
  m_Iterations = 20;
  m_TimeStep = 0.05;
  m_Modified = false;
}

template <class TPixel, unsigned int Dimension>
SimpleGreedyFluidRegistration<TPixel, Dimension>
::~SimpleGreedyFluidRegistration()
{
}

template <class TPixel, unsigned int Dimension>
void
SimpleGreedyFluidRegistration<TPixel, Dimension>
::SetFixedImages(const std::vector<ImagePointer>& images)
{
  m_FixedImages = images;
  m_Modified = true;
}

template <class TPixel, unsigned int Dimension>
void
SimpleGreedyFluidRegistration<TPixel, Dimension>
::SetFixedImages(const std::vector<ImagePointer>& images)
{
  m_MovingImages = images;
  m_Modified = true;
}

template <class TPixel, unsigned int Dimension>
void
SimpleGreedyFluidRegistration<TPixel, Dimension>
::Update()
{
  if (!m_Modified)
    return;

  // Initialize out = input moving
  m_OutputImages.clear();
  for (unsigned int c = 0; c < m_MovingImages.size(); c++)
  {
    dupef->SetInput(m_MovingImages[i]);
    m_OutputImages.push_back(dupef->GetOutput());
  }

  // Initialize phi(x) = x
  m_DeformationField = 
  m_DeformationField->SetPixel(ind, p);


  for (unsigned int i = 0; i < m_Iterations; i++)
    this->Step();
}

template <class TPixel, unsigned int Dimension>
void
SimpleGreedyFluidRegistration<TPixel, Dimension>
::Step()
{

  // Compute velocity field
  // v = sum_c { (fixed_c - moving_c) * grad(moving_c) }
  DeformationFieldPointer velocF = DeformationFieldType::New();
  velocf->FillBuffer(zerov);

  for (unsigned int ichan = 0; ichan < numChannels; ichan++)
  {
    // Gradient
  }

  // Apply Green's kernel to velocity field

  // Compose velocity field

  // Warp images
}

#endif
