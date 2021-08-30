#ifndef __TEST
#define __TEST

#include <deal.II/base/parameter_acceptor.h>

#include "../../class/include/class.h"

class Parameters : public ParameterAcceptor
{
public:
  Parameters()
  {
    add_parameter("Refinement", refinements);
    add_parameter("Forward degree", forward_degree);
    add_parameter("Inverse  degree", inverse_degree);
    add_parameter("N particles per cell", particles_per_cell);
  }

  unsigned int refinements              = 3;
  unsigned int forward_degree           = 3;
  unsigned int inverse_degree           = 4;
  unsigned int particles_per_cell       = 10;
};

template <int dim>
class Test
{
    
private:
    
    MappingQGeneric<dim> newton_inverse_mapping;
    
    PIM<dim> polynomial_inverse_mapping;
    
    unsigned int refinements;
    
    unsigned int particles_per_cell;
    
    Triangulation<dim> tria;
    
    
    
public:
    
    Test(const Parameters &par);
    
    void run();
    
};

#endif
