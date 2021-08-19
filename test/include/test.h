#ifndef __TEST
#define __TEST

#include "../../class/include/class.h"

template <int dim>
class Test
{
    
private:
    
    PIM<dim> mapping;
    
    unsigned int refinements;
    
    unsigned int particles_per_cell;
    
    Triangulation<dim> tria;
    
public:
    
    Test(unsigned int forward_degree, unsigned int inverse_degree, unsigned int refinements, unsigned int particles_per_cell);
    
    void run();
    
};

#endif
