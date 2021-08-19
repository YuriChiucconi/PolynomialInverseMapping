#ifndef __PIM
#define __PIM

//#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <iostream>
#include <memory>

using namespace dealii;

template <int dim>
class PIM : public MappingQGeneric<dim>
{

protected:
    
    const unsigned int inverse_polynomial_degree;
    
    const FE_DGQ<dim> inverse_fe_dg;
    FESystem<dim> inverse_fe;
    
    
    std::unique_ptr<DoFHandler<dim>> inverse_dh;
    Vector<double> inverse_vector;
    std::unique_ptr<MappingFEField<dim>> inverse_mapping;

 
public:
    
    PIM(int forward_degree, int inverse_degree);
    //PIM(const PIM<dim> &mapping);
    
    unsigned int get_inverse_degree();
    
    void
    initialize(const Triangulation<dim> & tria);
    
    Point<dim>
    poly_transform_real_to_unit_cell(const typename Triangulation<dim>::cell_iterator &cell
                                     , const Point<dim> &p) const;
    
    template<int> friend class Test;
};

#endif
