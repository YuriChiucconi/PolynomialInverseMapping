#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <boost/range/irange.hpp>

#include "../include/class.h"


using iota = boost::integer_range<unsigned int>;
using namespace dealii;


// CONSTRUCTOR
template<int dim>
PIM<dim>::PIM(int forward_degree, int inverse_degree)
    : MappingQGeneric<dim>(forward_degree)
    , inverse_polynomial_degree(inverse_degree)
    , inverse_fe_dg(inverse_degree)
    , inverse_fe_system(inverse_fe_dg, dim) {}


template<int dim>
unsigned int
PIM<dim>::get_inverse_degree()
{
    return inverse_polynomial_degree;
}


template<int dim>
void
PIM<dim>::initialize(const Triangulation<dim> & tria)
{
   
    // Distribuisco i dof sulla triangolazione rispetto al <dim>-FE di grado inverse_degree
    inverse_dh.reset(new DoFHandler<dim>(tria));
    inverse_dh->distribute_dofs(inverse_fe_system);

    
    // inverse_dh->n_dofs() è ((inverse_degree+1)^dim)x2x(tria.n_active_cells())
    inverse_vector.reinit(inverse_dh->n_dofs());

    inverse_mapping.reset(new MappingFEField<dim>(*inverse_dh, inverse_vector));
    
    
    // inverse_support_points è un std::vector dei punti di supporto di grado inverse_degree sull'elemento di riferimento
    const auto &inverse_support_points = inverse_fe_dg.get_unit_support_points();

    // std::vector della stessa lunghezza di quello sopra : (inverse_degree+1)^dim
    // inizializzato con tutti gli elementi a zero
    std::vector<types::global_dof_index> dof_indices(inverse_fe_system.dofs_per_cell);

    // itero sulle celle attive
    for (const auto &cell : inverse_dh->active_cell_iterators())
    {
        // per ogni cella questo vettore registra il corrispondente global_index per ognuno dei dof locali
        cell->get_dof_indices(dof_indices);

        const auto box = this->get_bounding_box(cell);
        

        // mapped_support_points parte come l'insieme dei (inverse_degree+1)^dim punti di supporto su Khat
        auto mapped_support_points = inverse_support_points;
        // e viene trasformato nell'insieme degli stessi punti mappati sulla bounding box della cella
        for (auto &p : mapped_support_points)
        {
            p = box.unit_to_real(p);
        }
            
            
        // pulled_back_support_points parte come una copia di mapped_support_points
        auto pulled_back_support_points = mapped_support_points;
        // e viene trasformato nell'insieme delle retro-immagini dei punti di supporto
        for (auto &p : pulled_back_support_points)
        {
            p = this->MappingQGeneric<dim>::transform_real_to_unit_cell(cell, p);
        }
        

        // ????????????????????????????????????????????????????????????????????????
        for (auto i : iota(0, dof_indices.size()))
            {
                const auto a = inverse_fe_system.system_to_component_index(i).first;
                const auto b = inverse_fe_system.system_to_component_index(i).second;
                
                inverse_vector(dof_indices[i]) = pulled_back_support_points[b][a];
            }
        // ????????????????????????????????????????????????????????????????????????

    }
    
    // creo la std::map con gli iteratori della triangolazione sulla celle
    for(auto& cell : tria.active_cell_iterators())
    {
        BoundingBox<dim> box = this->get_bounding_box(cell);
        bounding_boxes.insert(std::pair<const typename Triangulation<dim>::cell_iterator, BoundingBox<dim>>(cell,box) );
    }
}



template<int dim>
Point<dim>
PIM<dim>::transform_real_to_unit_cell(const typename Triangulation<dim>::cell_iterator &cell
                                           , const Point<dim> & p) const
{
    BoundingBox<dim>  box   = bounding_boxes.at(cell);
    return inverse_mapping->transform_unit_to_real_cell(cell, box.real_to_unit(p));
}


template class PIM<2>;
template class PIM<3>;
