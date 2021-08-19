//#include <deal.II/grid/tria.h>
//#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>


//#include <deal.II/base/bounding_box.h>
#include <deal.II/boost_adaptors/bounding_box.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>

#include <boost/range/irange.hpp>

#include "../include/class.h"


using iota = boost::integer_range<unsigned int>;
using namespace dealii;



// UNIT TO REAL
// ________________________________________________
// Prende in input la cella K e un punto p in Khat
// Restituisce F_K(p)

template <int spacedim, typename Number>
Point<spacedim, Number>
unit_to_real(const BoundingBox<spacedim, Number> &box,
             const Point<spacedim, Number> &      point)
{
  const auto boundary_points = box.get_boundary_points();
  auto       real            = boundary_points.first;
  const auto diag            = boundary_points.second - boundary_points.first;
  for (unsigned int d = 0; d < spacedim; ++d)
    real[d] += diag[d] * point[d];
  return real;
}

// REAL TO UNIT
// ________________________________________________
// Prende in input la cella K e un punto p in BB_K
// Restituisce F^{-1}_{BB_K}(p)
// dove F_{BB_K} è la mappa affine che porta Khat in BB_K

template <int spacedim, typename Number>
Point<spacedim, Number>
real_to_unit(const BoundingBox<spacedim, Number> &box,
             const Point<spacedim, Number> &      point)
{
  const auto boundary_points = box.get_boundary_points();
  auto       unit            = point;
  const auto diag            = boundary_points.second - boundary_points.first;
  unit -= boundary_points.first;
  for (unsigned int d = 0; d < spacedim; ++d)
    unit[d] /= diag[d];
  return unit;
}



// COSTRUTTORE
// inizializza la classe base, il grado della mappa polinomiale inversa, l'elemento finito discontinuo
template<int dim>
PIM<dim>::PIM(int forward_degree, int inverse_degree)
    : MappingQGeneric<dim>(forward_degree)
    , inverse_polynomial_degree(inverse_degree)
    , inverse_fe_dg(inverse_degree)
    , inverse_fe(inverse_fe_dg, dim) {}


// GET INVERSE DEGREE
// restituisce il grado della mappa polinomiale inversa
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
    inverse_dh->distribute_dofs(inverse_fe);
    
    // inverse_dh->n_dofs() è ((inverse_degree+1)^dim)x2x(tria.n_active_cells())
    inverse_vector.reinit(inverse_dh->n_dofs());
    
    
    inverse_mapping.reset(new MappingFEField<dim>(*inverse_dh, inverse_vector));
    
    
    // inverse_support_points è un std::vector dei punti di supporto di grado inverse_degree sull'elemento di riferimento
    const auto &inverse_support_points = inverse_fe_dg.get_unit_support_points();
    
    // std::vector della stessa lunghezza di quello sopra : (inverse_degree+1)^dim
    // inizializzato con tutti gli elementi a zero
    std::vector<types::global_dof_index> dof_indices(inverse_fe.dofs_per_cell);
  
    // vettore per le bounding box inizializzato vuoto
    std::vector<BoundingBox<dim>> boxes;
    
    
    // itero sulle celle attive
    for (const auto &cell : inverse_dh->active_cell_iterators())
    {
        // per ogni cella questo vettore registra il corrispondente global_index per ognuno dei dof locali
        cell->get_dof_indices(dof_indices);
 
        // prendo la BB della cella e la appendo in fondo a boxes
        const auto box = this->get_bounding_box(cell);
        boxes.emplace_back(box);

        
        
        // ------------------------------------------------------------------
        
        
        // mapped_support_points lo creo come una copia dei support_points su Khat
        auto mapped_support_points = inverse_support_points;
        for (auto &p : mapped_support_points)
            p = unit_to_real(box, p);
        
        // Pull them back through the cell of the triangulation on the unit cell
        auto pulled_back_support_points = mapped_support_points;
        for (auto &p : pulled_back_support_points)
            p = this->transform_real_to_unit_cell(cell, p);

        // Now set the values of the support points of the inverse
        for (auto i : iota(0, dof_indices.size()))
            {
            const auto d = inverse_fe.system_to_component_index(i).first;
            const auto j = inverse_fe.system_to_component_index(i).second;
            inverse_vector(dof_indices[i]) = pulled_back_support_points[j][d];
            }
    }
    
}


template<int dim>
Point<dim>
PIM<dim>::poly_transform_real_to_unit_cell(const typename Triangulation<dim>::cell_iterator &cell
                                           , const Point<dim> & p) const
{
    // DA SCRIVERE
}


template class PIM<2>;
template class PIM<3>;
