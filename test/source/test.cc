#include <map>
#include <deal.II/particles/particle_handler.h>

#include <boost/range/irange.hpp>

#include "../../class/include/class.h"
#include "../include/test.h"


using iota = boost::integer_range<unsigned int>;


// ------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------
template <int dim>
inline Point<dim> random_point()
{
  Point<dim> p;
  for (unsigned int i = 0; i < dim; ++i)
    p[i] = rand()/RAND_MAX;
  return p;
}
// ------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------



template<int dim>
Test<dim>::Test(unsigned int forward_degree, unsigned int inverse_degree, unsigned int refinements, unsigned int particles_per_cell)
                : mapping(forward_degree, inverse_degree)
                , refinements(refinements)
                , particles_per_cell(particles_per_cell) {}

template<int dim>
void Test<dim>::run()
{
    // Generate triangulation
    GridGenerator::hyper_shell(tria, Point<dim>(), 2.0, 5.0);
    tria.refine_global(2);
    
    // Initialize PIM
    mapping.initialize(tria);
    
    // Generate particles
    Particles::ParticleHandler<dim> particle_handler(tria, mapping);
    
    std::multimap<typename Triangulation<dim>::active_cell_iterator, Particles::Particle<dim>> particles;
    
    for (const auto &cell : mapping.inverse_dh->active_cell_iterators())
      for (auto i : iota(0, particles_per_cell))
        {
          Particles::Particle<dim> p;
          auto loc = random_point<dim>();
          p.set_reference_location(loc);
          p.set_location(mapping.transform_unit_to_real_cell(cell, loc));
          particles.insert(std::make_pair(cell, p));
        }
    particle_handler.insert_particles(particles);
    
    
    // NEWTON
    
    double newton_avg_error = 0;
    for (auto particle : particle_handler)
    {
        const auto &cell = particle.get_surrounding_cell(tria);
        const auto  ref_p = mapping.transform_real_to_unit_cell(cell, particle.get_location());
        newton_avg_error += ref_p.distance(particle.get_reference_location());
    }
    newton_avg_error /= particle_handler.n_global_particles();
    
    
    std::cout << "NEWTON:\t" << newton_avg_error << std::endl;
    
    // POLYNOMIAL INVERSE
    
    double poly_avg_error = 0;
    for (auto particle : particle_handler)
    {
        const auto &cell = particle.get_surrounding_cell(tria);
        const auto  ref_p = mapping.poly_transform_real_to_unit_cell(cell, particle.get_location());
        poly_avg_error += ref_p.distance(particle.get_reference_location());
    }
    poly_avg_error /= particle_handler.n_global_particles();
    
    
    std::cout << "POLYNOMIAL:\t" << newton_avg_error << std::endl;
}

template class Test<2>;
template class Test<3>;
