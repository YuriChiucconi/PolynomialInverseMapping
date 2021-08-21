
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/particles/particle_handler.h>

#include <boost/range/irange.hpp>





#include "../../class/include/class.h"
#include "../include/test.h"


using iota = boost::integer_range<unsigned int>;



// ------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------
template <int dim>
inline Point<dim> random_point(const int seed)
{
  srand(seed);
  Point<dim> p;
  for (unsigned int i = 0; i < dim; ++i)
    p[i] = rand()/(double)RAND_MAX;
  return p;
}
// ------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------


/*
template<int dim>
Test<dim>::Test(unsigned int forward_degree, unsigned int inverse_degree, unsigned int refinements, unsigned int particles_per_cell)
                : newton_inverse_mapping(forward_degree)
                , polynomial_inverse_mapping(forward_degree, inverse_degree)
                , refinements(refinements)
                , particles_per_cell(particles_per_cell) {}
 */

template<int dim>
Test<dim>::Test(const Parameters &par)
                : newton_inverse_mapping(par.forward_degree)
                , polynomial_inverse_mapping(par.forward_degree, par.inverse_degree)
                , refinements(par.refinements)
                , particles_per_cell(par.particles_per_cell) {}

template<int dim>
void Test<dim>::run()
{
    TimerOutput timer(std::cout
                      , TimerOutput::summary
                      , TimerOutput::cpu_and_wall_times);
    
    
    // Construction of the triangulation
    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
    {
        TimerOutput::Scope timer_section(timer, "(1) Triangulation construction");
        GridGenerator::hyper_shell(tria, Point<dim>(), 1.0, 2.0);
        tria.refine_global(refinements);
    }
    
    

    Particles::ParticleHandler<dim> particle_handler(tria, newton_inverse_mapping);
    // Generation of the particles
    {
        TimerOutput::Scope timer_section(timer, "(2) Random particle generation");
        
        
        std::multimap<typename Triangulation<dim>::active_cell_iterator, Particles::Particle<dim>> particles;
        
        for (const auto &cell : tria.active_cell_iterators())
        {
            for (auto i : iota(0, particles_per_cell))
            {
              Particles::Particle<dim> p;
              auto loc = random_point<dim>(i);
              p.set_reference_location(loc);
              p.set_location(newton_inverse_mapping.transform_unit_to_real_cell(cell, loc));
              particles.insert(std::make_pair(cell, p));
            }
        }
        particle_handler.insert_particles(particles);
    }
    
    
    
    // NEWTON
    {
        TimerOutput::Scope timer_section(timer, "(4) Newton");
        double newton_avg_error = 0;
        for (auto particle : particle_handler)
        {
            const auto &cell = particle.get_surrounding_cell(tria);
            const auto ref_p = newton_inverse_mapping.transform_real_to_unit_cell(cell, particle.get_location());
            newton_avg_error += ref_p.distance(particle.get_reference_location());
        }
        newton_avg_error /= particle_handler.n_global_particles();
        std::cout << "NEWTON average error:\t" << newton_avg_error << std::endl;
    }
    
    
    
    {
        TimerOutput::Scope timer_section(timer, "(5) Inverse map initilization");
        polynomial_inverse_mapping.initialize(tria);
    }
    
    // POLYNOMIAL INVERSE
    {
        TimerOutput::Scope timer_section(timer, "(6) Polynomial Inverse");
        double poly_avg_error = 0;
        for (auto particle : particle_handler)
        {
            const auto &cell = particle.get_surrounding_cell(tria);
            const auto  ref_p = polynomial_inverse_mapping.transform_real_to_unit_cell(cell, particle.get_location());
            poly_avg_error += ref_p.distance(particle.get_reference_location());
        }
        poly_avg_error /= particle_handler.n_global_particles();
        std::cout << "POLYNOMIAL average error:\t" << poly_avg_error << std::endl;
    }
    
}

template class Test<2>;
template class Test<3>;
