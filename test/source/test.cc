
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/particles/particle.h>
#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/data_out.h>

#include <boost/range/irange.hpp>


#include <random>


#include "../../class/include/class.h"
#include "../include/test.h"

using iota = boost::integer_range<unsigned int>;



// ------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------
template <int dim>
inline Point<dim> random_point()
{
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0);
    Point<dim> p;
    for (unsigned int i = 0; i < dim; ++i)
        p[i] = dis(gen);
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
    //tria.refine_global(refinements);
    const Point<dim> center(0.,0.);
    const double   inner_radius = 5., outer_radius = 10.;
    GridGenerator::hyper_shell(tria, center, inner_radius, outer_radius, 10);
    tria.refine_global(refinements);
    
    std::ofstream out("test_grid.vtk");
    GridOut       grid_out;
    grid_out.write_vtk(tria, out);
    std::cout << "Grid written to test_grid.vtk" << std::endl;

    Particles::ParticleHandler<dim> particle_handler(tria, newton_inverse_mapping);
    std::multimap<typename Triangulation<dim>::active_cell_iterator, Particles::Particle<dim>> particles;
    for (const auto &cell : tria.active_cell_iterators())
    {
        for (auto i : iota(0, particles_per_cell))
        {
            Particles::Particle<dim> p;
            auto loc = random_point<dim>();
            p.set_reference_location(loc);
            p.set_location(newton_inverse_mapping.transform_unit_to_real_cell(cell, loc));
            particles.insert(std::make_pair(cell, p));
        }
    }
    particle_handler.insert_particles(particles);
    
    std::ofstream   ofile("particles.vtu");
    Particles::DataOut<dim, dim> data_out;
    data_out.build_patches(particle_handler);
    data_out.write_vtu(ofile);
    std::cout << "Particles written to particles.vtu" << std::endl;
    
    /* OUTPUT per Python
    std::cout << "[";
    for(auto i = particle_handler.begin(); i != particle_handler.end(); ++i)
    {
        auto loc = i->get_location();
        std::cout << "[" << loc[0];
        for(int j=1; j<dim; j++)
        {
            std::cout << ", " << loc[j];
        }
        std::cout << "]," << std::endl;
    }
    */
    
    // NEWTON
    {
        TimerOutput::Scope timer_section(timer, "(1) Newton");
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
        TimerOutput::Scope timer_section(timer, "(2) OFF-line PIM");
        polynomial_inverse_mapping.initialize(tria);
    }
    
    // POLYNOMIAL INVERSE
    {
        TimerOutput::Scope timer_section(timer, "(3) IN-line PIM");
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
