#include "../../class/include/class.h"
#include "../include/test.h"
#include "../include/test_mpi.h"

int main(int argc, char **argv)
{
    Parameters par;
    ParameterAcceptor::initialize("parameters.prm", "used_parameters.prm");

    Utilities::MPI::MPI_InitFinalize init(argc, argv);
    mpi_initlog();
    
    Test<2> my_test(par);
    my_test.run();
    
    return 0;
}
