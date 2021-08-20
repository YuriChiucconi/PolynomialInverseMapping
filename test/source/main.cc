#include "../../class/include/class.h"
#include "../include/test.h"

int main()
{
    Parameters par;
    ParameterAcceptor::initialize("parameters.prm", "used_parameters.prm");

    Test<2> my_test(par);
    my_test.run();
    
    return 0;
}
